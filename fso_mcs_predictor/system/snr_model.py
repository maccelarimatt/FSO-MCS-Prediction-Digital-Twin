"""
SNR Model and MCS Selection
============================
Maps received optical power to electrical SNR, then selects the
optimal MCS level based on the thresholds from Table I (Dindar et al.).

The "ideal MCS" is the highest-rate scheme whose required SNR is
satisfied by the instantaneous channel. This is the ground truth
label for training the neural network predictor.
"""

import numpy as np
from typing import Optional
from ..config import (
    MCS_TABLE, MCSEntry, NUM_MCS_CLASSES, 
    SNRModelParams, LinkBudget
)


class SNRModel:
    """
    Convert received optical power [mW] to electrical SNR [dB].
    
    Uses the empirical linear model calibrated from the Dindar et al.
    experimental data (see SNRModelParams docstring for derivation).
    """
    
    def __init__(
        self,
        params: Optional[SNRModelParams] = None,
        include_rf_penalty: bool = False,
    ):
        """
        Args:
            params: SNR model parameters. Uses defaults if None.
            include_rf_penalty: If True, subtract the 17 dB hybrid 
                               RF penalty (Section IV-B of paper).
        """
        self.params = params or SNRModelParams()
        self.rf_penalty = self.params.rf_penalty_dB if include_rf_penalty else 0.0
    
    def power_to_snr(self, power_mW: np.ndarray) -> np.ndarray:
        """
        Map received optical power to electrical SNR.
        
        Args:
            power_mW: Received optical power in milliwatts.
            
        Returns:
            snr_dB: Electrical signal-to-noise ratio in dB.
        """
        # Convert to dBm (clip to avoid log(0))
        power_dBm = 10 * np.log10(np.maximum(power_mW, 1e-10))
        
        # Linear model: SNR_dB = slope * P_dBm + intercept
        snr_dB = (
            self.params.slope * power_dBm 
            + self.params.intercept_dB 
            - self.rf_penalty
        )
        
        # SNR cannot be negative in practice (noise floor)
        return np.maximum(snr_dB, 0.0)


class MCSSelector:
    """
    Selects the ideal MCS level given the instantaneous SNR.
    
    Class mapping:
      - Class 0:  Outage (SNR < MCS 0 threshold)
      - Class 1:  MCS 0  (QPSK, rate 1/2)
      - Class 2:  MCS 1  (QPSK, rate 2/3)
      - ...
      - Class 14: MCS 13 (256QAM, rate 5/6)
    
    The "ideal" MCS is the highest-rate scheme that the channel supports.
    A hysteresis margin can be added for more conservative selection
    (matching the real system's behaviour, Section II-A of the paper).
    """
    
    def __init__(self, hysteresis_dB: float = 0.0):
        """
        Args:
            hysteresis_dB: SNR margin above the threshold required 
                          to select a higher MCS. Set to 0 for ideal
                          (genie-aided) selection; use ~1-2 dB for 
                          realistic conservative selection.
        """
        self.hysteresis_dB = hysteresis_dB
        self.mcs_table = MCS_TABLE
        # Pre-sort by required SNR (should already be sorted, but be safe)
        self.snr_thresholds = np.array([
            m.required_snr_dB for m in self.mcs_table
        ])
    
    def select(self, snr_dB: np.ndarray) -> np.ndarray:
        """
        Select ideal MCS for each SNR value.
        
        Args:
            snr_dB: Array of SNR values [dB].
            
        Returns:
            mcs_class: Array of MCS class indices (0 = outage, 1-14 = MCS 0-13).
        """
        snr_dB = np.atleast_1d(snr_dB)
        mcs_class = np.zeros(len(snr_dB), dtype=np.int64)
        
        for i, threshold in enumerate(self.snr_thresholds):
            # Class i+1 corresponds to MCS index i
            mask = snr_dB >= (threshold + self.hysteresis_dB)
            mcs_class[mask] = i + 1
        
        return mcs_class
    
    def mcs_to_throughput(self, mcs_class: np.ndarray, bandwidth_Hz: float = 6e6) -> np.ndarray:
        """
        Estimate effective throughput for a given MCS selection.
        
        Uses a simplified model: throughput = bits_per_symbol * code_rate * bandwidth
        This is approximate but sufficient for comparing prediction strategies.
        
        Args:
            mcs_class: MCS class indices (0 = outage).
            bandwidth_Hz: Channel bandwidth (6 MHz for TVWS).
            
        Returns:
            throughput_bps: Throughput in bits per second.
        """
        throughput = np.zeros_like(mcs_class, dtype=np.float64)
        
        for i, mcs in enumerate(self.mcs_table):
            mask = mcs_class == (i + 1)
            # Spectral efficiency * bandwidth
            throughput[mask] = mcs.bits_per_symbol * mcs.code_rate * bandwidth_Hz
        
        # Outage class (0) gets zero throughput
        return throughput
    
    @staticmethod
    def class_to_mcs_name(cls: int) -> str:
        """Convert class index to human-readable MCS name."""
        if cls == 0:
            return "OUTAGE"
        mcs = MCS_TABLE[cls - 1]
        return f"MCS{mcs.index} ({mcs.modulation} R={mcs.code_rate:.2f})"
