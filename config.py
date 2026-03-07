"""
Configuration for FSO-TVWS MCS Prediction Simulation
=====================================================
System parameters extracted from:
  Dindar et al., "Hybrid Amplify and Forward TV Whitespace Radio 
  over Free-Space Optics," J. Lightwave Technol., 202X.

All physical constants and link-budget values are referenced to the
experimental setup described in Section III of that paper.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
BOLTZMANN_K = 1.381e-23        # J/K
ELECTRON_Q = 1.602e-19         # C
PLANCK_H = 6.626e-34           # J·s
SPEED_OF_LIGHT = 3e8           # m/s


# ---------------------------------------------------------------------------
# MCS Table  (Table I, Dindar et al.)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MCSEntry:
    index: int
    modulation: str          # e.g. "QPSK", "16QAM"
    bits_per_symbol: int     # log2(M)
    code_rate: float
    required_snr_dB: float   # minimum SNR for BER < 1e-3

MCS_TABLE: List[MCSEntry] = [
    MCSEntry(0,  "QPSK",   2, 1/2, 7.0),
    MCSEntry(1,  "QPSK",   2, 2/3, 8.0),
    MCSEntry(2,  "QPSK",   2, 3/4, 8.5),
    MCSEntry(3,  "QPSK",   2, 5/6, 9.5),
    MCSEntry(4,  "16QAM",  4, 1/2, 13.5),
    MCSEntry(5,  "16QAM",  4, 2/3, 14.5),
    MCSEntry(6,  "16QAM",  4, 3/4, 15.5),
    MCSEntry(7,  "16QAM",  4, 5/6, 16.0),
    MCSEntry(8,  "64QAM",  6, 2/3, 20.0),
    MCSEntry(9,  "64QAM",  6, 3/4, 21.0),
    MCSEntry(10, "64QAM",  6, 5/6, 21.5),
    MCSEntry(11, "256QAM", 8, 2/3, 27.0),
    MCSEntry(12, "256QAM", 8, 3/4, 28.0),
    MCSEntry(13, "256QAM", 8, 5/6, 29.0),
]

# Number of classes = 14 MCS levels + 1 "outage" class (SNR too low for MCS 0)
NUM_MCS_CLASSES = len(MCS_TABLE) + 1   # 15 total (class 0 = outage)


# ---------------------------------------------------------------------------
# FSO Link Budget  (Section III, Dindar et al.)
# ---------------------------------------------------------------------------
@dataclass
class LinkBudget:
    """Parameters of the 800 m folded-path FSO link."""
    wavelength_m: float = 1550e-9          # 1550 nm telecom laser
    tx_power_mW: float = 8.0               # 8 mW (9 dBm)
    link_distance_m: float = 800.0         # 800 m folded path
    lens_diameter_m: float = 0.075         # 75 mm plano-convex
    lens_focal_length_m: float = 0.150     # 150 mm focal length
    geometric_loss_dB: float = 12.0        # from beam spreading
    peak_rx_power_dBm: float = -3.0        # 0.5 mW at best alignment
    min_rx_power_dBm: float = -27.0        # link failure threshold
    saturation_power_dBm: float = -3.0     # receiver saturation
    fade_margin_dB: float = 24.0           # best-case
    # Photodiode parameters (custom receiver, Sec III)
    responsivity_A_per_W: float = 0.9      # typical InGaAs at 1550 nm
    bandwidth_Hz: float = 300e6            # RF output 15 kHz–300 MHz
    noise_figure_dB: float = 5.0           # estimated receiver NF

    @property
    def beam_divergence_rad(self) -> float:
        """Half-angle divergence from lens geometry."""
        return self.wavelength_m / (np.pi * self.lens_diameter_m / 2)

    @property
    def peak_rx_power_mW(self) -> float:
        return 10 ** (self.peak_rx_power_dBm / 10)

    @property
    def min_rx_power_mW(self) -> float:
        return 10 ** (self.min_rx_power_dBm / 10)


# ---------------------------------------------------------------------------
# SNR Model
# ---------------------------------------------------------------------------
@dataclass
class SNRModelParams:
    """
    Empirical linear SNR model calibrated to Dindar et al. results.
    
    From the paper's data (Figures 4-6):
      - At Pr = 0.5 mW  (-3 dBm):  SNR ≈ 30 dB  (supports MCS 13)
      - At Pr = 0.002 mW (-27 dBm): SNR ≈ 5 dB   (below MCS 0 threshold)
    
    Linear fit: SNR_dB ≈ slope * Pr_dBm + intercept
    Solving:  slope*(−3) + intercept = 30
              slope*(−27) + intercept = 5
    Gives:    slope ≈ 1.04,  intercept ≈ 33.1
    
    This ~1:1 dB relationship confirms thermal-noise-limited detection,
    consistent with SIM/DD at these power levels.
    """
    slope: float = 1.04
    intercept_dB: float = 33.1
    # Additional noise floor for hybrid RF stage (Section IV-B)
    rf_penalty_dB: float = 17.0   # mixer + RF propagation loss


# ---------------------------------------------------------------------------
# Turbulence Regimes
# ---------------------------------------------------------------------------
@dataclass
class TurbulenceRegime:
    """
    Atmospheric turbulence parameterization.
    
    Cn2: refractive index structure parameter [m^{-2/3}]
    Typical values (from concept note & literature):
      Weak:        Cn2 ≈ 1e-16 to 5e-16
      Moderate:    Cn2 ≈ 1e-15 to 5e-15
      Strong:      Cn2 ≈ 1e-14 to 1e-13
      Very Strong: Cn2 ≈ 1e-13 to 1e-12
    """
    name: str
    Cn2: float                    # m^{-2/3}
    wind_speed_m_s: float = 5.0   # transverse wind speed
    wind_speed_std: float = 1.0   # wind speed variation (for realism)

# Pre-defined regimes for dataset generation
TURBULENCE_REGIMES = {
    "weak":       TurbulenceRegime("weak",       Cn2=5e-16, wind_speed_m_s=2.0, wind_speed_std=0.5),
    "moderate":   TurbulenceRegime("moderate",    Cn2=5e-15, wind_speed_m_s=5.0, wind_speed_std=1.5),
    "strong":     TurbulenceRegime("strong",      Cn2=5e-14, wind_speed_m_s=8.0, wind_speed_std=2.0),
    "very_strong":TurbulenceRegime("very_strong", Cn2=5e-13, wind_speed_m_s=10.0, wind_speed_std=3.0),
}


# ---------------------------------------------------------------------------
# Dataset / Training Configuration
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Controls how the time-series dataset is generated and windowed."""
    sample_rate_Hz: float = 10_000.0   # Channel sampling rate (10 kHz)
                                        # matches the DC monitor output 
                                        # bandwidth (DC–15 kHz) from paper
    duration_s: float = 300.0           # 5 minutes per regime segment
    window_size: int = 100              # lookback window (samples)
    prediction_horizon: int = 1         # predict MCS this many steps ahead
    window_stride: int = 50             # step between consecutive windows
                                        # stride=1 uses every sample (OOM on long runs)
                                        # stride=50 = 5 ms spacing at 10 kHz
                                        # still well below coherence time for 
                                        # moderate turbulence (τ₀ ≈ 8 ms)
    si_window_size: int = 500           # samples for SI estimation
                                        # (~50 ms at 10 kHz, balancing 
                                        # latency vs estimation accuracy)
    include_si_context: bool = True     # feed SI as auxiliary input
    include_power_derivative: bool = True  # feed rate-of-change feature
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # For transitioning regimes (sunrise/sunset simulation)
    transition_duration_s: float = 60.0  # seconds to transition between regimes


# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15              # early stopping patience
    scheduler_factor: float = 0.5   # LR reduction factor
    scheduler_patience: int = 7
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"            # "auto", "cpu", "cuda"


# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------
@dataclass
class LSTMConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

@dataclass
class GRUConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

@dataclass  
class TransformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

@dataclass
class ESNConfig:
    """Echo State Network (reservoir computing)."""
    reservoir_size: int = 500
    spectral_radius: float = 0.95
    input_scaling: float = 0.5
    leaking_rate: float = 0.3
    sparsity: float = 0.9         # fraction of zero weights
    regularization: float = 1e-6  # ridge regression lambda

@dataclass
class TCNConfig:
    """Temporal Convolutional Network."""
    num_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    kernel_size: int = 7
    dropout: float = 0.2
