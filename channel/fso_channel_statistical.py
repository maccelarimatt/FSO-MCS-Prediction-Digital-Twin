"""
FSO Channel Simulator
=====================
Generates temporally correlated irradiance time-series following the
Gamma-Gamma distribution, with temporal statistics governed by the
frozen-flow hypothesis.

Method:
  1. Generate correlated Gaussian noise with the turbulence temporal
     power spectral density (Kolmogorov-based).
  2. Transform marginals to Gamma-Gamma via inverse CDF.
  
  This approach preserves both the correct marginal distribution
  AND physically motivated temporal correlations.

References:
  - Andrews & Phillips, "Laser Beam Propagation through Random Media," 2005.
  - Churnside, "Aperture averaging of optical scintillations in the
    turbulent atmosphere," Appl. Opt., 1991.
"""

import numpy as np
from scipy import signal
from scipy.special import gammainc, gamma as gamma_func
from scipy.stats import gamma as gamma_dist, norm
from scipy.interpolate import interp1d
from typing import Optional, Tuple
from .turbulence import TurbulenceParameters


class FSOChannel:
    """
    Simulates the FSO atmospheric channel for a given turbulence regime.
    
    The channel output h(t) represents the normalised irradiance fluctuation
    (mean = 1), which multiplies the mean received optical power:
        P_rx(t) = P_rx_mean * h(t)
    
    This corresponds to h(t) in equation (1) of Dindar et al.
    """
    
    def __init__(
        self,
        turb_params: TurbulenceParameters,
        wind_speed_m_s: float = 5.0,
        sample_rate_Hz: float = 10_000.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            turb_params: Pre-computed turbulence parameters.
            wind_speed_m_s: Transverse wind speed [m/s].
            sample_rate_Hz: Output sample rate [Hz].
            seed: Random seed for reproducibility.
        """
        self.turb = turb_params
        self.wind_speed = wind_speed_m_s
        self.fs = sample_rate_Hz
        self.rng = np.random.default_rng(seed)
        
        # Pre-compute the inverse CDF lookup table for Gamma-Gamma transform
        self._build_gamma_gamma_icdf()
    
    def _build_gamma_gamma_icdf(self, n_points: int = 10_000):
        """
        Build a lookup table for the Gamma-Gamma inverse CDF.
        
        The Gamma-Gamma PDF is a product of two Gamma distributions:
          h = X * Y,  where X ~ Gamma(α, 1/α),  Y ~ Gamma(β, 1/β)
        
        We build the CDF numerically via Monte Carlo and invert it.
        This is more robust than trying to evaluate the modified Bessel
        function K_{α-β} at extreme arguments.
        """
        alpha = self.turb.alpha
        beta = self.turb.beta
        
        # Generate a large sample to build empirical CDF
        n_mc = 500_000
        X = self.rng.gamma(alpha, 1.0 / alpha, size=n_mc)
        Y = self.rng.gamma(beta, 1.0 / beta, size=n_mc)
        h_samples = X * Y  # Gamma-Gamma distributed, E[h] = 1
        
        # Sort to build empirical CDF
        h_sorted = np.sort(h_samples)
        cdf_vals = np.linspace(0, 1, n_mc, endpoint=False) + 0.5 / n_mc
        
        # Subsample for interpolation efficiency
        indices = np.linspace(0, n_mc - 1, n_points, dtype=int)
        self._icdf_cdf = cdf_vals[indices]
        self._icdf_h = h_sorted[indices]
        
        # Build interpolator: maps uniform(0,1) -> Gamma-Gamma
        self._icdf_interp = interp1d(
            self._icdf_cdf, self._icdf_h,
            kind='linear', bounds_error=False,
            fill_value=(self._icdf_h[0], self._icdf_h[-1])
        )
    
    def _gamma_gamma_from_gaussian(self, z: np.ndarray) -> np.ndarray:
        """
        Transform standard Gaussian samples to Gamma-Gamma marginals
        using the probability integral transform:
          h = F_GG^{-1}(Φ(z))
        where Φ is the standard normal CDF.
        """
        u = norm.cdf(z)  # Gaussian -> Uniform(0,1)
        # Clip to avoid numerical issues at boundaries
        u = np.clip(u, 1e-7, 1 - 1e-7)
        return self._icdf_interp(u)
    
    def _generate_correlated_gaussian(
        self, n_samples: int
    ) -> np.ndarray:
        """
        Generate Gaussian noise with the frozen-flow temporal PSD.
        
        The temporal covariance of intensity fluctuations for a point
        receiver (plane wave, Kolmogorov) approximately follows:
        
          B_I(τ) ∝ exp(-(τ/τ_0)^{5/3})
        
        We implement this by filtering white Gaussian noise with a 
        filter whose magnitude response matches √(S(f)), where S(f)
        is the Fourier transform of B_I(τ).
        
        For computational efficiency, we use an AR(1) approximation
        matched to the coherence time, which captures the essential
        temporal correlation without the cost of full spectral shaping.
        The AR(1) model is:
          z[n] = a * z[n-1] + sqrt(1-a²) * w[n],  w[n] ~ N(0,1)
        where a = exp(-dt/τ_0).
        """
        tau0 = self.turb.coherence_time(self.wind_speed)
        dt = 1.0 / self.fs
        
        # AR(1) coefficient
        a = np.exp(-dt / tau0)
        
        # Generate correlated Gaussian process
        z = np.zeros(n_samples)
        w = self.rng.standard_normal(n_samples)
        z[0] = w[0]
        innovation_std = np.sqrt(1 - a**2)
        
        for i in range(1, n_samples):
            z[i] = a * z[i - 1] + innovation_std * w[i]
        
        return z
    
    def generate(
        self,
        duration_s: float,
        mean_rx_power_dBm: float = -10.0,
        include_pointing_error: bool = True,
        pointing_std_urad: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a time-series of received optical power.
        
        Args:
            duration_s: Duration in seconds.
            mean_rx_power_dBm: Mean received optical power [dBm]
                              (after geometric loss, before turbulence).
            include_pointing_error: Whether to add slow pointing drift.
            pointing_std_urad: RMS pointing jitter [µrad].
            
        Returns:
            time_s: Time vector [s].
            power_mW: Received optical power [mW] (the observable).
            metadata: Dict with SI, turbulence params, etc.
        """
        n_samples = int(duration_s * self.fs)
        time_s = np.arange(n_samples) / self.fs
        
        # --- Turbulence-induced scintillation ---
        z_corr = self._generate_correlated_gaussian(n_samples)
        h_turb = self._gamma_gamma_from_gaussian(z_corr)
        
        # --- Pointing error (slow drift + jitter) ---
        if include_pointing_error:
            h_point = self._generate_pointing_error(
                n_samples, pointing_std_urad
            )
        else:
            h_point = np.ones(n_samples)
        
        # --- Combined channel ---
        h_total = h_turb * h_point
        
        # Convert mean power to linear
        mean_power_mW = 10 ** (mean_rx_power_dBm / 10)
        power_mW = mean_power_mW * h_total
        
        # Clip at receiver limits (from link budget)
        # Below sensitivity: essentially noise floor
        # Above saturation: clipped
        power_mW = np.maximum(power_mW, 1e-6)  # noise floor ~-30 dBm
        
        # Compute actual scintillation index from the generated data
        si_actual = np.var(h_total) / (np.mean(h_total) ** 2)
        
        metadata = {
            "turb_params": self.turb,
            "wind_speed_m_s": self.wind_speed,
            "mean_rx_power_dBm": mean_rx_power_dBm,
            "scintillation_index_actual": si_actual,
            "scintillation_index_theoretical": self.turb.scintillation_index,
            "coherence_time_ms": self.turb.coherence_time(self.wind_speed) * 1e3,
            "rytov_variance": self.turb.rytov_variance,
            "n_samples": n_samples,
            "sample_rate_Hz": self.fs,
        }
        
        return time_s, power_mW, metadata
    
    def _generate_pointing_error(
        self, n_samples: int, std_urad: float
    ) -> np.ndarray:
        """
        Model pointing error as a slow random walk + faster jitter.
        
        The paper notes that mount settling and thermal drift cause
        slow misalignment, while wind causes faster "pole wobble."
        
        We model this as:
          - Slow drift: Brownian motion with ~minute timescale
          - Fast jitter: AR(1) with ~second timescale
        
        The pointing error reduces power as a Gaussian beam model:
          h_point = exp(-2 * θ² / θ_beam²)
        where θ is the angular displacement.
        """
        dt = 1.0 / self.fs
        std_rad = std_urad * 1e-6
        
        # Beam divergence (half-angle)
        theta_beam = self.turb.wavelength_m / (
            np.pi * 0.015  # ~30 mm beam waist
        )
        
        # Slow drift (random walk, ~100s correlation)
        drift_tau = 100.0
        drift_a = np.exp(-dt / drift_tau)
        drift = np.zeros(n_samples)
        w_drift = self.rng.standard_normal(n_samples) * std_rad * 0.3
        for i in range(1, n_samples):
            drift[i] = drift_a * drift[i-1] + np.sqrt(1 - drift_a**2) * w_drift[i]
        
        # Fast jitter (AR(1), ~1s correlation)
        jitter_tau = 1.0
        jitter_a = np.exp(-dt / jitter_tau)
        jitter = np.zeros(n_samples)
        w_jitter = self.rng.standard_normal(n_samples) * std_rad * 0.7
        for i in range(1, n_samples):
            jitter[i] = jitter_a * jitter[i-1] + np.sqrt(1 - jitter_a**2) * w_jitter[i]
        
        theta_total = drift + jitter
        
        # Gaussian beam pointing loss model
        h_point = np.exp(-2 * theta_total**2 / theta_beam**2)
        
        return h_point
    
    def generate_transitioning(
        self,
        turb_params_end: TurbulenceParameters,
        wind_speed_end_m_s: float,
        duration_s: float,
        mean_rx_power_dBm: float = -10.0,
        transition_type: str = "linear",
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a time-series with smoothly transitioning turbulence 
        conditions (e.g., simulating sunrise/sunset or weather changes).
        
        This addresses the concept note's requirement for "Transitioning"
        regime datasets.
        
        Args:
            turb_params_end: Turbulence parameters at end of transition.
            wind_speed_end_m_s: Wind speed at end.
            duration_s: Duration in seconds.
            mean_rx_power_dBm: Mean received power.
            transition_type: "linear" or "sigmoid".
            
        Returns:
            Same as generate().
        """
        n_samples = int(duration_s * self.fs)
        time_s = np.arange(n_samples) / self.fs
        
        # Create blending weight
        t_norm = time_s / duration_s
        if transition_type == "sigmoid":
            weight = 1 / (1 + np.exp(-10 * (t_norm - 0.5)))
        else:  # linear
            weight = t_norm
        
        # Generate samples from both regimes
        channel_start = FSOChannel(
            self.turb, self.wind_speed, self.fs, 
            seed=self.rng.integers(0, 2**31)
        )
        channel_end = FSOChannel(
            turb_params_end, wind_speed_end_m_s, self.fs,
            seed=self.rng.integers(0, 2**31)
        )
        
        _, power_start, _ = channel_start.generate(
            duration_s, mean_rx_power_dBm, include_pointing_error=False
        )
        _, power_end, _ = channel_end.generate(
            duration_s, mean_rx_power_dBm, include_pointing_error=False
        )
        
        # Blend (in dB domain for more physical interpolation)
        power_start_dB = 10 * np.log10(np.maximum(power_start, 1e-10))
        power_end_dB = 10 * np.log10(np.maximum(power_end, 1e-10))
        power_blend_dB = (1 - weight) * power_start_dB + weight * power_end_dB
        power_mW = 10 ** (power_blend_dB / 10)
        
        metadata = {
            "type": "transitioning",
            "start_regime": self.turb.Cn2,
            "end_regime": turb_params_end.Cn2,
            "transition_type": transition_type,
            "n_samples": n_samples,
            "sample_rate_Hz": self.fs,
        }
        
        return time_s, power_mW, metadata
