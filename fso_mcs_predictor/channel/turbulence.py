"""
Atmospheric Turbulence Physics
==============================
Computes turbulence parameters for the FSO channel model.

Supports both Kolmogorov and Von Kármán spectrum models.
Von Kármán is the default and recommended choice because it
includes the finite outer scale L0, which prevents the unphysical
divergence of low-frequency (large-eddy) contributions that
Kolmogorov exhibits. For ground-level FSO links where L0 is
comparable to the beam footprint, this correction is significant.

The aperture averaging factor is also included, which accounts for
the spatial integration of scintillation across the receiver aperture.

Key references:
  - Andrews & Phillips, "Laser Beam Propagation through Random Media,"
    SPIE Press, 2nd ed., 2005. (Chapters 8-10)
  - Cox et al., "Structured Light in Turbulence," IEEE JSTQE, 2021.
  - Trichili et al., "Roadmap to Free Space Optics," JOSA B, 2020.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TurbulenceParameters:
    """
    Computed turbulence parameters for a specific link configuration.
    
    Supports Von Karman spectrum (default) which includes finite outer
    scale correction, or pure Kolmogorov as a limiting case.
    """
    Cn2: float               # Refractive index structure parameter [m^{-2/3}]
    wavelength_m: float      # Operating wavelength [m]
    link_distance_m: float   # Propagation distance [m]
    
    # Von Karman parameters (set outer_scale_m to np.inf for Kolmogorov)
    outer_scale_m: float = 25.0    # L0: largest eddy size [m]
                                    # Typical ground-level: 5-50 m
    aperture_diameter_m: float = 0.075  # Receiver aperture [m]
                                         # 75 mm matches Dindar et al.
    
    # Derived quantities (computed in __post_init__)
    k: float = 0.0                 # Optical wavenumber [rad/m]
    rytov_variance: float = 0.0    # Kolmogorov Rytov variance
    rytov_variance_vk: float = 0.0 # Von Karman corrected Rytov variance
    scintillation_index: float = 0.0  # After aperture averaging
    r0: float = 0.0                # Fried parameter [m]
    alpha: float = 0.0             # Gamma-Gamma large-scale parameter
    beta: float = 0.0              # Gamma-Gamma small-scale parameter
    aperture_averaging_factor: float = 1.0
    spectrum_model: str = ""
    
    def __post_init__(self):
        self.k = 2 * np.pi / self.wavelength_m
        self.spectrum_model = (
            "kolmogorov" if np.isinf(self.outer_scale_m)
            else "von_karman"
        )
        self._compute_rytov()
        self._compute_fried()
        self._compute_aperture_averaging()
        self._compute_gamma_gamma_params()
    
    def _compute_rytov(self):
        """
        Rytov variance for a plane wave.
        
        Kolmogorov:
          sigma2_R = 1.23 * Cn2 * k^{7/6} * L^{11/6}
        
        Von Karman correction (Andrews & Phillips Ch. 9):
          The finite outer scale reduces the contribution from
          large eddies. The correction depends on the dimensionless
          parameter Q_L = k * L0^2 / L (outer-scale Fresnel number).
          When Q_L >> 1, correction is small (Kolmogorov limit).
          When Q_L ~ 1, reduction is significant.
        """
        self.rytov_variance = (
            1.23 * self.Cn2 
            * self.k ** (7.0 / 6.0) 
            * self.link_distance_m ** (11.0 / 6.0)
        )
        
        if np.isinf(self.outer_scale_m):
            self.rytov_variance_vk = self.rytov_variance
        else:
            L = self.link_distance_m
            L0 = self.outer_scale_m
            Q_L = self.k * L0**2 / L
            
            # Correction factor from modified Von Karman spectrum
            # (fitted to Andrews & Phillips numerical results)
            correction = 1.0 - 0.65 * (1.0 + Q_L)**(-5.0/6.0)
            correction = max(correction, 0.1)
            
            self.rytov_variance_vk = self.rytov_variance * correction
    
    def _compute_fried(self):
        """
        Fried parameter (atmospheric coherence diameter):
          r0 = (0.423 * k^2 * Cn2 * L)^{-3/5}
        
        r0 is defined within the Kolmogorov inertial range and does
        not depend on the outer scale.
        """
        self.r0 = (
            0.423 * self.k**2 * self.Cn2 * self.link_distance_m
        ) ** (-3.0 / 5.0)
    
    def _compute_aperture_averaging(self):
        """
        Aperture averaging factor for a circular receiver.
        
        A finite aperture spatially averages intensity fluctuations:
          sigma2_I(D) = A(D) * sigma2_I(point)
        
        For Kolmogorov (Andrews & Phillips Eq. 10.62):
          A = [1 + 1.062*(D/2)^2 / (lambda*L)]^{-7/6}
        
        The Fresnel zone sqrt(lambda*L) is the comparison length.
        For D >> sqrt(lambda*L), aperture averaging is strong.
        """
        D = self.aperture_diameter_m
        L = self.link_distance_m
        lam = self.wavelength_m
        
        d_ratio = (D / 2.0)**2 / (lam * L)
        self.aperture_averaging_factor = (1.0 + 1.062 * d_ratio) ** (-7.0 / 6.0)
    
    def _compute_gamma_gamma_params(self):
        """
        Gamma-Gamma distribution parameters using Von Karman Rytov
        variance and aperture averaging.
        
        Large-scale (alpha): affected by outer scale but NOT by aperture
          averaging (large eddies > aperture pass through unaveraged).
        Small-scale (beta): reduced by aperture averaging (small eddies
          < aperture are spatially averaged out).
        """
        sr2 = self.rytov_variance_vk
        sr_12_5 = sr2 ** (6.0 / 5.0)
        
        # Large-scale log-irradiance variance (point receiver)
        sigma2_ln_X = 0.49 * sr2 / (1.0 + 1.11 * sr_12_5) ** (7.0 / 6.0)
        
        # Small-scale log-irradiance variance (point receiver)
        sigma2_ln_Y = 0.51 * sr2 / (1.0 + 0.69 * sr_12_5) ** (5.0 / 6.0)
        
        # Apply aperture averaging to small-scale only
        A = self.aperture_averaging_factor
        sigma2_ln_Y_ap = sigma2_ln_Y * A
        
        # Gamma-Gamma parameters
        self.alpha = (
            1.0 / (np.exp(sigma2_ln_X) - 1.0) 
            if sigma2_ln_X > 1e-10 else 1e6
        )
        self.beta = (
            1.0 / (np.exp(sigma2_ln_Y_ap) - 1.0) 
            if sigma2_ln_Y_ap > 1e-10 else 1e6
        )
        
        # Scintillation index (after aperture averaging)
        self.scintillation_index = (
            1.0 / self.alpha + 1.0 / self.beta 
            + 1.0 / (self.alpha * self.beta)
        )
    
    def coherence_time(self, wind_speed_m_s: float) -> float:
        """
        Atmospheric coherence time (frozen flow hypothesis).
          tau_0 = 0.3 * r_0 / v_perp
        """
        if wind_speed_m_s <= 0:
            return float('inf')
        return 0.3 * self.r0 / wind_speed_m_s
    
    def greenwood_frequency(self, wind_speed_m_s: float) -> float:
        """
        Greenwood frequency: bandwidth of turbulence fluctuations.
          f_G = 0.43 * v_perp / r_0
        """
        return 0.43 * wind_speed_m_s / self.r0

    def summary(self, wind_speed_m_s: float = 5.0) -> str:
        """Human-readable summary of turbulence parameters."""
        tau0 = self.coherence_time(wind_speed_m_s)
        fG = self.greenwood_frequency(wind_speed_m_s)
        regime = (
            "weak" if self.rytov_variance < 0.3
            else "moderate" if self.rytov_variance < 2.0
            else "strong" if self.rytov_variance < 10.0
            else "saturated"
        )
        vk_note = ""
        if not np.isinf(self.outer_scale_m):
            reduction = (1 - self.rytov_variance_vk / self.rytov_variance) * 100
            vk_note = (
                f"  Von Karman L0  = {self.outer_scale_m:.1f} m\n"
                f"  VK correction  = {reduction:.1f}% reduction in Rytov var.\n"
                f"  Aperture avg.  = {self.aperture_averaging_factor:.3f} "
                f"(D={self.aperture_diameter_m*1e3:.0f} mm)\n"
            )
        return (
            f"Turbulence Parameters ({regime} regime, {self.spectrum_model})\n"
            f"  Cn2           = {self.Cn2:.2e} m^{{-2/3}}\n"
            f"  Rytov var.    = {self.rytov_variance:.4f} (Kolmogorov)\n"
            f"  Rytov var.(VK)= {self.rytov_variance_vk:.4f}\n"
            f"  Fried r0      = {self.r0*1e3:.2f} mm\n"
            f"{vk_note}"
            f"  Gamma-Gamma a = {self.alpha:.3f}\n"
            f"  Gamma-Gamma b = {self.beta:.3f}\n"
            f"  Scint. Index  = {self.scintillation_index:.4f}\n"
            f"  Coherence t0  = {tau0*1e3:.3f} ms  (v={wind_speed_m_s} m/s)\n"
            f"  Greenwood f_G = {fG:.1f} Hz  (v={wind_speed_m_s} m/s)"
        )
