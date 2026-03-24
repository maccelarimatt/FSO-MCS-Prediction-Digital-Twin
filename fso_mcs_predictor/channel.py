"""
FSO turbulence channel model — wave-optics implementation.

Uses aotools PhaseScreenVonKarman (Von Kármán infinite phase screens) +
angular spectrum Fresnel propagation to generate aperture-averaged
intensity time-series.

Physics pipeline per realisation
─────────────────────────────────
1. Compute r0, f_G analytically from Cn² and sampled wind speed.
2. Choose subsampling rate: evaluate wave-optics at 4×f_G
   (intensity is bandlimited to ~f_G; 4× gives 2× Nyquist margin).
3. Instantiate PhaseScreenVonKarman(N, dx, r0, L0).
4. Each wave-optics step: advance screen by round(v_wind * dt_screen / dx)
   rows (Taylor frozen turbulence), propagate a plane wave through the
   current N×N phase screen, average intensity over the 75 mm aperture.
5. Normalise coarse series to unit mean, then linear-interpolate to 10 kHz.
6. Multiply by pointing-error factor and mean power, map to SNR.

Grid parameters
───────────────
dx = min(r0/4, 2 mm)   → always ≥ 4 pixels per r0
N  = next power of 2 covering (D_rx + 4·r_F) / dx, clamped to [64, 256]
     r_F = √(λL) ≈ 35 mm  (Fresnel zone radius)
     ensures the propagated field fits in the grid without wrap-around.

Propagation (paraxial angular spectrum)
────────────────────────────────────────
H(fx, fy) = exp(−iπλz (fx² + fy²))   [Fresnel / paraxial regime]

Valid because λ·f_max = 1550e-9 × 250 ≈ 4×10⁻⁴ ≪ 1 for dx = 2 mm, so
the non-paraxial correction is < 10⁻⁷ and can be ignored.

H is precomputed once per realisation; each propagation is then only
fft2 + pointwise multiply + ifft2 + aperture average.

Known limitations
─────────────────
- Plane wave input (Gaussian beam would be more physical but the
  statistical properties of aperture-averaged scintillation are the same
  for a well-collimated beam at 800 m).
- Linear interpolation from coarse to fine time grid adds a small
  spectral shaping below f_G. This is invisible to the NN because
  the windowed features already low-pass at the window timescale.

References
──────────
- Andrews & Phillips, "Laser Beam Propagation through Random Media" (2005)
- Assemat & Wilson, "Method for simulating infinitely long and non-stationary
  phase screens", Optics Express 14, 988 (2006)
- Schmidt, "Numerical Simulation of Optical Wave Propagation", SPIE (2010)
"""

import numpy as np
from . import config


# ─────────────────────────────────────────────────────────────────────────────
# Analytical parameter computation (unchanged — used by print_regime_summary)
# ─────────────────────────────────────────────────────────────────────────────

def compute_turbulence_params(Cn2, v_wind):
    """
    Compute all turbulence parameters analytically from Cn² and wind speed.

    Returns dict with: sigma2_R, sigma2_R_VK, r0, tau0, f_G, alpha, beta,
                        aperture_avg, sigma2_ln_X, sigma2_ln_Y
    """
    k = 2 * np.pi / config.WAVELENGTH
    L = config.LINK_DISTANCE
    D = config.RX_APERTURE_D
    L0 = config.L0

    # Rytov variance (plane wave, Kolmogorov)
    sigma2_R = 1.23 * Cn2 * k ** (7 / 6) * L ** (11 / 6)

    # Von Kármán correction (Q_L ≈ 3.2 M for this link → < 0.1 % effect)
    Q_L = k * L0 ** 2 / L
    sigma2_R_VK = sigma2_R * (1 - 0.65 * (1 + Q_L) ** (-5 / 6))

    # Fried parameter
    r0 = (0.423 * k ** 2 * Cn2 * L) ** (-3 / 5)

    # Aperture averaging factor  (D=75 mm, λ=1550 nm, L=800 m → A ≈ 0.398)
    A = (1 + 1.062 * (D / 2) ** 2 / (config.WAVELENGTH * L)) ** (-7 / 6)

    # Gamma-Gamma parameters
    sigma_R_VK = np.sqrt(sigma2_R_VK)
    sigma2_ln_X = 0.49 * sigma2_R_VK / (1 + 1.11 * sigma_R_VK ** (12 / 5)) ** (7 / 6)
    sigma2_ln_Y = 0.51 * sigma2_R_VK / (1 + 0.69 * sigma_R_VK ** (12 / 5)) ** (5 / 6) * A
    alpha = 1.0 / (np.exp(sigma2_ln_X) - 1)
    beta = 1.0 / (np.exp(sigma2_ln_Y) - 1)

    # Temporal parameters
    tau0 = 0.3 * r0 / v_wind
    f_G = 0.43 * v_wind / r0

    return {
        "sigma2_R":    sigma2_R,
        "sigma2_R_VK": sigma2_R_VK,
        "r0":          r0,
        "tau0":        tau0,
        "f_G":         f_G,
        "alpha":       alpha,
        "beta":        beta,
        "aperture_avg": A,
        "sigma2_ln_X": sigma2_ln_X,
        "sigma2_ln_Y": sigma2_ln_Y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wave-optics propagator
# ─────────────────────────────────────────────────────────────────────────────

def _build_propagator(N, dx, wavelength, L, rx_diameter):
    """
    Precompute the Fresnel transfer function and receiver aperture mask.

    H(fx, fy) = exp(−iπλL (fx² + fy²))  in the paraxial regime.

    Returns
    -------
    H       : (N, N) complex128  — Fresnel transfer function
    aperture: (N, N) bool        — circular receiver aperture mask
    """
    fx = np.fft.fftfreq(N, d=dx)   # cycles/m
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    # Paraxial angular spectrum: exp(−iπλL(fx²+fy²))
    H = np.exp(-1j * np.pi * wavelength * L * (FX ** 2 + FY ** 2))

    # Circular aperture centred on grid
    cy, cx = N // 2, N // 2
    y_idx, x_idx = np.ogrid[:N, :N]
    r_px = (rx_diameter / 2.0) / dx
    aperture = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= r_px ** 2

    return H, aperture


def _propagate_and_average(phase_screen, H, aperture):
    """
    Propagate a unit-amplitude plane wave through phase_screen.

    E_in = exp(i·φ)  →  E_rx = IFFT[ FFT[E_in] · H ]
    Returns aperture-averaged intensity (scalar float).
    """
    E = np.exp(1j * phase_screen)
    E_rx = np.fft.ifft2(np.fft.fft2(E) * H)
    I = np.abs(E_rx) ** 2
    return float(I[aperture].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Pointing error (two-component AR(1) beam jitter)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pointing_error(n_samples, rng=None):
    """
    Generate pointing error loss factor using two AR(1) components.

    Slow drift (τ = 100 s) + fast jitter (τ = 1 s) → Gaussian beam
    displacement → exp(−2r²/w²) coupling loss.  Values in (0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = config.DT

    # Slow drift
    a_slow = np.exp(-dt / 100.0)
    z_slow = np.zeros(n_samples)
    z_slow[0] = rng.normal()
    noise_slow = rng.normal(size=n_samples) * np.sqrt(1 - a_slow ** 2)
    for i in range(1, n_samples):
        z_slow[i] = a_slow * z_slow[i - 1] + noise_slow[i]

    # Fast jitter
    a_fast = np.exp(-dt / 1.0)
    z_fast = np.zeros(n_samples)
    z_fast[0] = rng.normal()
    noise_fast = rng.normal(size=n_samples) * np.sqrt(1 - a_fast ** 2)
    for i in range(1, n_samples):
        z_fast[i] = a_fast * z_fast[i - 1] + noise_fast[i]

    # Combined beam displacement (σ_slow ≈ 0.05 w, σ_fast ≈ 0.03 w)
    displacement = 0.05 * z_slow + 0.03 * z_fast
    return np.exp(-2.0 * displacement ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main channel realisation generator — wave-optics
# ─────────────────────────────────────────────────────────────────────────────

def generate_channel_realisation(regime_name, duration_s, mean_power_dbm=None, seed=None):
    """
    Generate a complete channel realisation using Von Kármán phase screens.

    Steps
    -----
    1. Sample wind speed; compute r0, f_G.
    2. Choose temporal subsampling: wave-optics at 4×f_G.
    3. Set grid: dx = min(r0/4, 2 mm), N = next power of 2 covering
       the aperture + Fresnel margin, clamped to [64, 256].
    4. Instantiate PhaseScreenVonKarman; advance by pixels_per_eval rows
       per evaluation (Taylor frozen turbulence).
    5. Propagate plane wave → aperture-average intensity.
    6. Normalise to unit mean; interpolate to 10 kHz.
    7. Multiply by pointing error and mean power; convert to SNR.

    Returns
    -------
    dict with keys: power_dbm, snr_db, h_turb, h_point, params,
                    regime, v_wind, mean_power_dbm, n_samples,
                    wave_optics_meta (grid / subsampling info)
    """
    try:
        from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman
    except ImportError:
        raise ImportError(
            "aotools is required for wave-optics channel simulation.\n"
            "Install with:  pip install aotools"
        )

    rng = np.random.default_rng(seed)
    regime = config.TURBULENCE_REGIMES[regime_name]

    # ── 1. Wind speed and turbulence parameters ───────────────────────────
    v_wind = float(max(0.5, rng.normal(regime["wind_mean"], regime["wind_std"])))
    params = compute_turbulence_params(regime["Cn2"], v_wind)
    r0 = params["r0"]
    f_G = params["f_G"]
    n_samples = int(duration_s * config.SAMPLE_RATE)

    # ── 2. Temporal subsampling ────────────────────────────────────────────
    # Sample wave-optics at 4×f_G so turbulent bandwidth is well covered.
    subsample = max(1, int(config.SAMPLE_RATE / (4.0 * f_G)))
    subsample = min(subsample, 1000)          # cap: avoid excessive interp range
    dt_screen = subsample * config.DT         # seconds between WO evaluations

    # ── 3. Grid parameters ────────────────────────────────────────────────
    # dx constraints:
    #   lower bound 3 mm  — aotools Cholesky fails for dx < ~3 mm
    #   upper bound 8 mm  — keeps ≥ 4.7 px in aperture radius (D/2 = 37.5 mm)
    #   target      r0/3  — ≥ 3 pixels per r0 for accurate phase statistics
    dx = float(np.clip(min(r0 / 3.0, 8e-3), 3e-3, 8e-3))

    # N: cover D_rx + 4·r_Fresnel so propagated field doesn't wrap
    r_fresnel = np.sqrt(config.WAVELENGTH * config.LINK_DISTANCE)   # ≈ 35 mm
    min_width = config.RX_APERTURE_D + 4.0 * r_fresnel              # ≈ 215 mm
    N = int(2 ** np.ceil(np.log2(min_width / dx)))
    N = int(np.clip(N, 32, 128))

    # ── 4. Phase screen + propagator ─────────────────────────────────────
    H, aperture = _build_propagator(
        N, dx, config.WAVELENGTH, config.LINK_DISTANCE, config.RX_APERTURE_D
    )

    ao_seed = int(rng.integers(0, 2 ** 31))
    screen = PhaseScreenVonKarman(
        nx_size=N,
        pixel_scale=dx,
        r0=r0,
        L0=config.L0,
        random_seed=ao_seed,
    )

    # Pixels to advance per WO evaluation (Taylor hypothesis)
    pixels_per_eval = max(1, round(v_wind * dt_screen / dx))

    # ── 5. Generate coarse intensity series ──────────────────────────────
    n_screen = int(np.ceil(n_samples / subsample)) + 1
    I_coarse = np.empty(n_screen, dtype=np.float64)

    for i in range(n_screen):
        I_coarse[i] = _propagate_and_average(screen.scrn, H, aperture)
        for _ in range(pixels_per_eval):
            screen.add_row()

    # ── 6. Normalise and interpolate ──────────────────────────────────────
    # Unit-mean normalisation: on average the aperture captures <I>=1,
    # but each realisation has a slightly different mean.
    I_coarse /= (I_coarse.mean() + 1e-12)

    # Linear interpolation from coarse grid to 10 kHz
    t_coarse = np.arange(n_screen, dtype=np.float64) * subsample
    t_fine = np.arange(n_samples, dtype=np.float64)
    h_turb = np.interp(t_fine, t_coarse, I_coarse)

    # ── 7. Pointing error and power ──────────────────────────────────────
    h_point = generate_pointing_error(n_samples, rng)

    if mean_power_dbm is None:
        power_map = {
            "weak":        -8.0,
            "moderate":   -13.0,
            "strong":     -18.0,
            "very_strong": -22.0,
        }
        mean_power_dbm = power_map[regime_name]

    mean_power_mw = 10.0 ** (mean_power_dbm / 10.0)
    rx_power_mw = mean_power_mw * h_turb * h_point
    rx_power_mw = np.clip(rx_power_mw, 1e-10, 10.0 ** (config.PEAK_RX_POWER_DBM / 10.0))
    power_dbm = 10.0 * np.log10(rx_power_mw)
    snr_db = config.SNR_SLOPE * power_dbm + config.SNR_INTERCEPT

    return {
        "power_dbm":      power_dbm,
        "snr_db":         snr_db,
        "h_turb":         h_turb,
        "h_point":        h_point,
        "params":         params,
        "regime":         regime_name,
        "v_wind":         v_wind,
        "mean_power_dbm": mean_power_dbm,
        "n_samples":      n_samples,
        "wave_optics_meta": {
            "N":               N,
            "dx_mm":           dx * 1e3,
            "subsample":       subsample,
            "pixels_per_eval": pixels_per_eval,
            "n_screen_evals":  n_screen,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCS mapping
# ─────────────────────────────────────────────────────────────────────────────

def snr_to_mcs(snr_db, hysteresis=0.0):
    """
    Map SNR (dB) to ideal MCS class using thresholds with optional hysteresis.

    Args:
        snr_db    : scalar or array of SNR values
        hysteresis: dB margin subtracted from thresholds (conservative selection)

    Returns: MCS class index array (0–14)
    """
    thresholds = config.MCS_SNR_THRESHOLDS.copy()
    thresholds[1:] -= hysteresis   # do not apply to outage threshold

    snr_db = np.asarray(snr_db)
    mcs = np.zeros_like(snr_db, dtype=int)
    for i in range(config.NUM_CLASSES - 1, 0, -1):
        mcs[snr_db >= thresholds[i]] = np.maximum(mcs[snr_db >= thresholds[i]], i)
    return mcs


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_regime_summary(regime_name):
    """Print turbulence parameters for a regime (useful for verification)."""
    regime = config.TURBULENCE_REGIMES[regime_name]
    params = compute_turbulence_params(regime["Cn2"], regime["wind_mean"])

    print(f"\n{'=' * 55}")
    print(f"Regime: {regime_name}")
    print(f"  Cn²           = {regime['Cn2']:.1e} m^(-2/3)")
    print(f"  Wind          = {regime['wind_mean']} ± {regime['wind_std']} m/s")
    print(f"  Rytov σ²_R    = {params['sigma2_R']:.4f}")
    print(f"  VK σ²_R       = {params['sigma2_R_VK']:.4f}")
    print(f"  r₀            = {params['r0'] * 100:.2f} cm")
    print(f"  τ₀            = {params['tau0'] * 1000:.3f} ms")
    print(f"  f_G           = {params['f_G']:.1f} Hz")
    print(f"  α (large)     = {params['alpha']:.2f}")
    print(f"  β (small)     = {params['beta']:.2f}")
    print(f"  Aperture avg  = {params['aperture_avg']:.4f}")
    # Grid that would be used
    r0 = params["r0"]
    dx = float(np.clip(min(r0 / 3.0, 8e-3), 3e-3, 8e-3))
    r_fresnel = np.sqrt(config.WAVELENGTH * config.LINK_DISTANCE)
    min_width = config.RX_APERTURE_D + 4.0 * r_fresnel
    N = int(np.clip(2 ** np.ceil(np.log2(min_width / dx)), 32, 128))
    subsample = int(np.clip(max(1, int(config.SAMPLE_RATE / (4.0 * params["f_G"]))), 1, 1000))
    print(f"  WO grid       = {N}×{N} px, dx={dx * 1e3:.1f} mm")
    print(f"  WO subsample  = {subsample}  ({config.SAMPLE_RATE / subsample:.0f} Hz eval rate)")
    print(f"{'=' * 55}")
