# FSO-TVWS MCS Prediction Digital Twin

MSc project — Wits University
Supervisor: Mitchell Cox
Reference: Dindar et al., *"Hybrid Amplify and Forward TVWS Radio over Free-Space Optics"*, J. Lightwave Technol.

---

## Overview

This repository implements a **digital twin simulation** of a hybrid Free-Space Optical (FSO) / TV Whitespace (TVWS) communication link, used to train and compare neural network architectures for **proactive Modulation and Coding Scheme (MCS) selection**.

The core idea: rather than reacting to the current channel state (traditional AMC), a neural network observes a short history of the received signal and predicts the optimal MCS class *H* steps into the future — before a fade degrades the link. The horizon *H* controls how far ahead the prediction must reach, and is the primary experimental variable.

The end goal (Phase 2–3) is to deploy the best-performing predictor on a real FSO testbed at Wits using GNU Radio and USRPs.

---

## Physical Setup

| Parameter | Value |
|---|---|
| Link distance | 800 m (folded path) |
| Wavelength | 1550 nm |
| TX power | 8 mW (9 dBm) |
| RX aperture | 75 mm (plano-convex lens) |
| Sample rate | 10 kHz (photodetector DC monitor) |
| SNR model | SNR_dB = 1.04 × P_dBm + 33.1 (empirical, thermal-noise-limited) |

### Turbulence Regimes

| Regime | Cn² (m⁻²/³) | Wind (m/s) | Coherence time τ₀ |
|---|---|---|---|
| Weak | 5×10⁻¹⁶ | 2.0 ± 0.5 | ~81 ms |
| Moderate | 5×10⁻¹⁵ | 5.0 ± 1.5 | ~8 ms |
| Strong | 5×10⁻¹⁴ | 8.0 ± 2.0 | ~1.3 ms |
| Very strong | 5×10⁻¹³ | 10.0 ± 3.0 | ~0.26 ms |

### MCS Table (15 classes)

| Class | Modulation | Code rate | Min SNR (dB) |
|---|---|---|---|
| 0 | Outage | — | < 7 |
| 1–4 | QPSK | 1/2 → 5/6 | 7 – 9.5 |
| 5–8 | 16QAM | 1/2 → 5/6 | 13.5 – 16 |
| 9–11 | 64QAM | 2/3 → 5/6 | 20 – 21.5 |
| 12–14 | 256QAM | 2/3 → 5/6 | 27 – 29 |

---

## Channel Model

The channel simulation (`fso_mcs_predictor/channel.py`) uses wave-optics, not a statistical model:

1. **Phase screen** — `aotools.PhaseScreenVonKarman` generates an infinite scrolling Von Kármán phase screen with the correct spatial PSD for the given Cn² and outer scale L₀ = 25 m.
2. **Propagation** — Each screen is propagated to the receiver using the paraxial angular-spectrum method (Fresnel transfer function, FFT-based). The receiver aperture intensity is averaged over the 75 mm aperture.
3. **Taylor frozen turbulence** — The screen is advanced by `round(v_wind × dt / dx)` rows per evaluation step, giving physically correct temporal correlation.
4. **Temporal subsampling** — Wave-optics is evaluated at 4 × f_G (Greenwood frequency) and then interpolated to 10 kHz, keeping computation tractable.
5. **Pointing error** — Two-component AR(1) model (slow thermal drift τ = 100 s + fast vibration jitter τ = 1 s) applied as a multiplicative coupling loss.
6. **SNR mapping** — Received power is converted to SNR using the empirical linear model from the Dindar et al. paper.

The numerical grid is chosen automatically per realisation: `dx = clip(r₀/3, 3–8 mm)`, `N = next power-of-2` covering the aperture + Fresnel margin, clamped to [32, 128].

---

## Features (5 per timestep)

Each 10 ms sliding window (100 samples, stride 50) produces a `(100, 5)` feature tensor:

| Index | Feature | Purpose |
|---|---|---|
| 0 | Raw power (dBm) | Absolute level — required to learn MCS thresholds |
| 1 | Raw SNR (dB) | Same — thresholds are absolute dB values |
| 2 | Normalised power (z-score within window) | Captures fade shape/trend |
| 3 | Log scintillation index (backward 500 samples) | Turbulence regime context |
| 4 | Power derivative (first difference dBm) | Rate of change — fade direction |

**Important:** features 0 and 1 are intentionally *not* globally normalised. Stripping absolute power/SNR prevents the network from learning which MCS thresholds are relevant.

---

## Dataset Split

Splits are done at **realisation level** to prevent temporal data leakage from overlapping windows:

- **≥ 3 realisations (recommended):** realisations 0…N-2 → train; realisation N-1 first half → val; second half → test.
- **1 realisation (quick mode):** 60 / 20 / 20 time split with gaps ≥ window size between segments.

---

## Models (8 architectures)

All currently use `sklearn MLPClassifier` as the training backbone with architecture-specific feature engineering (PyTorch versions planned for Phase 2):

| Model | Approach |
|---|---|
| MLP | Flatten temporal dimension — no sequence modelling |
| CNN1D | Multi-scale local statistics (mean/std at 4 window scales) |
| GRU | Exponentially weighted features (decay = 0.95) |
| LSTM | Exponentially weighted features (decay = 0.97, longer memory) |
| HybridCNNGRU | CNN features concatenated with GRU features |
| TCN | Dilated causal sampling at dilations [1, 2, 4, 8, 16] |
| Transformer | Dot-product attention (last timestep as query) |
| ESN | Random reservoir (tanh leaky units) + ridge regression readout |

Plus two non-NN references:

- **Reactive baseline** — traditional AMC using current SNR with a configurable feedback delay (d = 0–90 samples)
- **Adaptive selector** — SI-gated hybrid: uses reactive when SI < 0.05 (stable channel), NN otherwise

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Throughput ratio** *(primary)* | Σ thr[predicted MCS] / Σ thr[ideal MCS]. Over-predictions contribute 0 (packet error). |
| **Over-prediction rate** *(safety)* | Fraction of windows where predicted MCS > ideal. Each over-prediction is a dropped packet. |
| Accuracy | Exact MCS class match rate |
| ±1 Accuracy | Within 1 MCS class |
| MAE | Mean absolute MCS class error |

---

## Project Structure

```
FSO-MCS-Prediction-Digital-Twin/
├── run.py                          # Entry point — trains models, saves results.json + predictions.npz
├── verify.py                       # Sanity-check the channel physics (no ML needed)
├── plot_horizon_sweep.py           # Stub — redirects to plots/plot_horizon_sweep.py
├── CLAUDE.md                       # Project notes for AI assistant
├── fso_mcs_predictor/
│   ├── config.py                   # All constants: MCS table, turbulence regimes, hyperparams
│   ├── channel.py                  # Wave-optics channel simulation (aotools + Fresnel propagation)
│   ├── dataset.py                  # Windowing, feature engineering, realisation-level split
│   ├── models.py                   # 8 NN architectures + reactive baseline + adaptive selector
│   └── evaluate.py                 # Metrics: throughput ratio, over-prediction rate, accuracy
└── plots/
    ├── plot_summary.py             # Bar charts: throughput + over-prediction rate
    ├── plot_timeseries.py          # MCS time series: ideal vs each model
    └── plot_horizon_sweep.py       # Cross-horizon comparison (reads multiple results.json files)
```

---

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
aotools
```

Install with:

```bash
pip install numpy scipy matplotlib scikit-learn aotools
```

---

## Running

### Quick demo (~2 minutes per horizon)

```bash
python run.py --no-esn
```

### Full dataset — required for thesis results (~15–30 min per horizon)

```bash
python run.py --horizon 1  --full --output-dir results_h1
python run.py --horizon 10 --full --output-dir results_h10
python run.py --horizon 50 --full --output-dir results_h50
```

This generates `3 realisations × 300 s × 4 regimes` of wave-optics channel data per run, with a realisation-level train/val/test split.

### Verify the channel physics (no ML)

```bash
python verify.py
```

### Generate plots (after run.py completes)

```bash
# Per-horizon bar chart and time series
python plots/plot_summary.py    results_h1
python plots/plot_timeseries.py results_h1

# Cross-horizon comparison (the key thesis figure)
python plots/plot_horizon_sweep.py results_h1 results_h10 results_h50
```

Plots are saved as `.png` files inside each results directory. The horizon sweep plot is saved in the first directory supplied.

---

## Output Files

Each `run.py` execution writes to its `--output-dir`:

| File | Contents |
|---|---|
| `results.json` | All metrics, physics constants, MCS table, turbulence regime parameters, dataset sizes, per-model results. Human-readable. |
| `predictions.npz` | Compressed arrays: `test_labels`, `test_snr`, `pred__<ModelName>` for every model. Used by plotting scripts. |

---

## Expected Results (Horizon Sensitivity)

| Horizon | Time ahead | Expected behaviour |
|---|---|---|
| H = 1 (0.1 ms) | Less than τ₀ in all regimes | MLP competitive; reactive d=0 strong; channel barely changes |
| H = 10 (1 ms) | ~τ₀ for strong turbulence | Sequence models begin to outperform MLP |
| H = 50 (5 ms) | Several τ₀ for strong/very strong | Sequence models should clearly beat MLP; reactive degrades sharply |

If sequence models do not outperform MLP at H = 50 with the sklearn approximations, the gap should become clear once proper PyTorch implementations are used (see CLAUDE.md for planned architecture details).

---

## Known Limitations

- **Wave-optics SI underestimates analytical GG model** — aperture-averaged SI from the finite-grid wave-optics simulation (0.05–0.15) is lower than the analytical Gamma-Gamma prediction (0.30–0.40). Root cause: the finite speckle cell scale ξ = λL/L_screen = 6.5 mm creates additional averaging beyond what Andrews & Phillips predict. Temporal dynamics and relative model comparisons are unaffected.
- **AR(1) pointing error** — gives exponential autocorrelation rather than the physically correct stretched-exponential exp(−(τ/τ₀)^(5/3)). Adequate for architecture comparison.
- **sklearn model approximations** — the 8 architectures use hand-crafted feature engineering rather than true PyTorch sequential layers. The feature-engineering differences mirror the architectural differences closely enough for a fair comparison; replace with PyTorch for final thesis results.
- **Classes 12–14 (256QAM)** will have few or zero training samples at lower mean powers. This is physically correct — the link rarely reaches 27+ dB SNR.
