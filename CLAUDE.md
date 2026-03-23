# CLAUDE.md — FSO-TVWS MCS Prediction Digital Twin

## What This Project Is

An MSc project (Wits University, supervisor: Mitchell Cox) building a neural network predictor for Modulation and Coding Scheme (MCS) selection in a hybrid Free-Space Optical (FSO) / TV Whitespace (TVWS) communication link.

This is **Phase 1: a digital twin simulation** that generates synthetic FSO channel data and compares NN architectures for predicting the optimal MCS class before a fade happens.

The end goal (Phase 2-3) is to deploy the best predictor on a real FSO testbed at Wits using GNU Radio + USRPs.

Reference paper: Dindar et al., "Hybrid Amplify and Forward TVWS Radio over Free-Space Optics," J. Lightwave Technol.

## Project Structure

```
FSO-Digital-Twin/
├── run.py                         # Entry point — trains all models, evaluates, plots
├── plot_horizon_sweep.py          # Combines results from multiple horizon runs
├── CLAUDE.md                      # This file
└── fso_mcs_predictor/             # Main package
    ├── __init__.py
    ├── config.py                  # All constants, MCS table, turbulence regimes, hyperparams
    ├── channel.py                 # aotools phase screens + angular spectrum propagation
    ├── dataset.py                 # Windowing, feature engineering, realisation-level split
    ├── models.py                  # 8 NN architectures + reactive baseline + adaptive selector
    └── evaluate.py                # Metrics: accuracy, throughput ratio, over-prediction rate
```

## How to Run

```bash
# Quick demo (~2 min with aotools wave-optics; ~5 min with physics simulation)
python run.py --no-esn

# Full dataset for thesis results (~15-30 min)
python run.py --full --max-iter 100

# THE KEY EXPERIMENT — horizon sweep
python run.py --horizon 1 --output-dir results_h1
python run.py --horizon 10 --output-dir results_h10
python run.py --horizon 50 --output-dir results_h50

# Then plot the comparison
python plot_horizon_sweep.py
```

## Dependencies

```
numpy, scipy, matplotlib, scikit-learn, aotools
```

Optional (for proper NN architectures, not yet implemented): `torch`

## The Physics

- **FSO link:** 800m folded path, 1550nm, 8mW TX, 75mm Rx aperture
- **Channel model:** aotools `PhaseScreenVonKarman` (Von Kármán PSD, r0 from analytical path-integral) → angular spectrum Fresnel propagation → aperture-averaged intensity
- **Temporal correlation:** Taylor frozen turbulence — screen translates at wind speed; each unique phase screen (one pixel advance = dx/v_wind seconds) is propagated once and reused
- **SNR model:** SNR_dB = 1.04 × P_dBm + 33.1 (empirical from paper, thermal-noise-limited)
- **SI note:** Wave-optics aperture-averaged SI (0.05–0.15 for strong turbulence) is lower than the analytical GG model (0.30–0.40). Root cause: finite-grid fine speckle scale ξ = λL/L_screen = 6.5 mm creates additional averaging beyond what Andrews & Phillips predict. This is a known thin-screen/finite-grid limitation. Temporal dynamics and relative NN comparisons are unaffected.
- **SNR model:** SNR_dB = 1.04 × P_dBm + 33.1 (empirical from paper, thermal-noise-limited)
- **MCS table:** 15 classes (0=outage, 1-14 from QPSK 1/2 to 256QAM 5/6). Thresholds in config.py.

### Turbulence Regimes

| Regime | Cn² (m⁻²/³) | Wind (m/s) | Coherence time τ₀ |
|--------|-------------|------------|-------------------|
| Weak | 5e-16 | 2.0 ± 0.5 | ~81 ms |
| Moderate | 5e-15 | 5.0 ± 1.5 | ~8 ms |
| Strong | 5e-14 | 8.0 ± 2.0 | ~1.3 ms |
| Very strong | 5e-13 | 10.0 ± 3.0 | ~0.26 ms |

### Key Equations (in channel.py)

- Rytov variance: σ²_R = 1.23 · Cn² · k^(7/6) · L^(11/6)
- Aperture averaging: A ≈ 0.398 for this link (60% scintillation reduction)
- Gamma-Gamma params: α from large-scale, β from small-scale (aperture-averaged)
- AR(1) coefficient: a = exp(-dt/τ₀), then Gaussian→Uniform→GG inverse CDF transform
- Known limitation: AR(1) gives exponential autocorrelation instead of true stretched exponential exp(-(τ/τ₀)^(5/3)). Adequate for architecture comparison.

## Features (5 per timestep)

| # | Feature | Why |
|---|---------|-----|
| 0 | Raw power (dBm) | Absolute level — NN needs this to learn MCS thresholds |
| 1 | Raw SNR (dB) | Same reason — thresholds are in absolute dB |
| 2 | Normalised power | Zero-mean, unit-var within window — captures temporal pattern |
| 3 | Log scintillation index | Backward-looking 500-sample SI — regime context |
| 4 | Power derivative | First difference in dB — rate of change |

**Critical lesson from previous version:** Features 0 and 1 must NOT be z-score normalised per-realisation. The previous version stripped absolute power/SNR, so the NN couldn't distinguish SNR=10dB from SNR=25dB. Both raw AND normalised features are needed.

## Dataset Split — Realisation-Level (No Data Leakage)

**DO NOT shuffle all windows then split.** Overlapping windows from the same time-series would leak into train and test.

- With ≥3 realisations: reals 0,1 → train; real 2 → first half val, second half test
- With 1 realisation (quick mode): 60/20/20 time split with gaps ≥ window_size between segments

## Models (8 architectures)

Currently implemented as sklearn MLPClassifier with architecture-specific feature engineering (PyTorch versions not yet built):

1. **MLP** — Flatten temporal dimension. Baseline: is temporal structure needed?
2. **CNN1D** — Local window statistics at multiple scales
3. **GRU** — Exponentially weighted features (recurrent approximation). Deployment candidate.
4. **LSTM** — Similar to GRU, slightly different decay
5. **HybridCNNGRU** — Combines CNN local + GRU sequential features
6. **TCN** — Multi-scale causal sampling (dilated convolution approximation)
7. **Transformer** — Attention-weighted aggregation
8. **ESN** — Echo State Network, proper numpy reservoir + ridge regression (slow, use --no-esn to skip)

Plus:
- **Reactive baseline** — Traditional AMC with feedback delay (delay 0-90 samples)
- **Adaptive selector** — SI-gated: uses reactive when channel stable (SI < 0.05), NN when volatile

## TODO: PyTorch Upgrade

The sklearn models are approximations. For thesis-quality results, replace with proper PyTorch nn.Module implementations:

- MLP: Flatten → FC(256) → FC(128) → FC(15), dropout 0.3
- CNN1D: 3× Conv1d kernel=5, channels [32,64,64] → GlobalAvgPool → FC
- GRU: 2-layer, hidden=128 → FC(15)
- LSTM: 2-layer, hidden=128 → FC(15)
- HybridCNNGRU: 2× Conv1d [32,64] → 1-layer GRU hidden=64 → FC
- TCN: 3 residual blocks, kernel=7, dilations [1,2,4] → GlobalAvgPool → FC
- Transformer: d_model=64, 4 heads, 3 layers, d_ff=256, causal mask → FC
- ESN: reservoir=500, spectral_radius=0.95, leak=0.3, ridge regression

Training config: AdamW (lr=1e-3, weight_decay=1e-5), weighted cross-entropy (NOT focal loss — it was tested and destroyed training), batch=256, max_epochs=100, early_stopping patience=15, ReduceLROnPlateau, gradient clipping max_norm=1.0.

## Evaluation Metrics

- **Throughput ratio** — PRIMARY METRIC. sum(actual_throughput) / sum(ideal_throughput). Over-predictions get zero throughput.
- **Over-prediction rate** — CRITICAL SAFETY METRIC. Predicted MCS > ideal = packet error.
- **Accuracy** — Exact MCS match
- **±1 Accuracy** — Within 1 MCS level
- **MAE** — Mean absolute MCS error

## Expected Behaviour

- Horizon=1 (0.1ms): MLP competitive, reactive d=0 strong. Channel barely changes.
- Horizon=10 (1ms ≈ τ₀ for strong turbulence): sequence models start pulling ahead.
- Horizon=50 (5ms): sequence models should significantly beat MLP. Reactive degrades badly.
- If MLP still beats sequence models at horizon=50 with PyTorch, something is wrong.

## Known Issues / Things to Watch

1. **Classes 12-14 (256QAM) will have very few or zero training samples** at lower mean powers. This is correct physics — SNR rarely reaches 27-29 dB. Don't treat it as a bug.
2. **plot_horizon_sweep.py has hardcoded numbers** from the demo run. Update them after your runs, or modify to read from JSON.
3. **ESN is very slow in numpy** (~5 min for 5k samples). Use --no-esn during development.
4. **ConvergenceWarning from sklearn** is normal — increase --max-iter to reduce it.
5. **AR(1) limitation** — underestimates fade duration clustering. Document in thesis.

## Style Preferences (for Claude)

- Answers that are to the point
- Mention better alternatives if they exist
- Address bad ideas and suggest better ones
- Keep wording simple without losing technical explanation
- Use tables and equations where they explain better than prose
