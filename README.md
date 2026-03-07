# FSO-TVWS MCS Prediction — Phase 1 Digital Twin

**MSc Research: Context-Aware Channel Prediction for Cognitive Optical-Wireless Links**

Simulated FSO channel environment for comparing neural network architectures at predicting the optimal MCS for a TVWS-over-FSO link. Parameters calibrated to the Wits 800m hybrid RF/FSO testbed (Dindar et al.).

---

## Quick Start

```bash
pip install -r requirements.txt

# Verify channel model (no PyTorch needed)
python verify_channel.py

# Full experiment (all 5 architectures)
python -m fso_mcs_predictor.run_experiment

# Quick debug run (~2 min)
python -m fso_mcs_predictor.run_experiment --quick

# Specific models only
python -m fso_mcs_predictor.run_experiment --models lstm transformer

# Ablation: disable SI context input
python -m fso_mcs_predictor.run_experiment --no-context
```

---

## Project Structure

```
fso_mcs_predictor/
├── config.py                  # Link budget, MCS table (Table I), turbulence regimes
├── channel/
│   ├── turbulence.py         # Rytov variance, Fried parameter, Gamma-Gamma params
│   └── fso_channel.py        # Temporally correlated Gamma-Gamma channel simulator
├── system/
│   └── snr_model.py          # Optical power → SNR → ideal MCS mapping
├── dataset/
│   └── generator.py          # Sliding-window dataset with feature engineering
├── models/
│   ├── base.py               # Abstract predictor interface
│   ├── lstm.py               # LSTM classifier
│   ├── gru.py                # GRU classifier
│   ├── transformer.py        # Causal Transformer encoder
│   ├── esn.py                # Echo State Network (reservoir computing)
│   ├── tcn.py                # Temporal Convolutional Network
│   └── __init__.py           # Model registry/factory
├── trainer.py                 # Training loop (SGD + ESN ridge regression)
├── evaluator.py               # Accuracy, throughput ratio, over-prediction rate
├── run_experiment.py          # Main entry point
├── verify_channel.py          # Channel verification (no PyTorch needed)
└── requirements.txt
```

---

## Channel Model

Gamma-Gamma irradiance model with frozen-flow temporal correlations:

1. AR(1) correlated Gaussian noise matched to coherence time τ₀
2. Marginal transform to Gamma-Gamma via inverse CDF
3. Multiplicative pointing error (slow drift + fast jitter)

| Regime      | Cn²       | σ²_R  | r₀     | τ₀      | f_G     |
|-------------|-----------|-------|--------|---------|---------|
| Weak        | 5e-16     | 0.007 | 541 mm | 81 ms   | 1.6 Hz  |
| Moderate    | 5e-15     | 0.066 | 136 mm | 8.2 ms  | 16 Hz   |
| Strong      | 5e-14     | 0.66  | 34 mm  | 1.3 ms  | 101 Hz  |
| Very Strong | 5e-13     | 6.6   | 8.6 mm | 0.26 ms | 501 Hz  |

**Note:** Measured SI > theoretical SI because pointing errors add intensity fluctuations — this matches the real system (Section IV-A, Dindar et al.). Use `include_pointing_error=False` for pure turbulence studies.

---

## Neural Network Architectures

| Model       | Type             | Why Include It                              |
|-------------|------------------|---------------------------------------------|
| LSTM        | Recurrent        | Baseline, widely used in channel prediction |
| GRU         | Recurrent        | Fewer params, similar performance           |
| Transformer | Attention        | Long-range dependencies                     |
| ESN         | Reservoir         | Ultra-fast training, hardware-friendly      |
| TCN         | Convolutional    | Parallelisable, good for real-time          |

Add a new model: inherit `BaseMCSPredictor`, implement `forward()`, register in `models/__init__.py`.

---

## Input Features

| Feature  | Description                                               |
|----------|-----------------------------------------------------------|
| Power    | Log-normalised received optical power (dBm)               |
| SNR      | Electrical SNR (normalised)                               |
| SI       | Sliding-window scintillation index — the **context** input |
| ΔPower   | Rate of change (first difference)                         |

The SI context is the concept note's key hypothesis. Run `--no-context` for ablation.

---

## Evaluation Metrics

- **Over-prediction rate**: Predicted MCS too high → packet errors (most critical)
- **Under-prediction rate**: Predicted MCS too low → wasted throughput
- **Throughput ratio**: Actual/ideal throughput (accounts for errors from over-prediction)
- **Per-regime breakdown**: Generalisation across turbulence conditions
- **Inference speed**: Relevant for real-time deployment (Phase 3)

---

## Extending to Phase 2 & 3

**Phase 2 (real data):** Replace `generate_regime_data()` with `np.load("recorded_channel.npz")` — the windowing and feature extraction pipeline stays identical.

**Phase 3 (GNU Radio):** Run the NN as a ZMQ service (concept note Solution 2):
```
GNU Radio → ZMQ PUB (samples) → Python NN → ZMQ PUB (MCS cmd) → GNU Radio
```

---

## Known Limitations

1. AR(1) temporal model — higher order or full spectral shaping would improve long-horizon prediction
2. No aperture averaging correction (75mm lens reduces scintillation vs point receiver)
3. Class imbalance — uses inverse-frequency weighting; consider focal loss or SMOTE
