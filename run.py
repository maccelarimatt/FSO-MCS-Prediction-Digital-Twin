#!/usr/bin/env python3
"""
FSO-TVWS MCS Prediction Digital Twin — Main Runner

Trains all models and saves results to disk. Does NOT generate plots.
Run the scripts in plots/ to generate figures after this completes.

Usage:
    python run.py                          # Quick demo (~2 min)
    python run.py --full                   # Full dataset (300s × 3 reals)
    python run.py --horizon 10             # Custom prediction horizon
    python run.py --no-esn                 # Skip ESN (slow)
    python run.py --horizon 1  --output-dir results_h1
    python run.py --horizon 10 --output-dir results_h10
    python run.py --horizon 50 --output-dir results_h50

After running, generate plots with:
    python plots/plot_summary.py    <output_dir>
    python plots/plot_timeseries.py <output_dir>
    python plots/plot_horizon_sweep.py results_h1 results_h10 results_h50
"""

import argparse, time, os, sys, json, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fso_mcs_predictor import config
from fso_mcs_predictor.channel import print_regime_summary, compute_turbulence_params
from fso_mcs_predictor.dataset import generate_dataset
from fso_mcs_predictor.models import (
    get_model, ALL_MODELS, reactive_baseline, adaptive_selector
)
from fso_mcs_predictor.evaluate import evaluate, print_table


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to plain Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(args, splits, class_weights, metadata, all_results, all_preds, test_y, test_snr):
    """
    Save all run results to:
      <output_dir>/results.json   — metrics, physics, metadata (human-readable)
      <output_dir>/predictions.npz — test labels + per-model predictions (compact binary)
    """
    # --- Turbulence regime physics (computed from config, not hardcoded) ---
    regime_info = {}
    for name, reg in config.TURBULENCE_REGIMES.items():
        params = compute_turbulence_params(reg["Cn2"], reg["wind_mean"])
        regime_info[name] = {
            "Cn2_m23":          reg["Cn2"],
            "wind_mean_ms":     reg["wind_mean"],
            "wind_std_ms":      reg["wind_std"],
            "r0_m":             float(params["r0"]),
            "tau0_ms":          float(params["tau0"] * 1000),
            "f_G_hz":           float(params["f_G"]),
            "sigma2_R":         float(params["sigma2_R"]),
            "sigma2_R_VK":      float(params["sigma2_R_VK"]),
            "aperture_avg_factor": float(params["aperture_avg"]),
            "alpha_GG":         float(params["alpha"]),
            "beta_GG":          float(params["beta"]),
        }

    # --- Annotate model results with type tag ---
    annotated_models = []
    for r in all_results:
        entry = dict(r)
        name = r["name"]
        if name.startswith("Reactive_"):
            entry["model_type"] = "reactive"
            entry["delay_samples"] = int(name.split("_d")[-1])
        elif name.startswith("Adaptive_"):
            entry["model_type"] = "adaptive"
            entry["base_model"] = name[len("Adaptive_"):]
        else:
            entry["model_type"] = "nn"
        annotated_models.append(entry)

    results_dict = {
        "meta": {
            "timestamp":        datetime.datetime.now().isoformat(),
            "horizon_samples":  args.horizon,
            "horizon_ms":       round(args.horizon * config.DT * 1000, 4),
            "seed":             args.seed,
            "full_mode":        args.full,
            "duration_s":       config.FULL_DURATION_S if args.full else config.QUICK_DURATION_S,
            "n_realisations":   config.FULL_REALISATIONS if args.full else config.QUICK_REALISATIONS,
            "max_iter":         args.max_iter,
            "output_dir":       args.output_dir,
            "channel_model":    "aotools PhaseScreenVonKarman + Fresnel angular spectrum",
        },
        "physics": {
            "wavelength_nm":            config.WAVELENGTH * 1e9,
            "link_distance_m":          config.LINK_DISTANCE,
            "tx_power_dbm":             config.TX_POWER_DBM,
            "rx_aperture_mm":           config.RX_APERTURE_D * 1e3,
            "sample_rate_hz":           config.SAMPLE_RATE,
            "dt_ms":                    config.DT * 1000,
            "outer_scale_L0_m":         config.L0,
            "snr_model":                "SNR_dB = 1.04 * P_dBm + 33.1",
            "snr_slope":                config.SNR_SLOPE,
            "snr_intercept_db":         config.SNR_INTERCEPT,
            "window_size_samples":      config.WINDOW_SIZE,
            "window_stride_samples":    config.WINDOW_STRIDE,
            "si_window_samples":        config.SI_WINDOW,
            "n_features":               config.N_FEATURES,
            "n_mcs_classes":            config.NUM_CLASSES,
            "hysteresis_db":            config.HYSTERESIS_DB,
            "mcs_snr_thresholds_db":    config.MCS_SNR_THRESHOLDS.tolist(),
            "mcs_throughputs_normalised": config.MCS_THROUGHPUT.tolist(),
            "mcs_table": [
                {"class": m["class"], "name": m["name"], "snr_min_db": m["snr_min"]}
                for m in config.MCS_TABLE
            ],
        },
        "turbulence_regimes": regime_info,
        "dataset": {
            "n_train":              int(len(splits["train"]["labels"])),
            "n_val":                int(len(splits["val"]["labels"])),
            "n_test":               int(len(splits["test"]["labels"])),
            "class_counts_train":   [int(x) for x in metadata.get("class_counts", [])],
            "class_weights":        class_weights.tolist(),
            "regime_metadata":      metadata.get("regimes", {}),
        },
        "models": annotated_models,
    }

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2, cls=_NumpyEncoder)
    print(f"Saved: {json_path}")

    # --- Save predictions as compressed numpy archive ---
    npz_data = {"test_labels": test_y.astype(np.int16)}
    if test_snr is not None and len(test_snr) > 0:
        npz_data["test_snr"] = test_snr.astype(np.float32)

    for name, preds in all_preds.items():
        # prefix "pred__" + sanitised name; double-underscore is the separator
        key = "pred__" + name.replace(" ", "_").replace("-", "_")
        npz_data[key] = preds.astype(np.int16)

    npz_path = os.path.join(args.output_dir, "predictions.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved: {npz_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",       action="store_true")
    parser.add_argument("--horizon",    type=int, default=1)
    parser.add_argument("--models",     nargs="+", default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-esn",     action="store_true")
    parser.add_argument("--max-iter",   type=int, default=80)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("FSO-TVWS MCS Prediction Digital Twin")
    print(f"Horizon: {args.horizon} samples ({args.horizon * config.DT * 1000:.1f} ms)")
    print("=" * 60)

    for r in config.TURBULENCE_REGIMES:
        print_regime_summary(r)

    # --- Generate dataset ---
    n_real = config.FULL_REALISATIONS if args.full else config.QUICK_REALISATIONS
    dur    = config.FULL_DURATION_S   if args.full else config.QUICK_DURATION_S

    print(f"\nGenerating dataset: {n_real} realisations × {dur}s per regime...")
    t0 = time.time()
    splits, class_weights, metadata = generate_dataset(
        n_realisations=n_real, duration_s=dur,
        horizon=args.horizon, seed=args.seed
    )
    print(f"Done in {time.time()-t0:.1f}s")

    train_X  = splits["train"]["features"]
    train_y  = splits["train"]["labels"]
    test_X   = splits["test"]["features"]
    test_y   = splits["test"]["labels"]
    test_snr = splits["test"]["raw_snr"]

    if len(test_y) == 0:
        print("ERROR: No test data. Use --full or increase duration.")
        return

    # --- Train & evaluate models ---
    model_names = args.models or ALL_MODELS.copy()
    if args.no_esn and "ESN" in model_names:
        model_names.remove("ESN")

    all_results = []
    all_preds   = {}

    for name in model_names:
        print(f"\n--- Training {name} ---")
        t0 = time.time()

        model = get_model(name, max_iter=args.max_iter)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)

        elapsed = time.time() - t0
        result  = evaluate(preds, test_y, name=name)
        result["train_time_s"] = round(elapsed, 2)
        all_results.append(result)
        all_preds[name] = preds

        print(f"  {name}: acc={result['accuracy']:.1%}  thru={result['throughput_ratio']:.1%}  "
              f"over={result['over_pred_rate']:.1%}  ({elapsed:.1f}s)")

        # Adaptive selector (SI-gated hybrid)
        if test_snr is not None and len(test_snr) > 0:
            ada_preds  = adaptive_selector(preds, test_snr, test_X)
            ada_name   = f"Adaptive_{name}"
            ada_result = evaluate(ada_preds, test_y, name=ada_name)
            all_results.append(ada_result)
            all_preds[ada_name] = ada_preds

    # --- Reactive baselines ---
    if test_snr is not None and len(test_snr) > 0:
        print(f"\n--- Reactive AMC Baselines ---")
        for d in [0, 1, 5, 10, 20, 50, 90]:
            if d >= config.WINDOW_SIZE:
                continue
            preds  = reactive_baseline(test_snr, delay=d)
            name   = f"Reactive_d{d}"
            result = evaluate(preds, test_y, name=name)
            all_results.append(result)
            all_preds[name] = preds
            print(f"  {name}: acc={result['accuracy']:.1%}  thru={result['throughput_ratio']:.1%}")

    # --- Print summary table to terminal ---
    print_table(all_results)

    # --- Persist everything ---
    save_results(args, splits, class_weights, metadata, all_results, all_preds, test_y, test_snr)

    print(f"\nTo generate plots run:")
    print(f"  python plots/plot_summary.py    {args.output_dir}")
    print(f"  python plots/plot_timeseries.py {args.output_dir}")
    print(f"  python plots/plot_horizon_sweep.py results_h1 results_h10 results_h50")


if __name__ == "__main__":
    main()
