#!/usr/bin/env python3
"""
Horizon sweep: compare throughput ratio and over-prediction rate across
multiple prediction horizons. All values are read from results.json files
produced by run.py — nothing is hardcoded.

Reads:  <dir>/results.json  for each supplied directory
Writes: horizon_sweep.png  in the first supplied directory (or --output)

Usage:
    python plots/plot_horizon_sweep.py results_h1 results_h10 results_h50
    python plots/plot_horizon_sweep.py results_h1 results_h10 results_h50 --output results_combined
"""

import sys, os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir):
    path = os.path.join(results_dir, "results.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run run.py --output-dir {results_dir} first.")
    with open(path) as f:
        return json.load(f)


def plot_horizon_sweep(result_dirs, output_dir=None):
    # Load and sort by horizon
    all_data = [load_results(d) for d in result_dirs]
    all_data.sort(key=lambda x: x["meta"]["horizon_samples"])

    horizons       = [d["meta"]["horizon_samples"] for d in all_data]
    horizon_labels = [f"{d['meta']['horizon_samples']}\n({d['meta']['horizon_ms']:.1f} ms)"
                      for d in all_data]

    # Collect model names present across all runs: NNs + Reactive_d0
    # Preserve insertion order so the legend is stable
    seen = {}
    for d in all_data:
        for m in d["models"]:
            if m["model_type"] == "nn" or m["name"] == "Reactive_d0":
                seen[m["name"]] = True
    model_names = list(seen.keys())

    # Build metric matrices: model → list of values (one per horizon)
    throughput = {n: [] for n in model_names}
    over_pred  = {n: [] for n in model_names}

    for d in all_data:
        lookup = {m["name"]: m for m in d["models"]}
        for n in model_names:
            if n in lookup:
                throughput[n].append(lookup[n]["throughput_ratio"])
                over_pred[n].append(lookup[n]["over_pred_rate"])
            else:
                throughput[n].append(None)
                over_pred[n].append(None)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, name in enumerate(model_names):
        hs  = [horizons[j] for j, v in enumerate(throughput[name]) if v is not None]
        ts  = [v for v in throughput[name] if v is not None]
        ovs = [v for v in over_pred[name]  if v is not None]

        ls     = "--" if "Reactive" in name else "-"
        lw     = 2.5 if "Reactive" in name else 1.5
        marker = "s" if "Reactive" in name else "o"

        ax1.plot(hs, ts,  ls, marker=marker, linewidth=lw,
                 color=colors[i], label=name, markersize=6)
        ax2.plot(hs, ovs, ls, marker=marker, linewidth=lw,
                 color=colors[i], label=name, markersize=6)

    ax1.set_xlabel("Prediction Horizon (samples)", fontsize=11)
    ax1.set_ylabel("Throughput Ratio", fontsize=11)
    ax1.set_title("Throughput Ratio vs Prediction Horizon", fontsize=12, fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.set_xticklabels(horizon_labels)
    ax1.legend(loc="lower left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Prediction Horizon (samples)", fontsize=11)
    ax2.set_ylabel("Over-Prediction Rate  (lower = safer)", fontsize=11)
    ax2.set_title("Over-Prediction Rate vs Prediction Horizon", fontsize=12, fontweight="bold")
    ax2.set_xticks(horizons)
    ax2.set_xticklabels(horizon_labels)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Subtitle shows channel model and run config from the first result file
    first_meta = all_data[0]["meta"]
    fig.suptitle(
        f"Channel: {first_meta['channel_model']}  |  "
        f"{first_meta['n_realisations']} realisations × {first_meta['duration_s']}s per regime",
        fontsize=9, color="gray"
    )

    plt.tight_layout()

    out_dir = output_dir or result_dirs[0]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "horizon_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+",
                        help="Result directories to compare (one per horizon run)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for the sweep plot (default: first result dir)")
    args = parser.parse_args()

    if len(args.result_dirs) < 2:
        print("Warning: supply ≥ 2 result directories for a meaningful comparison.")

    plot_horizon_sweep(args.result_dirs, args.output)
