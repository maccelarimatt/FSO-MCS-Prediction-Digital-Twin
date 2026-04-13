#!/usr/bin/env python3
"""
Bar chart summary: throughput ratio and over-prediction rate per model.

Reads:  <results_dir>/results.json
Writes: <results_dir>/summary.png

Usage:
    python plots/plot_summary.py <results_dir>
    python plots/plot_summary.py results_h1
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir):
    path = os.path.join(results_dir, "results.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run run.py first.")
    with open(path) as f:
        return json.load(f)


def plot_summary(results_dir):
    data   = load_results(results_dir)
    meta   = data["meta"]
    models = data["models"]

    # Show NN models + Reactive_d0 only; skip adaptive variants and other reactive delays
    show = [m for m in models
            if m["model_type"] == "nn" or m["name"] == "Reactive_d0"]
    show.sort(key=lambda x: -x["throughput_ratio"])

    if not show:
        sys.exit("No models to plot.")

    names  = [m["name"] for m in show]
    thru   = [m["throughput_ratio"] for m in show]
    over   = [m["over_pred_rate"]   for m in show]
    colors = ["#c0392b" if "Reactive" in n else "#2980b9" for n in names]

    horizon_label = (f"horizon = {meta['horizon_samples']} samples "
                     f"({meta['horizon_ms']:.1f} ms)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))

    # --- Throughput ratio ---
    bars1 = ax1.bar(x, thru, 0.6, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_ylabel("Throughput Ratio")
    ax1.set_title(f"Throughput Ratio  [{horizon_label}]", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(1.0, color="green", ls="--", alpha=0.4, linewidth=1, label="ideal (100%)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars1, thru):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.1%}", ha="center", fontsize=8)

    # --- Over-prediction rate ---
    bars2 = ax2.bar(x, over, 0.6, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_ylabel("Over-Prediction Rate  (lower = safer)")
    ax2.set_title(f"Over-Prediction Rate  [{horizon_label}]", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0, max(over) * 1.35 + 0.01)
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, over):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                 f"{v:.1%}", ha="center", fontsize=8)

    fig.suptitle(f"Channel: {meta['channel_model']}  |  "
                 f"{meta['n_realisations']} realisations × {meta['duration_s']}s per regime",
                 fontsize=8, color="gray")

    plt.tight_layout()
    out = os.path.join(results_dir, "summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plots/plot_summary.py <results_dir>")
    plot_summary(sys.argv[1])
