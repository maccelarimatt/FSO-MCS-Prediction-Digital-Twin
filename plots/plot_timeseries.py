#!/usr/bin/env python3
"""
MCS time series: ideal MCS vs each model's prediction over the test set.
Red shading marks over-prediction windows (selected MCS > ideal → packet error).

Reads:  <results_dir>/results.json
        <results_dir>/predictions.npz
Writes: <results_dir>/mcs_timeseries.png

Usage:
    python plots/plot_timeseries.py <results_dir>
    python plots/plot_timeseries.py results_h1
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(results_dir):
    json_path = os.path.join(results_dir, "results.json")
    npz_path  = os.path.join(results_dir, "predictions.npz")
    if not os.path.exists(json_path):
        sys.exit(f"ERROR: {json_path} not found. Run run.py first.")
    if not os.path.exists(npz_path):
        sys.exit(f"ERROR: {npz_path} not found. Run run.py first.")
    with open(json_path) as f:
        results = json.load(f)
    npz = np.load(npz_path)
    return results, npz


def plot_timeseries(results_dir):
    results, npz = load_data(results_dir)
    meta    = results["meta"]
    physics = results["physics"]

    test_labels = npz["test_labels"].astype(int)

    # Reconstruct predictions dict from npz keys (prefix "pred__")
    predictions = {}
    for key in npz.files:
        if key.startswith("pred__"):
            name = key[len("pred__"):]
            predictions[name] = npz[key].astype(int)

    if len(test_labels) == 0:
        sys.exit("ERROR: test_labels array is empty.")

    # Find the most variable 600-sample segment for an informative plot
    n_show = min(600, len(test_labels))
    best_start, best_score = 0, 0
    for i in range(0, max(1, len(test_labels) - n_show), 50):
        seg   = test_labels[i:i + n_show]
        score = len(np.unique(seg)) + np.sum(np.abs(np.diff(seg)) > 0) * 0.01
        if score > best_score:
            best_score, best_start = score, i

    sl          = slice(best_start, best_start + n_show)
    test_labels = test_labels[sl]
    predictions = {k: v[sl] for k, v in predictions.items()}

    # Time axis: each plotted point is one window (spaced by window_stride samples × dt)
    dt_ms          = physics["dt_ms"]
    window_stride  = physics["window_stride_samples"]
    t_ms           = np.arange(n_show) * dt_ms * window_stride

    # Core NN models only — skip adaptive variants and all reactive-delay variants except d=0
    core         = [k for k in predictions
                    if not k.startswith("Adaptive_") and not k.startswith("Reactive_d")]
    reactive_key = "Reactive_d0" if "Reactive_d0" in predictions else None

    n_panels = len(core) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 2.8 * n_panels),
                             sharex=True, gridspec_kw={"hspace": 0.08})
    if n_panels == 1:
        axes = [axes]

    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(len(core), 1)))

    # --- Top panel: ideal MCS + reactive baseline ---
    ax = axes[0]
    ax.step(t_ms, test_labels, where="mid", color="black", linewidth=1.5, label="Ideal MCS")
    if reactive_key:
        ax.step(t_ms, predictions[reactive_key], where="mid",
                color="gray", linewidth=0.8, alpha=0.6, linestyle="--",
                label="Reactive (d=0, no prediction)")
    ax.set_ylabel("MCS Class", fontsize=9)
    ax.set_ylim(-0.5, 14.5)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        f"MCS Prediction vs Ideal  |  "
        f"horizon = {meta['horizon_samples']} samples ({meta['horizon_ms']:.1f} ms)  |  "
        f"channel: {meta['channel_model']}",
        fontsize=10, fontweight="bold"
    )
    ax.grid(True, alpha=0.2)

    # --- One panel per NN model ---
    for i, model_name in enumerate(core):
        ax    = axes[i + 1]
        preds = predictions[model_name]
        over_mask    = preds > test_labels
        correct_mask = preds == test_labels

        ax.step(t_ms, test_labels, where="mid", color="black", linewidth=0.7, alpha=0.35)
        ax.step(t_ms, preds, where="mid", color=cmap[i % len(cmap)],
                linewidth=1.1, label=model_name)

        # Shade over-prediction windows red
        for j in range(n_show - 1):
            if over_mask[j]:
                ax.axvspan(t_ms[j], t_ms[j + 1], color="red", alpha=0.12)

        acc       = correct_mask.mean()
        over_rate = over_mask.mean()
        ax.set_ylabel("MCS", fontsize=8)
        ax.set_ylim(-0.5, 14.5)
        ax.legend(loc="upper right", fontsize=7)
        ax.text(0.005, 0.92, f"Acc={acc:.0%}  Over={over_rate:.0%}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.6))
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (ms)", fontsize=9)

    out = os.path.join(results_dir, "mcs_timeseries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plots/plot_timeseries.py <results_dir>")
    plot_timeseries(sys.argv[1])
