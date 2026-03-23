#!/usr/bin/env python3
"""Generate combined horizon sweep comparison plot."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Results from the three runs (extracted from output)
horizons = [1, 10, 50]
horizon_labels = ["1 (0.1ms)", "10 (1ms)", "50 (5ms)"]

# Throughput ratios per model per horizon
data = {
    "MLP":         [87.6, 84.8, 85.5],
    "CNN1D":       [88.8, 86.3, 85.7],
    "GRU":         [88.3, 85.2, 85.6],
    "LSTM":        [88.7, 85.3, 85.5],
    "HybridCNNGRU":[89.2, 85.4, 84.3],
    "TCN":         [88.2, 85.9, 83.4],
    "Transformer": [90.5, 85.6, 82.7],
    "Reactive_d0": [89.0, 81.4, 76.8],
}

# Over-prediction rates
over_data = {
    "MLP":         [14.9, 18.6, 15.3],
    "CNN1D":       [11.7, 18.0, 14.9],
    "GRU":         [13.5, 19.5, 14.1],
    "LSTM":        [12.4, 19.1, 14.0],
    "HybridCNNGRU":[13.5, 19.9, 17.8],
    "TCN":         [13.0, 19.9, 16.8],
    "Transformer": [12.2, 19.9, 22.0],
    "Reactive_d0": [14.3, 22.9, 24.2],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.tab10(np.linspace(0, 0.8, len(data)))
x = np.arange(len(horizons))
width = 0.09

# --- Throughput ratio ---
for i, (model, vals) in enumerate(data.items()):
    vals_frac = [v/100 for v in vals]
    ls = "--" if "Reactive" in model else "-"
    lw = 2.5 if "Reactive" in model else 1.5
    marker = "s" if "Reactive" in model else "o"
    ax1.plot(horizons, vals_frac, ls, marker=marker, linewidth=lw,
             color=colors[i], label=model, markersize=6)

ax1.set_xlabel("Prediction Horizon (samples)", fontsize=11)
ax1.set_ylabel("Throughput Ratio", fontsize=11)
ax1.set_title("Throughput vs Prediction Horizon\n(key thesis result)", 
              fontsize=12, fontweight="bold")
ax1.set_xticks(horizons)
ax1.set_xticklabels(horizon_labels)
ax1.set_ylim(0.72, 0.95)
ax1.legend(loc="lower left", fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Annotation
ax1.annotate("NNs maintain throughput\nat longer horizons",
             xy=(50, 0.855), xytext=(30, 0.92),
             arrowprops=dict(arrowstyle="->", color="gray"),
             fontsize=8, color="gray")
ax1.annotate("Reactive degrades\nwith horizon",
             xy=(50, 0.768), xytext=(30, 0.74),
             arrowprops=dict(arrowstyle="->", color="gray"),
             fontsize=8, color="gray")

# --- Over-prediction rate ---
for i, (model, vals) in enumerate(over_data.items()):
    vals_frac = [v/100 for v in vals]
    ls = "--" if "Reactive" in model else "-"
    lw = 2.5 if "Reactive" in model else 1.5
    marker = "s" if "Reactive" in model else "o"
    ax2.plot(horizons, vals_frac, ls, marker=marker, linewidth=lw,
             color=colors[i], label=model, markersize=6)

ax2.set_xlabel("Prediction Horizon (samples)", fontsize=11)
ax2.set_ylabel("Over-Prediction Rate (lower is better)", fontsize=11)
ax2.set_title("Over-Prediction Rate vs Horizon\n(packet error proxy)", 
              fontsize=12, fontweight="bold")
ax2.set_xticks(horizons)
ax2.set_xticklabels(horizon_labels)
ax2.legend(loc="upper left", fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
import os; os.makedirs("results", exist_ok=True)
fig.savefig("results/horizon_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/horizon_sweep.png")
