#!/usr/bin/env python3
"""
<<<<<<< HEAD
FSO-TVWS MCS Prediction Digital Twin — Main Runner

Usage:
    python run.py                          # Quick demo (30s per regime)
    python run.py --full                   # Full dataset (300s × 3 reals)
    python run.py --horizon 10             # Custom prediction horizon
    python run.py --no-esn                 # Skip ESN (slow)
"""

import argparse, time, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fso_mcs_predictor import config
from fso_mcs_predictor.channel import print_regime_summary
from fso_mcs_predictor.dataset import generate_dataset
from fso_mcs_predictor.models import (
    get_model, ALL_MODELS, reactive_baseline, adaptive_selector
)
from fso_mcs_predictor.evaluate import evaluate, print_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-esn", action="store_true")
    parser.add_argument("--max-iter", type=int, default=80)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("FSO-TVWS MCS Prediction Digital Twin")
    print(f"Horizon: {args.horizon} samples ({args.horizon * 0.1:.1f} ms)")
    print("=" * 60)
    
    for r in config.TURBULENCE_REGIMES:
        print_regime_summary(r)
    
    # --- Generate dataset ---
    n_real = config.FULL_REALISATIONS if args.full else config.QUICK_REALISATIONS
    dur = config.FULL_DURATION_S if args.full else config.QUICK_DURATION_S
    
    print(f"\nGenerating dataset: {n_real} realisations × {dur}s per regime...")
    t0 = time.time()
    splits, class_weights, metadata = generate_dataset(
        n_realisations=n_real, duration_s=dur,
        horizon=args.horizon, seed=args.seed
    )
    print(f"Done in {time.time()-t0:.1f}s")
    
    train_X = splits["train"]["features"]
    train_y = splits["train"]["labels"]
    test_X = splits["test"]["features"]
    test_y = splits["test"]["labels"]
    test_snr = splits["test"]["raw_snr"]
    
    if len(test_y) == 0:
        print("ERROR: No test data. Use --full or increase duration.")
        return
    
    # --- Train & evaluate models ---
    model_names = args.models or ALL_MODELS.copy()
    if args.no_esn and "ESN" in model_names:
        model_names.remove("ESN")
    
    all_results = []
    all_preds = {}
    
    for name in model_names:
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        
        model = get_model(name, max_iter=args.max_iter)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        
        elapsed = time.time() - t0
        result = evaluate(preds, test_y, name=name)
        result["time"] = elapsed
        all_results.append(result)
        all_preds[name] = preds
        
        print(f"  {name}: acc={result['accuracy']:.1%} thru={result['throughput_ratio']:.1%} "
              f"over={result['over_pred_rate']:.1%} ({elapsed:.1f}s)")
        
        # Adaptive selector
        if test_snr is not None and len(test_snr) > 0:
            ada_preds = adaptive_selector(preds, test_snr, test_X)
            ada_name = f"Adaptive_{name}"
            ada_result = evaluate(ada_preds, test_y, name=ada_name)
            all_results.append(ada_result)
            all_preds[ada_name] = ada_preds
    
    # --- Reactive baselines ---
    if test_snr is not None and len(test_snr) > 0:
        print(f"\n--- Reactive AMC Baselines ---")
        for d in [0, 1, 5, 10, 20, 50, 90]:
            if d >= config.WINDOW_SIZE:
                continue
            preds = reactive_baseline(test_snr, delay=d)
            name = f"Reactive_d{d}"
            result = evaluate(preds, test_y, name=name)
            all_results.append(result)
            all_preds[name] = preds
            print(f"  {name}: acc={result['accuracy']:.1%} thru={result['throughput_ratio']:.1%}")
    
    # --- Results ---
    print_table(all_results)
    
    # --- Plots ---
    plot_mcs_timeseries(test_y, all_preds, args.output_dir, args.horizon)
    plot_bar_summary(all_results, args.output_dir, args.horizon)
    
    print(f"\nPlots saved to {args.output_dir}/")


def plot_mcs_timeseries(test_labels, predictions, output_dir, horizon):
    """Time plot: ideal MCS vs each model's prediction over a window."""
    n_show = min(600, len(test_labels))
    
    # Find the most interesting section (most MCS variation)
    best_start, best_score = 0, 0
    for i in range(0, len(test_labels) - n_show, 50):
        seg = test_labels[i:i+n_show]
        score = len(np.unique(seg)) + np.sum(np.abs(np.diff(seg)) > 0) * 0.01
        if score > best_score:
            best_score = score
            best_start = i
    
    sl = slice(best_start, best_start + n_show)
    test_labels = test_labels[sl]
    predictions = {k: v[sl] for k, v in predictions.items()}
    
    t_ms = np.arange(n_show) * config.DT * config.WINDOW_STRIDE * 1000
    
    # Select core models (skip adaptive variants for clarity)
    core = [k for k in predictions if not k.startswith("Adaptive_") 
            and not k.startswith("Reactive_d")]
    reactive_key = "Reactive_d0" if "Reactive_d0" in predictions else None
    
    n_panels = len(core) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 2.8 * n_panels),
                              sharex=True, gridspec_kw={"hspace": 0.08})
    if n_panels == 1:
        axes = [axes]
    
    cmap = plt.cm.Set2(np.linspace(0, 0.9, max(len(core), 1)))
    
    # Top panel: ideal + reactive
    ax = axes[0]
    ax.step(t_ms, test_labels[:n_show], where="mid", color="black",
            linewidth=1.5, label="Ideal MCS")
    if reactive_key:
        ax.step(t_ms, predictions[reactive_key][:n_show], where="mid",
                color="gray", linewidth=0.8, alpha=0.6, linestyle="--",
                label="Reactive (d=0)")
    ax.set_ylabel("MCS Class", fontsize=9)
    ax.set_ylim(-0.5, 14.5)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"MCS Prediction Comparison — horizon={horizon} "
                 f"({horizon*0.1:.1f} ms)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    
    # One panel per NN
    for i, model_name in enumerate(core):
        ax = axes[i + 1]
        preds = predictions[model_name][:n_show]
        ideal = test_labels[:n_show]
        
        over_mask = preds > ideal
        correct_mask = preds == ideal
        
        ax.step(t_ms, ideal, where="mid", color="black", linewidth=0.7, alpha=0.35)
        ax.step(t_ms, preds, where="mid", color=cmap[i % len(cmap)],
                linewidth=1.1, label=model_name)
        
        # Red shading for over-predictions
        for j in range(n_show - 1):
            if over_mask[j]:
                ax.axvspan(t_ms[j], t_ms[j+1], color="red", alpha=0.12)
        
        acc = correct_mask.mean()
        over_rate = over_mask.mean()
        ax.set_ylabel("MCS", fontsize=8)
        ax.set_ylim(-0.5, 14.5)
        ax.legend(loc="upper right", fontsize=7)
        ax.text(0.005, 0.92, f"Acc={acc:.0%}  Over={over_rate:.0%}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.6))
        ax.grid(True, alpha=0.2)
    
    axes[-1].set_xlabel("Time (ms)", fontsize=9)
    
    path = os.path.join(output_dir, "mcs_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_bar_summary(results, output_dir, horizon):
    """Bar chart: throughput ratio and over-prediction rate."""
    # Standalone NNs + Reactive d=0
    show = [r for r in results
            if not r["name"].startswith("Adaptive_")
            and not r["name"].startswith("Reactive_d")
            or r["name"] == "Reactive_d0"]
    show.sort(key=lambda x: -x["throughput_ratio"])
    
    if not show:
        return
    
    names = [r["name"] for r in show]
    thru = [r["throughput_ratio"] for r in show]
    over = [r["over_pred_rate"] for r in show]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    
    bars1 = ax1.bar(x, thru, 0.6, color="steelblue", edgecolor="white")
    ax1.set_ylabel("Throughput Ratio")
    ax1.set_title(f"Throughput Ratio (horizon={horizon})", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(1.0, color="green", ls="--", alpha=0.3)
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars1, thru):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{v:.1%}", ha="center", fontsize=7)
    
    bars2 = ax2.bar(x, over, 0.6, color="indianred", edgecolor="white")
    ax2.set_ylabel("Over-Prediction Rate")
    ax2.set_title(f"Over-Prediction Rate (horizon={horizon})", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, max(over) * 1.3 + 0.02)
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, over):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f"{v:.1%}", ha="center", fontsize=7)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "results_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

=======
Run the FSO-TVWS MCS Prediction experiment.

Usage (from repo root):
    python run.py                                    # full experiment
    python run.py --quick --models gru lstm          # quick test
    python run.py --no-context --output-dir results_ablation
    python run.py --mean-rx-power -15 --output-dir results_low_power

This is a convenience wrapper so you don't need:
    python -m fso_mcs_predictor.run_experiment
"""

from fso_mcs_predictor.run_experiment import main
>>>>>>> 74334969080a49abe0fa27248652f80d22e6b8d6

if __name__ == "__main__":
    main()
