"""Evaluation metrics for MCS prediction."""

import numpy as np
from . import config


def evaluate(preds, labels, name="Model"):
    """
    Compute all metrics. Returns dict.
    
    Key metrics:
    - throughput_ratio: primary ranking metric
    - over_pred_rate: critical safety metric (over-prediction = packet error = 0 throughput)
    """
    labels, preds = np.asarray(labels), np.asarray(preds)
    
    accuracy = (preds == labels).mean()
    pm1 = (np.abs(preds - labels) <= 1).mean()
    over = (preds > labels).mean()
    mae = np.abs(preds - labels).mean()
    
    # Throughput: over-predictions get zero (packet error)
    thr = config.MCS_THROUGHPUT
    ideal_thr = thr[labels].sum()
    actual_thr = np.where(preds <= labels, thr[preds], 0.0).sum()
    thr_ratio = actual_thr / max(ideal_thr, 1e-10)
    
    return {"name": name, "accuracy": accuracy, "pm1_accuracy": pm1,
            "over_pred_rate": over, "throughput_ratio": thr_ratio,
            "mae": mae, "n_samples": len(labels)}


def print_table(results):
    """Print sorted comparison table."""
    print(f"\n{'='*68}")
    print(f"{'Model':<22s} {'Acc':>6s} {'±1 Acc':>7s} {'OverPred':>9s} {'Thruput':>8s} {'MAE':>5s}")
    print(f"{'-'*68}")
    for r in sorted(results, key=lambda x: -x["throughput_ratio"]):
        print(f"{r['name']:<22s} {r['accuracy']:>6.1%} {r['pm1_accuracy']:>7.1%} "
              f"{r['over_pred_rate']:>9.1%} {r['throughput_ratio']:>8.1%} {r['mae']:>5.2f}")
    print(f"{'='*68}")
