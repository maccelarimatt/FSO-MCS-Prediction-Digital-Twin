"""
Evaluation Engine
=================
Comprehensive evaluation of MCS predictors with metrics relevant
to the FSO-TVWS application:
  - Classification accuracy (overall and per-class)
  - Confusion matrix  
  - Throughput gain vs reactive baseline
  - "Dangerous" misprediction rate (predicting too high an MCS)
  - Per-regime performance breakdown
  - Inference latency
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Optional
from pathlib import Path
from torch.utils.data import DataLoader

from .config import MCS_TABLE, NUM_MCS_CLASSES
from .models.base import BaseMCSPredictor
from .dataset.generator import FSODataset
from .system.snr_model import MCSSelector


class Evaluator:
    """
    Evaluates a trained MCS predictor on a test dataset.
    
    Key metrics beyond standard accuracy:
      1. Over-prediction rate: How often does the model predict a 
         higher MCS than the channel can support? This directly 
         causes packet errors in the real system.
      2. Under-prediction rate: How often does the model select a 
         lower MCS than optimal? This wastes throughput.
      3. Throughput ratio: Predicted throughput / ideal throughput.
      4. Per-regime breakdown: How well does the model generalise?
    """
    
    def __init__(
        self,
        model: BaseMCSPredictor,
        test_dataset: FSODataset,
        device: str = "auto",
        batch_size: int = 512,
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.dataset = test_dataset
        self.loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0,
        )
        self.mcs_selector = MCSSelector()
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Run full evaluation suite.
        
        Returns dict with all metrics.
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        # Inference timing
        start_time = time.time()
        n_batches = 0
        
        for features, targets in self.loader:
            features = features.to(self.device)
            logits = self.model(features)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())
            n_batches += 1
        
        inference_time = time.time() - start_time
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        n = len(preds)
        
        # --- Core metrics ---
        accuracy = np.mean(preds == targets)
        
        # Confusion matrix
        cm = np.zeros((NUM_MCS_CLASSES, NUM_MCS_CLASSES), dtype=np.int64)
        for t, p in zip(targets, preds):
            cm[t, p] += 1
        
        # Per-class accuracy
        per_class_acc = {}
        for c in range(NUM_MCS_CLASSES):
            mask = targets == c
            if mask.sum() > 0:
                per_class_acc[c] = np.mean(preds[mask] == c)
            else:
                per_class_acc[c] = float('nan')
        
        # --- Application-specific metrics ---
        
        # Over-prediction: predicted MCS higher than ideal
        # This is the dangerous case — causes packet errors
        over_pred_mask = preds > targets
        over_prediction_rate = np.mean(over_pred_mask)
        
        # Under-prediction: predicted MCS lower than ideal
        # This wastes throughput but doesn't cause errors
        under_pred_mask = preds < targets
        under_prediction_rate = np.mean(under_pred_mask)
        
        # Adjacent accuracy: within ±1 MCS of ideal
        adjacent_acc = np.mean(np.abs(preds.astype(int) - targets.astype(int)) <= 1)
        
        # Throughput analysis
        pred_throughput = self.mcs_selector.mcs_to_throughput(preds)
        ideal_throughput = self.mcs_selector.mcs_to_throughput(targets)
        
        # For over-predictions, actual throughput is 0 (packet error)
        actual_throughput = pred_throughput.copy()
        actual_throughput[over_pred_mask] = 0.0
        
        throughput_ratio = (
            np.sum(actual_throughput) / max(np.sum(ideal_throughput), 1.0)
        )
        
        # Mean absolute MCS error
        mae_mcs = np.mean(np.abs(preds.astype(float) - targets.astype(float)))
        
        # --- Per-regime breakdown ---
        regime_metrics = {}
        regime_labels = self.dataset.regime_labels
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            r_preds = preds[mask]
            r_targets = targets[mask]
            
            r_over = np.mean(r_preds > r_targets) if len(r_preds) > 0 else 0
            r_acc = np.mean(r_preds == r_targets) if len(r_preds) > 0 else 0
            
            regime_metrics[str(regime)] = {
                "accuracy": float(r_acc),
                "over_prediction_rate": float(r_over),
                "n_samples": int(mask.sum()),
            }
        
        # --- Inference speed ---
        samples_per_second = n / max(inference_time, 1e-6)
        
        results = {
            "model_name": self.model.__class__.__name__,
            "n_parameters": self.model.count_parameters(),
            "n_test_samples": n,
            "accuracy": float(accuracy),
            "adjacent_accuracy": float(adjacent_acc),
            "over_prediction_rate": float(over_prediction_rate),
            "under_prediction_rate": float(under_prediction_rate),
            "throughput_ratio": float(throughput_ratio),
            "mae_mcs": float(mae_mcs),
            "per_class_accuracy": {int(k): float(v) for k, v in per_class_acc.items()},
            "confusion_matrix": cm.tolist(),
            "regime_metrics": regime_metrics,
            "inference_time_s": float(inference_time),
            "samples_per_second": float(samples_per_second),
        }
        
        return results
    
    @staticmethod
    def print_results(results: Dict):
        """Pretty-print evaluation results."""
        print(f"\n{'='*60}")
        print(f"  {results['model_name']} — Test Results")
        print(f"{'='*60}")
        print(f"  Parameters:          {results['n_parameters']:,}")
        print(f"  Test samples:        {results['n_test_samples']:,}")
        print(f"  Accuracy:            {results['accuracy']:.1%}")
        print(f"  Adjacent (±1) Acc:   {results['adjacent_accuracy']:.1%}")
        print(f"  Over-prediction:     {results['over_prediction_rate']:.1%}  ← packet errors")
        print(f"  Under-prediction:    {results['under_prediction_rate']:.1%}  ← wasted throughput")
        print(f"  Throughput ratio:    {results['throughput_ratio']:.1%} of ideal")
        print(f"  MAE (MCS levels):    {results['mae_mcs']:.2f}")
        print(f"  Inference speed:     {results['samples_per_second']:.0f} samples/s")
        
        if results.get('regime_metrics'):
            print(f"\n  Per-Regime Breakdown:")
            for regime, metrics in results['regime_metrics'].items():
                print(f"    {regime:30s}  acc={metrics['accuracy']:.1%}  "
                      f"over={metrics['over_prediction_rate']:.1%}  "
                      f"n={metrics['n_samples']}")
    
    @staticmethod
    def compare_models(all_results: List[Dict]) -> str:
        """
        Generate a comparison table across all evaluated models.
        Returns formatted string.
        """
        header = (
            f"{'Model':>20} | {'Params':>8} | {'Acc':>6} | {'±1 Acc':>6} | "
            f"{'Over%':>6} | {'Thru%':>6} | {'MAE':>5} | {'Speed':>10} | "
            f"{'Train(s)':>8}"
        )
        lines = ["\n" + "=" * len(header), "  MODEL COMPARISON", "=" * len(header)]
        lines.append(header)
        lines.append("-" * len(header))
        
        for r in sorted(all_results, key=lambda x: -x['throughput_ratio']):
            train_time = r.get('training_time_s', 0)
            lines.append(
                f"{r['model_name']:>20} | "
                f"{r['n_parameters']:>8,} | "
                f"{r['accuracy']:>5.1%} | "
                f"{r['adjacent_accuracy']:>5.1%} | "
                f"{r['over_prediction_rate']:>5.1%} | "
                f"{r['throughput_ratio']:>5.1%} | "
                f"{r['mae_mcs']:>5.2f} | "
                f"{r['samples_per_second']:>8.0f}/s | "
                f"{train_time:>7.1f}"
            )
        
        lines.append("=" * len(header))
        return "\n".join(lines)
    
    @staticmethod
    def save_results(results: Dict, filepath: str):
        """Save results to JSON for later analysis."""
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        clean = json.loads(json.dumps(results, default=convert))
        with open(filepath, 'w') as f:
            json.dump(clean, f, indent=2)
