"""
Hybrid MCS Selector
====================
Combines the NN predictor with a reactive SNR measurement to avoid
the bottleneck where the NN caps MCS below the channel's capability.

The Problem:
  At -10 dBm, MCS levels 13-14 appear in <0.05% of training samples.
  The NN learns to never predict them. When conditions ARE good enough
  for MCS 13 (rare but real), the NN under-predicts by 1-2 levels,
  wasting throughput. Meanwhile, the reactive system has no problem
  selecting MCS 13 in stable, good conditions.

The Solution:
  Use the NN when the channel is volatile (high SI, fast changes).
  Use the reactive measurement when the channel is stable (low SI).
  The scintillation index itself tells you which to trust.

  When SI is low → channel is stable → reactive SNR is reliable
    → use reactive MCS selection (no delay penalty in stable channels)
  When SI is high → channel is volatile → reactive SNR is stale
    → use NN prediction (exploits temporal correlations)

This is not a hack — it's how real adaptive systems work. LTE uses
both inner-loop (fast, reactive) and outer-loop (slow, statistical)
adaptation for exactly this reason.

For the thesis:
  This hybrid approach should be presented as the deployment
  recommendation. The NN comparison (Phase 1) identifies which
  architecture to use for the "volatile channel" branch. The
  reactive system handles the "stable channel" branch where it
  already works well.
"""

import numpy as np
from typing import Dict, Optional
from ..system.snr_model import MCSSelector
from ..dataset.generator import FSODataset


class HybridMCSSelector:
    """
    Combines NN prediction with reactive AMC based on channel stability.
    
    Decision logic:
      1. Compute current SI from the observation window
      2. If SI < threshold → channel is stable → use reactive MCS
         (direct SNR threshold lookup, no delay penalty because 
          stable channels don't change between measurement and TX)
      3. If SI ≥ threshold → channel is volatile → use NN prediction
         (NN has learned to anticipate fades from temporal patterns)
    
    This naturally handles the rare-class problem: in stable conditions
    where MCS 13-14 are achievable, the reactive system correctly
    selects them. In volatile conditions where prediction matters,
    the NN handles it.
    """
    
    def __init__(
        self,
        si_threshold: float = 0.05,
        hysteresis_dB: float = 0.5,
    ):
        """
        Args:
            si_threshold: SI below this → use reactive. 
                         0.05 corresponds roughly to the boundary between
                         weak and moderate turbulence (Rytov σ²_R ≈ 0.07).
                         In weak turbulence, fades are shallow and slow
                         enough that reactive AMC works fine.
            hysteresis_dB: SNR margin for reactive MCS selection.
        """
        self.si_threshold = si_threshold
        self.selector = MCSSelector(hysteresis_dB=hysteresis_dB)
    
    def select(
        self,
        nn_predictions: np.ndarray,
        raw_snr_windows: np.ndarray,
        si_values: np.ndarray,
    ) -> np.ndarray:
        """
        For each sample, choose between NN prediction and reactive.
        
        Args:
            nn_predictions: NN's MCS class predictions (n_samples,)
            raw_snr_windows: Un-normalised SNR windows (n_samples, window_size)
            si_values: SI estimate at each sample (n_samples,)
        
        Returns:
            hybrid_predictions: MCS class predictions (n_samples,)
        """
        n = len(nn_predictions)
        hybrid = np.zeros(n, dtype=np.int64)
        
        # Reactive prediction: use the most recent SNR in the window
        latest_snr = raw_snr_windows[:, -1]  # last position = most recent
        reactive_mcs = self.selector.select(latest_snr)
        
        # Decision: use SI to choose which prediction to trust
        stable_mask = si_values < self.si_threshold
        volatile_mask = ~stable_mask
        
        hybrid[stable_mask] = reactive_mcs[stable_mask]
        hybrid[volatile_mask] = nn_predictions[volatile_mask]
        
        return hybrid
    
    def evaluate_on_dataset(
        self,
        nn_predictions: np.ndarray,
        test_dataset: FSODataset,
    ) -> Dict:
        """
        Evaluate the hybrid selector on the test dataset.
        
        Returns metrics in the same format as Evaluator and 
        ReactiveAMCBaseline for direct comparison.
        """
        targets = test_dataset.targets.numpy()
        raw_snr = test_dataset.raw_snr_windows
        si_values = test_dataset.si_values.numpy()
        
        # Get hybrid predictions
        preds = self.select(nn_predictions, raw_snr, si_values)
        
        # Count how many went to each branch
        stable_count = np.sum(si_values < self.si_threshold)
        volatile_count = len(si_values) - stable_count
        
        # Standard metrics
        accuracy = np.mean(preds == targets)
        over_pred = np.mean(preds > targets)
        under_pred = np.mean(preds < targets)
        adjacent = np.mean(np.abs(preds.astype(int) - targets.astype(int)) <= 1)
        mae = np.mean(np.abs(preds.astype(float) - targets.astype(float)))
        
        # Throughput
        pred_tp = self.selector.mcs_to_throughput(preds)
        ideal_tp = self.selector.mcs_to_throughput(targets)
        actual_tp = pred_tp.copy()
        actual_tp[preds > targets] = 0.0
        tp_ratio = np.sum(actual_tp) / max(np.sum(ideal_tp), 1.0)
        
        # Per-regime
        regime_metrics = {}
        for regime in np.unique(test_dataset.regime_labels):
            mask = test_dataset.regime_labels == regime
            r_p, r_t = preds[mask], targets[mask]
            if len(r_p) > 0:
                regime_metrics[str(regime)] = {
                    "accuracy": float(np.mean(r_p == r_t)),
                    "over_prediction_rate": float(np.mean(r_p > r_t)),
                    "n_samples": int(mask.sum()),
                }
        
        return {
            "model_name": f"Hybrid(SI<{self.si_threshold})",
            "n_parameters": 0,
            "n_test_samples": len(targets),
            "accuracy": float(accuracy),
            "adjacent_accuracy": float(adjacent),
            "over_prediction_rate": float(over_pred),
            "under_prediction_rate": float(under_pred),
            "throughput_ratio": float(tp_ratio),
            "mae_mcs": float(mae),
            "per_class_accuracy": {},
            "regime_metrics": regime_metrics,
            "inference_time_s": 0.0,
            "samples_per_second": float('inf'),
            "training_time_s": 0.0,
            "epochs_trained": 0,
            "stable_samples": int(stable_count),
            "volatile_samples": int(volatile_count),
            "stable_fraction": float(stable_count / len(si_values)),
        }
