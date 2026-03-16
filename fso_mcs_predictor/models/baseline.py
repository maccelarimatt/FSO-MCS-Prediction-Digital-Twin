"""
Reactive AMC Baseline
======================
Implements the traditional Adaptive Modulation and Coding strategy
that Mikaeel's system already uses (Section II-A, Dindar et al.).

How it works in the real system:
  1. Receiver measures current SNR from received signal
  2. Applies MCS threshold table (Table I) to select best MCS
  3. Feeds this back to the transmitter
  4. Transmitter uses it for the NEXT frame

The problem is step 3 takes time. By the time the transmitter gets
the feedback, the channel has moved on. This baseline quantifies
how much throughput is lost to that staleness.

How this implementation works:
  Each test sample contains a window of 100 consecutive SNR values
  (raw, un-normalised). The last position (index 99) is the most
  recent observation before the target. The target is the ideal MCS
  at the NEXT timestep after the window.

  For delay=d, the reactive system bases its MCS decision on the
  SNR at position (window_size - 1 - d), i.e. d samples before the
  most recent observation. It applies the MCS threshold table to
  that (potentially stale) SNR and uses that as its prediction.

  At 10 kHz sample rate:
    delay=0  → 0 ms   (sees current SNR, nearly perfect)
    delay=10 → 1 ms   (≈ τ₀ for strong turbulence)
    delay=50 → 5 ms   (typical processing + feedback delay)
    delay=99 → 9.9 ms (maximum lookback within window)
"""

import numpy as np
from typing import Dict, List
from ..config import MCS_TABLE, NUM_MCS_CLASSES
from ..system.snr_model import MCSSelector
from ..dataset.generator import FSODataset


class ReactiveAMCBaseline:
    """
    Simulates the traditional reactive AMC system.
    
    For each test sample, looks at the raw SNR from `delay` steps
    before the prediction target, applies MCS thresholds, and uses
    that as the prediction. Shuffle-safe because it operates within
    each sample's window, not across samples.
    """
    
    def __init__(
        self,
        delay_samples: int = 0,
        hysteresis_dB: float = 0.5,
    ):
        self.delay = delay_samples
        self.selector = MCSSelector(hysteresis_dB=hysteresis_dB)
        self._name = f"ReactiveAMC_d{delay_samples}"
    
    def evaluate_on_dataset(self, test_dataset: FSODataset) -> Dict:
        """
        Evaluate reactive AMC on the test dataset.
        
        Uses the raw (un-normalised) SNR stored at each window position
        to apply MCS thresholds at the appropriate delay.
        """
        raw_snr = test_dataset.raw_snr_windows     # (n, window_size)
        targets = test_dataset.targets.numpy()       # (n,)
        
        if raw_snr is None:
            raise ValueError(
                "Test dataset does not contain raw_snr_windows. "
                "Regenerate the dataset with the updated generator."
            )
        
        n_samples = len(targets)
        window_size = raw_snr.shape[1]
        
        # The last position in the window (index window_size-1) is the
        # most recent observation before the target. Going back by
        # `delay` positions gives us the SNR the reactive system sees.
        lookback_idx = max(0, window_size - 1 - self.delay)
        
        # Get the SNR at the delayed position for every sample
        delayed_snr = raw_snr[:, lookback_idx]
        
        # Apply MCS threshold table (same as the real system)
        preds = self.selector.select(delayed_snr)
        
        # --- Compute metrics (same format as Evaluator) ---
        accuracy = np.mean(preds == targets)
        over_pred = np.mean(preds > targets)
        under_pred = np.mean(preds < targets)
        adjacent = np.mean(np.abs(preds.astype(int) - targets.astype(int)) <= 1)
        mae = np.mean(np.abs(preds.astype(float) - targets.astype(float)))
        
        # Throughput calculation
        pred_tp = self.selector.mcs_to_throughput(preds)
        ideal_tp = self.selector.mcs_to_throughput(targets)
        actual_tp = pred_tp.copy()
        # Over-predictions cause packet errors -> zero throughput for those frames
        actual_tp[preds > targets] = 0.0
        tp_ratio = np.sum(actual_tp) / max(np.sum(ideal_tp), 1.0)
        
        # Per-regime breakdown
        regime_metrics = {}
        for regime in np.unique(test_dataset.regime_labels):
            mask = test_dataset.regime_labels == regime
            r_p, r_t = preds[mask], targets[mask]
            if len(r_p) > 0:
                r_tp_pred = self.selector.mcs_to_throughput(r_p)
                r_tp_ideal = self.selector.mcs_to_throughput(r_t)
                r_tp_actual = r_tp_pred.copy()
                r_tp_actual[r_p > r_t] = 0.0
                regime_metrics[str(regime)] = {
                    "accuracy": float(np.mean(r_p == r_t)),
                    "over_prediction_rate": float(np.mean(r_p > r_t)),
                    "throughput_ratio": float(
                        np.sum(r_tp_actual) / max(np.sum(r_tp_ideal), 1.0)
                    ),
                    "n_samples": int(mask.sum()),
                }
        
        return {
            "model_name": self._name,
            "n_parameters": 0,
            "n_test_samples": n_samples,
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
        }


def run_baseline_sweep(
    test_dataset: FSODataset,
    delays: List[int] = None,
) -> List[Dict]:
    """
    Run the reactive AMC baseline at multiple delay values.
    
    Delays are in SAMPLES at 10 kHz, so:
      0 samples  = 0 ms    (genie-aided, perfect)
      1 sample   = 0.1 ms  (near-instantaneous feedback)
      10 samples = 1 ms    (approx tau_0 for strong turbulence)
      20 samples = 2 ms    (moderate processing delay)
      50 samples = 5 ms    (typical feedback loop)
      90 samples = 9 ms    (slow feedback, near window edge)
    
    Note: max delay is (window_size - 1). Delays beyond that
    are clamped to position 0 in the window.
    """
    if delays is None:
        delays = [0, 1, 5, 10, 20, 50, 90]
    
    # Deduplicate and sort
    delays = sorted(set(delays))
    
    results = []
    print("\nReactive AMC Baseline Sweep")
    print(f"  {'Delay':>6} | {'ms':>6} | {'Acc':>6} | {'+-1Acc':>6} | {'Over%':>6} | {'Thru%':>6}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    
    for d in delays:
        baseline = ReactiveAMCBaseline(delay_samples=d)
        r = baseline.evaluate_on_dataset(test_dataset)
        results.append(r)
        
        delay_ms = d / 10.0  # 10 kHz sample rate
        print(f"  {d:>6} | {delay_ms:>5.1f}ms | "
              f"{r['accuracy']:>5.1%} | {r['adjacent_accuracy']:>5.1%} | "
              f"{r['over_prediction_rate']:>5.1%} | "
              f"{r['throughput_ratio']:>5.1%}")
    
    return results
