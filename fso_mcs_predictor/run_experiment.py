#!/usr/bin/env python3
"""
run_experiment.py -- Phase 1 Experiment Runner
==============================================
Generates FSO channel datasets, trains all neural network architectures,
evaluates them, and produces a comparison report.

Usage:
  python -m fso_mcs_predictor.run_experiment                    # full experiment
  python -m fso_mcs_predictor.run_experiment --models lstm gru  # specific models
  python -m fso_mcs_predictor.run_experiment --quick            # fast debug run
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

from fso_mcs_predictor.config import (
    DatasetConfig, LinkBudget, TrainingConfig,
    TURBULENCE_REGIMES, NUM_MCS_CLASSES,
)
from fso_mcs_predictor.channel.turbulence import TurbulenceParameters
from fso_mcs_predictor.dataset.generator import DatasetGenerator
from fso_mcs_predictor.models import create_model, list_models
from fso_mcs_predictor.models.baseline import run_baseline_sweep
from fso_mcs_predictor.models.hybrid_selector import HybridMCSSelector
from fso_mcs_predictor.trainer import Trainer
from fso_mcs_predictor.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="FSO-TVWS MCS Prediction -- Phase 1 Experiment"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to train. Available: {list_models()}. Default: all."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with reduced data and epochs (for debugging)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory for results output."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--mean-rx-power", type=float, default=-10.0,
        help="Mean received optical power in dBm."
    )
    parser.add_argument(
        "--no-context", action="store_true",
        help="Disable SI context input (ablation study)."
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Prediction horizon in samples (at 10 kHz: 1=0.1ms, 10=1ms, 50=5ms). "
             "Controls how far ahead the NN predicts. The reactive baseline "
             "with the same delay is the thing to beat."
    )
    parser.add_argument(
        "--focal-loss", action="store_true",
        help="Use focal loss instead of standard weighted cross-entropy (experimental)."
    )
    return parser.parse_args()


def print_turbulence_summary():
    """Print the turbulence regime parameters for reference."""
    print("\n" + "=" * 60)
    print("  TURBULENCE REGIMES")
    print("=" * 60)
    link = LinkBudget()
    for name, regime in TURBULENCE_REGIMES.items():
        tp = TurbulenceParameters(
            Cn2=regime.Cn2,
            wavelength_m=link.wavelength_m,
            link_distance_m=link.link_distance_m,
        )
        print(f"\n{tp.summary(regime.wind_speed_m_s)}")
    print()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = args.models or list_models()

    print("=" * 60)
    print("  FSO-TVWS MCS Prediction -- Phase 1 Experiment")
    print("=" * 60)
    print(f"  Models:        {model_names}")
    print(f"  Quick mode:    {args.quick}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Mean Rx power: {args.mean_rx_power} dBm")
    print(f"  SI context:    {not args.no_context}")
    print(f"  Horizon:       {args.horizon} samples ({args.horizon/10:.1f} ms)")
    print(f"  Device:        {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print_turbulence_summary()

    # --- 1. Generate dataset ---
    print("\n" + "=" * 60)
    print("  STEP 1: Generating Dataset")
    print("=" * 60)

    focal = args.focal_loss
    
    if args.quick:
        ds_config = DatasetConfig(
            duration_s=30.0, window_size=50, si_window_size=200,
            transition_duration_s=15.0, window_stride=20,
            prediction_horizon=args.horizon,
            include_si_context=not args.no_context,
        )
        n_real = 1
        train_config = TrainingConfig(
            max_epochs=10, batch_size=128, patience=5,
            use_focal_loss=focal,
        )
    else:
        ds_config = DatasetConfig(
            duration_s=300.0, window_size=100, si_window_size=500,
            transition_duration_s=60.0, window_stride=50,
            prediction_horizon=args.horizon,
            include_si_context=not args.no_context,
        )
        n_real = 3
        train_config = TrainingConfig(use_focal_loss=focal)

    gen = DatasetGenerator(dataset_config=ds_config, seed=args.seed)
    t0 = time.time()
    train_ds, val_ds, test_ds, ds_stats = gen.build_full_dataset(
        mean_rx_power_dBm=args.mean_rx_power,
        n_realisations_per_regime=n_real,
    )
    print(f"Dataset generation took {time.time()-t0:.1f}s")

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(ds_stats, f, indent=2)

    # --- 2. Reactive AMC baseline (the thing we need to beat) ---
    print("\n" + "=" * 60)
    print("  STEP 2: Reactive AMC Baseline")
    print("=" * 60)

    baseline_results = run_baseline_sweep(
        test_ds,
        # Include the prediction horizon as a delay point — this is the
        # direct comparison: "reactive with same delay as NN prediction horizon"
        delays=[0, 1, 5, 10, 20, 50, 90, args.horizon],
    )
    # The delay=10 result (1 ms) is the most realistic comparison
    # point for strong turbulence where tau_0 ~ 1 ms
    all_results = []
    for br in baseline_results:
        all_results.append(br)

    # Save baseline results
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)

    # --- 3. Train and evaluate each NN model ---
    print("\n" + "=" * 60)
    print("  STEP 3: Training Neural Network Models")
    print("=" * 60)

    # NOTE: all_results already contains the baseline sweep from step 2
    for model_name in model_names:
        print(f"\n{'_'*60}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'_'*60}")

        model = create_model(
            name=model_name,
            seq_length=train_ds.seq_length,
            n_features=train_ds.n_features,
            n_classes=NUM_MCS_CLASSES,
        )
        print(f"  {model.model_summary()}")

        trainer = Trainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds,
            config=train_config,
            output_dir=str(output_dir / model_name),
        )
        history = trainer.train()

        evaluator = Evaluator(model, test_ds)
        results = evaluator.evaluate()
        results["training_time_s"] = history.get("training_time_s", 0)
        results["epochs_trained"] = history.get("epochs_trained", 0)

        Evaluator.print_results(results)
        Evaluator.save_results(results, str(output_dir / f"{model_name}_results.json"))
        torch.save(model.state_dict(), output_dir / f"{model_name}_best.pt")
        all_results.append(results)

        # Also evaluate the hybrid NN+reactive selector for this model
        # This combines the NN for volatile conditions with reactive for stable
        if test_ds.raw_snr_windows is not None:
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                all_preds = []
                from torch.utils.data import DataLoader
                loader = DataLoader(test_ds, batch_size=512, shuffle=False)
                for feats, _ in loader:
                    logits = model(feats.to(device))
                    all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                nn_preds = np.concatenate(all_preds)

            hybrid = HybridMCSSelector(si_threshold=0.05)
            hybrid_results = hybrid.evaluate_on_dataset(nn_preds, test_ds)
            hybrid_results["model_name"] = f"Hybrid_{model_name}"
            hybrid_results["training_time_s"] = results["training_time_s"]
            stable_pct = hybrid_results["stable_fraction"] * 100
            print(f"\n  Hybrid ({model_name}): {hybrid_results['throughput_ratio']:.1%} throughput "
                  f"({stable_pct:.0f}% reactive / {100-stable_pct:.0f}% NN)")
            all_results.append(hybrid_results)

    # --- 4. Compare ---
    print("\n" + "=" * 60)
    print("  STEP 4: Model Comparison")
    print("=" * 60)

    comparison = Evaluator.compare_models(all_results)
    print(comparison)

    with open(output_dir / "comparison.txt", "w") as f:
        f.write(comparison)
    with open(output_dir / "all_results.json", "w") as f:
        summary = [{k: v for k, v in r.items() if k != "confusion_matrix"}
                   for r in all_results]
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
