#!/usr/bin/env python3
"""
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

if __name__ == "__main__":
    main()
