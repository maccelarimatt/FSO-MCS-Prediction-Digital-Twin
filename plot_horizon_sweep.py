"""
This script has been superseded.

The horizon sweep plot is now generated from live results.json files
with no hardcoded values. Use:

    python plots/plot_horizon_sweep.py results_h1 results_h10 results_h50

Run run.py with the appropriate --output-dir for each horizon first:

    python run.py --horizon 1  --output-dir results_h1
    python run.py --horizon 10 --output-dir results_h10
    python run.py --horizon 50 --output-dir results_h50
"""
import sys
sys.exit(
    "Moved to plots/plot_horizon_sweep.py\n"
    "Usage: python plots/plot_horizon_sweep.py results_h1 results_h10 results_h50"
)
