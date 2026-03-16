#!/usr/bin/env python3
"""
Verify the FSO channel model produces physically reasonable results.
Does NOT require PyTorch.

Usage (from repo root):
    python verify.py
"""

from fso_mcs_predictor.verify_channel import main

if __name__ == "__main__":
    main()
