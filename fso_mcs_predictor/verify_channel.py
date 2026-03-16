#!/usr/bin/env python3
"""
verify_channel.py — Channel Model Verification
================================================
Run this first to verify the FSO channel simulation produces
physically reasonable results. Does NOT require PyTorch.

Generates diagnostic plots showing:
  1. Turbulence parameter table for all regimes
  2. Channel time-series for each regime  
  3. Power distribution (should match Gamma-Gamma)
  4. SNR to MCS mapping
  5. Scintillation index estimation accuracy

Usage:
  python verify_channel.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fso_mcs_predictor.config import (
    TURBULENCE_REGIMES, LinkBudget, MCS_TABLE,
)
from fso_mcs_predictor.channel.turbulence import TurbulenceParameters
from fso_mcs_predictor.channel.fso_channel import FSOChannel
from fso_mcs_predictor.system.snr_model import SNRModel, MCSSelector

# Try to import matplotlib (optional, for plots)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not found - skipping plots (text output only)")


def main():
    link = LinkBudget()
    snr_model = SNRModel()
    mcs_selector = MCSSelector()
    
    print("=" * 70)
    print("  FSO Channel Model Verification")
    print("=" * 70)
    
    # --- 1. Turbulence parameters ---
    print("\n1. TURBULENCE REGIME PARAMETERS")
    print("-" * 70)
    print(f"  Link: {link.link_distance_m:.0f} m, "
          f"λ={link.wavelength_m*1e9:.0f} nm, "
          f"Tx={link.tx_power_mW:.0f} mW")
    print(f"  Lens: {link.lens_diameter_m*1e3:.0f} mm diameter")
    print()
    
    regimes_data = {}
    for name, regime in TURBULENCE_REGIMES.items():
        tp = TurbulenceParameters(
            Cn2=regime.Cn2,
            wavelength_m=link.wavelength_m,
            link_distance_m=link.link_distance_m,
        )
        print(tp.summary(regime.wind_speed_m_s))
        
        # Generate channel
        ch = FSOChannel(tp, regime.wind_speed_m_s, 10000, seed=42)
        time_s, power_mW, meta = ch.generate(
            duration_s=30.0, mean_rx_power_dBm=-10.0
        )
        snr_dB = snr_model.power_to_snr(power_mW)
        mcs_class = mcs_selector.select(snr_dB)
        
        regimes_data[name] = {
            "time": time_s, "power": power_mW, "snr": snr_dB,
            "mcs": mcs_class, "meta": meta, "turb": tp,
        }
        print()
    
    # --- 2. MCS Table ---
    print("\n2. MCS TABLE (from Dindar et al., Table I)")
    print("-" * 50)
    print(f"  {'MCS':>4} | {'Mod':>6} | {'Rate':>5} | {'SNR(dB)':>7} | {'bits/sym':>8}")
    print(f"  {'----':>4}-+-{'------':>6}-+-{'-----':>5}-+-{'-------':>7}-+-{'--------':>8}")
    for m in MCS_TABLE:
        print(f"  {m.index:>4} | {m.modulation:>6} | {m.code_rate:>5.2f} | "
              f"{m.required_snr_dB:>7.1f} | {m.bits_per_symbol:>8}")
    
    # --- 3. Statistics summary ---
    print("\n3. CHANNEL STATISTICS PER REGIME")
    print("-" * 70)
    print(f"  {'Regime':>12} | {'Mean Pwr':>9} | {'Std Pwr':>9} | "
          f"{'SI meas':>7} | {'SI theory':>9} | {'Outage%':>7} | {'MCS modes':>10}")
    print(f"  {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*10}")
    
    for name, data in regimes_data.items():
        power = data["power"]
        mcs = data["mcs"]
        si_meas = data["meta"]["scintillation_index_actual"]
        si_theory = data["meta"]["scintillation_index_theoretical"]
        outage_pct = np.mean(mcs == 0) * 100
        
        unique, counts = np.unique(mcs, return_counts=True)
        top_mcs = unique[np.argsort(-counts)][:3]
        mcs_str = ",".join(str(m) for m in top_mcs)
        
        print(f"  {name:>12} | {np.mean(power):>8.4f}mW | {np.std(power):>8.4f}mW | "
              f"{si_meas:>7.3f} | {si_theory:>9.4f} | {outage_pct:>6.1f}% | "
              f"{mcs_str:>10}")
    
    # --- 4. SNR model check ---
    print("\n4. SNR MODEL CALIBRATION CHECK")
    print("-" * 50)
    test_powers_dBm = [-27, -20, -15, -10, -5, -3]
    for p in test_powers_dBm:
        p_mW = 10**(p/10)
        snr = snr_model.power_to_snr(np.array([p_mW]))[0]
        mcs = mcs_selector.select(np.array([snr]))[0]
        mcs_name = MCSSelector.class_to_mcs_name(mcs)
        print(f"  Pr = {p:>4d} dBm  ->  SNR = {snr:>5.1f} dB  ->  {mcs_name}")
    
    # --- 5. Plots (if matplotlib available) ---
    if HAS_MATPLOTLIB:
        _generate_plots(regimes_data, snr_model, mcs_selector)
    
    print("\n" + "=" * 70)
    print("  Verification complete!")
    if HAS_MATPLOTLIB:
        print("  Plots saved to channel_verification.png")
    print("=" * 70)


def _generate_plots(regimes_data, snr_model, mcs_selector):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FSO Channel Model Verification", fontsize=14, fontweight='bold')
    
    colors = {'weak': '#2196F3', 'moderate': '#4CAF50', 
              'strong': '#FF9800', 'very_strong': '#F44336'}
    
    # Plot 1: Power time-series (first 5 seconds)
    ax = axes[0, 0]
    for name, data in regimes_data.items():
        mask = data["time"] < 5.0
        ax.plot(data["time"][mask] * 1000, 
                10*np.log10(np.maximum(data["power"][mask], 1e-10)),
                color=colors.get(name, 'gray'), alpha=0.7, 
                linewidth=0.5, label=name)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Received Power (dBm)")
    ax.set_title("Channel Time-Series (first 5s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Power PDF
    ax = axes[0, 1]
    for name, data in regimes_data.items():
        power_dBm = 10*np.log10(np.maximum(data["power"], 1e-10))
        ax.hist(power_dBm, bins=100, density=True, alpha=0.5,
                color=colors.get(name, 'gray'), label=name)
    ax.set_xlabel("Received Power (dBm)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Power Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: SNR vs MCS mapping
    ax = axes[1, 0]
    snr_range = np.linspace(0, 35, 1000)
    mcs_vals = mcs_selector.select(snr_range)
    ax.plot(snr_range, mcs_vals, 'k-', linewidth=1.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MCS Class")
    ax.set_title("SNR → MCS Mapping (Table I)")
    ax.grid(True, alpha=0.3)
    ax.set_yticks(range(0, 15, 2))
    
    # Plot 4: MCS distribution per regime
    ax = axes[1, 1]
    regime_names = list(regimes_data.keys())
    n_regimes = len(regime_names)
    width = 0.8 / n_regimes
    
    for i, name in enumerate(regime_names):
        mcs = regimes_data[name]["mcs"]
        unique, counts = np.unique(mcs, return_counts=True)
        pct = counts / len(mcs) * 100
        offset = (i - n_regimes/2 + 0.5) * width
        ax.bar(unique + offset, pct, width=width, 
               color=colors.get(name, 'gray'), label=name, alpha=0.8)
    
    ax.set_xlabel("MCS Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("MCS Distribution per Regime")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("channel_verification.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
