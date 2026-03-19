"""
Dataset Generator
=================
Generates training/validation/test datasets from FSO channel simulations.

Each sample consists of:
  Input features (per timestep in the lookback window):
    - Received optical power (log-normalised)
    - Electrical SNR (normalised)
    - Estimated scintillation index (log, normalised) — the "context"
    - Rate of change of power (first difference)
    
  Target:
    - Ideal MCS class for the next transmission opportunity.

The generator supports multiple turbulence regimes including smooth
transitions, producing datasets that test generalisation across
changing atmospheric conditions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from ..config import (
    DatasetConfig, LinkBudget, TURBULENCE_REGIMES,
    TurbulenceRegime,
)
from ..channel.turbulence import TurbulenceParameters
from ..channel.fso_channel import FSOChannel
from ..system.snr_model import SNRModel, MCSSelector


class FSODataset(Dataset):
    """
    PyTorch Dataset for MCS prediction from FSO channel observations.
    
    Each item returns:
        features: Tensor of shape (window_size, n_features)
        target:   Scalar LongTensor — ideal MCS class at prediction horizon
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        si_values: np.ndarray,
        regime_labels: np.ndarray,
        raw_snr_windows: np.ndarray = None,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.si_values = torch.FloatTensor(si_values)
        self.regime_labels = regime_labels  # keep as numpy string array
        # Raw (un-normalised) SNR at each window position — used by
        # reactive AMC baseline to apply MCS thresholds at delayed positions
        self.raw_snr_windows = raw_snr_windows  # (n_samples, window_size) or None
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    @property
    def n_features(self) -> int:
        return self.features.shape[-1]
    
    @property
    def seq_length(self) -> int:
        return self.features.shape[1]


class DatasetGenerator:
    """
    Orchestrates channel simulation and dataset creation.
    
    Workflow:
      1. For each turbulence regime, generate channel time-series.
      2. Compute SNR and ideal MCS at each time step.
      3. Extract sliding windows with features and targets.
      4. Concatenate and split into train/val/test.
    """
    
    def __init__(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        link_budget: Optional[LinkBudget] = None,
        snr_model: Optional[SNRModel] = None,
        mcs_selector: Optional[MCSSelector] = None,
        seed: int = 42,
    ):
        self.cfg = dataset_config or DatasetConfig()
        self.link = link_budget or LinkBudget()
        self.snr_model = snr_model or SNRModel()
        self.mcs_selector = mcs_selector or MCSSelector(hysteresis_dB=0.5)
        self.rng = np.random.default_rng(seed)
    
    def generate_regime_data(
        self,
        regime: TurbulenceRegime,
        duration_s: Optional[float] = None,
        mean_rx_power_dBm: float = -10.0,
    ) -> Dict:
        """Generate raw time-series data for a single turbulence regime."""
        duration = duration_s or self.cfg.duration_s
        
        turb_params = TurbulenceParameters(
            Cn2=regime.Cn2,
            wavelength_m=self.link.wavelength_m,
            link_distance_m=self.link.link_distance_m,
        )
        
        # Vary wind speed slightly for each realisation
        wind = max(0.5, self.rng.normal(
            regime.wind_speed_m_s, regime.wind_speed_std
        ))
        
        channel = FSOChannel(
            turb_params, wind, self.cfg.sample_rate_Hz,
            seed=self.rng.integers(0, 2**31),
        )
        time_s, power_mW, meta = channel.generate(
            duration, mean_rx_power_dBm
        )
        
        snr_dB = self.snr_model.power_to_snr(power_mW)
        mcs_class = self.mcs_selector.select(snr_dB)
        si = self._compute_running_si(power_mW)
        
        return {
            "time": time_s,
            "power_mW": power_mW,
            "snr_dB": snr_dB,
            "mcs_class": mcs_class,
            "scintillation_index": si,
            "regime_name": regime.name,
            "metadata": meta,
        }
    
    def generate_transition_data(
        self,
        regime_start: TurbulenceRegime,
        regime_end: TurbulenceRegime,
        duration_s: Optional[float] = None,
        mean_rx_power_dBm: float = -10.0,
    ) -> Dict:
        """Generate data for a transitioning turbulence scenario."""
        duration = duration_s or self.cfg.transition_duration_s
        
        turb_start = TurbulenceParameters(
            Cn2=regime_start.Cn2,
            wavelength_m=self.link.wavelength_m,
            link_distance_m=self.link.link_distance_m,
        )
        turb_end = TurbulenceParameters(
            Cn2=regime_end.Cn2,
            wavelength_m=self.link.wavelength_m,
            link_distance_m=self.link.link_distance_m,
        )
        
        channel = FSOChannel(
            turb_start, regime_start.wind_speed_m_s,
            self.cfg.sample_rate_Hz,
            seed=self.rng.integers(0, 2**31),
        )
        time_s, power_mW, meta = channel.generate_transitioning(
            turb_end, regime_end.wind_speed_m_s,
            duration, mean_rx_power_dBm,
            transition_type="sigmoid",
        )
        
        snr_dB = self.snr_model.power_to_snr(power_mW)
        mcs_class = self.mcs_selector.select(snr_dB)
        si = self._compute_running_si(power_mW)
        
        return {
            "time": time_s,
            "power_mW": power_mW,
            "snr_dB": snr_dB,
            "mcs_class": mcs_class,
            "scintillation_index": si,
            "regime_name": f"transition_{regime_start.name}_to_{regime_end.name}",
            "metadata": meta,
        }
    
    def _compute_running_si(self, power_mW: np.ndarray) -> np.ndarray:
        """
        Compute running (sliding-window) scintillation index.
        
        SI = Var(I) / <I>²
        
        Uses a causal (backward-looking) window so the SI at time t
        only uses past observations — no future leakage.
        """
        n = len(power_mW)
        si = np.zeros(n)
        win = self.cfg.si_window_size
        
        # O(n) via cumulative sums
        cumsum = np.cumsum(power_mW)
        cumsum2 = np.cumsum(power_mW ** 2)
        
        for i in range(n):
            start = max(0, i - win + 1)
            count = i - start + 1
            if count < 10:
                si[i] = 0.0
                continue
            
            if start == 0:
                s1 = cumsum[i]
                s2 = cumsum2[i]
            else:
                s1 = cumsum[i] - cumsum[start - 1]
                s2 = cumsum2[i] - cumsum2[start - 1]
            
            mean_val = s1 / count
            mean_sq = s2 / count
            variance = mean_sq - mean_val ** 2
            
            if mean_val > 1e-10:
                si[i] = max(0.0, variance / (mean_val ** 2))
            else:
                si[i] = 0.0
        
        return si
    
    def _extract_windows(
        self, data: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract sliding-window samples from a regime's time-series.
        
        Returns:
            features:         (n_windows, window_size, n_features)
            targets:          (n_windows,)
            si_vals:          (n_windows,)
            raw_snr_windows:  (n_windows, window_size) — un-normalised SNR
        """
        win = self.cfg.window_size
        horizon = self.cfg.prediction_horizon
        n = len(data["power_mW"])
        
        # --- Feature engineering ---
        power_dBm = 10 * np.log10(np.maximum(data["power_mW"], 1e-10))
        snr = data["snr_dB"]
        si = data["scintillation_index"]
        
        # Normalise to ~N(0,1)
        power_mean = np.mean(power_dBm)
        power_std = max(np.std(power_dBm), 1e-6)
        snr_mean = np.mean(snr)
        snr_std = max(np.std(snr), 1e-6)
        si_log = np.log10(np.maximum(si, 1e-6))
        si_mean = np.mean(si_log)
        si_std = max(np.std(si_log), 1e-6)
        
        power_norm = (power_dBm - power_mean) / power_std
        snr_norm = (snr - snr_mean) / snr_std
        si_norm = (si_log - si_mean) / si_std
        
        power_diff = np.diff(power_dBm, prepend=power_dBm[0])
        diff_std = max(np.std(power_diff), 1e-6)
        diff_norm = power_diff / diff_std
        
        # Assemble features
        feature_list = [power_norm, snr_norm]
        if self.cfg.include_si_context:
            feature_list.append(si_norm)
        if self.cfg.include_power_derivative:
            feature_list.append(diff_norm)
        
        features_full = np.stack(feature_list, axis=-1)  # (n, n_feat)
        
        # --- Sliding windows with stride ---
        stride = self.cfg.window_stride
        n_possible = n - win - horizon + 1
        if n_possible <= 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Window start indices, spaced by stride
        start_indices = np.arange(0, n_possible, stride)
        n_windows = len(start_indices)
        
        features = np.zeros((n_windows, win, features_full.shape[1]), 
                           dtype=np.float32)
        targets = np.zeros(n_windows, dtype=np.int64)
        si_vals = np.zeros(n_windows, dtype=np.float32)
        raw_snr_windows = np.zeros((n_windows, win), dtype=np.float32)
        
        for i, idx in enumerate(start_indices):
            features[i] = features_full[idx:idx + win]
            targets[i] = data["mcs_class"][idx + win + horizon - 1]
            si_vals[i] = si[idx + win - 1]
            raw_snr_windows[i] = snr[idx:idx + win]
        
        return features, targets, si_vals, raw_snr_windows
    
    def build_full_dataset(
        self,
        regimes: Optional[List[str]] = None,
        include_transitions: bool = True,
        mean_rx_power_dBm: float = -10.0,
        power_levels_dBm: Optional[List[float]] = None,
        n_realisations_per_regime: int = 3,
    ) -> Tuple[FSODataset, FSODataset, FSODataset, Dict]:
        """
        Build complete train/val/test datasets across multiple regimes.
        
        Args:
            regimes: Regime names (default: all four).
            include_transitions: Include transition scenarios.
            mean_rx_power_dBm: Mean Rx power (used if power_levels_dBm is None).
            power_levels_dBm: List of power levels to generate data for.
                             If provided, generates one realisation per regime
                             per power level. This ensures all MCS classes are
                             well-represented: high power populates high MCS,
                             low power populates low MCS and outage.
            n_realisations_per_regime: Independent channel runs per regime per power level.
        
        Returns:
            train_dataset, val_dataset, test_dataset, stats_dict
        """
        if regimes is None:
            regimes = list(TURBULENCE_REGIMES.keys())
        
        # If a single power is given, wrap it in a list
        if power_levels_dBm is None:
            power_levels_dBm = [mean_rx_power_dBm]
        
        all_features, all_targets, all_si, all_labels, all_raw_snr = [], [], [], [], []
        
        print(f"Generating datasets for regimes: {regimes}")
        print(f"  Power levels: {power_levels_dBm} dBm")
        print(f"  {n_realisations_per_regime} realisations × "
              f"{self.cfg.duration_s}s each per power level")
        
        for power_dBm in power_levels_dBm:
            print(f"\n  --- Power level: {power_dBm} dBm ---")
            for regime_name in regimes:
                regime = TURBULENCE_REGIMES[regime_name]
                for r in range(n_realisations_per_regime):
                    print(f"  {regime_name} [{r+1}/{n_realisations_per_regime}] "
                          f"@ {power_dBm} dBm...", end=" ", flush=True)
                    data = self.generate_regime_data(
                        regime, mean_rx_power_dBm=power_dBm
                    )
                    features, targets, si_vals, raw_snr = self._extract_windows(data)
                    if len(features) == 0:
                        print("skipped (too short)")
                        continue
                    all_features.append(features)
                    all_targets.append(targets)
                    all_si.append(si_vals)
                    all_raw_snr.append(raw_snr)
                    all_labels.append(
                        np.full(len(targets), regime_name, dtype=object)
                    )
                    print(f"{len(targets)} windows")
        
        if include_transitions and len(regimes) >= 2:
            # Generate transitions at the middle power level
            mid_power = power_levels_dBm[len(power_levels_dBm) // 2]
            for i in range(len(regimes) - 1):
                s, e = regimes[i], regimes[i + 1]
                print(f"  transition {s} → {e} @ {mid_power} dBm...", 
                      end=" ", flush=True)
                data = self.generate_transition_data(
                    TURBULENCE_REGIMES[s], TURBULENCE_REGIMES[e],
                    mean_rx_power_dBm=mid_power,
                )
                features, targets, si_vals, raw_snr = self._extract_windows(data)
                if len(features) == 0:
                    print("skipped")
                    continue
                all_features.append(features)
                all_targets.append(targets)
                all_si.append(si_vals)
                all_raw_snr.append(raw_snr)
                all_labels.append(
                    np.full(len(targets), f"trans_{s}_{e}", dtype=object)
                )
                print(f"{len(targets)} windows")
        
        # Concatenate and shuffle
        features = np.concatenate(all_features, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        si_vals = np.concatenate(all_si, axis=0)
        raw_snr = np.concatenate(all_raw_snr, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        perm = self.rng.permutation(len(targets))
        features, targets, si_vals, raw_snr, labels = (
            features[perm], targets[perm], si_vals[perm], 
            raw_snr[perm], labels[perm]
        )
        
        # Split
        n = len(targets)
        n_train = int(n * self.cfg.train_fraction)
        n_val = int(n * self.cfg.val_fraction)
        
        def _slice(arr, a, b):
            return arr[a:b]
        
        train_ds = FSODataset(
            _slice(features, 0, n_train), _slice(targets, 0, n_train),
            _slice(si_vals, 0, n_train), _slice(labels, 0, n_train),
            _slice(raw_snr, 0, n_train),
        )
        val_ds = FSODataset(
            _slice(features, n_train, n_train+n_val),
            _slice(targets, n_train, n_train+n_val),
            _slice(si_vals, n_train, n_train+n_val),
            _slice(labels, n_train, n_train+n_val),
            _slice(raw_snr, n_train, n_train+n_val),
        )
        test_ds = FSODataset(
            _slice(features, n_train+n_val, n),
            _slice(targets, n_train+n_val, n),
            _slice(si_vals, n_train+n_val, n),
            _slice(labels, n_train+n_val, n),
            _slice(raw_snr, n_train+n_val, n),
        )
        
        unique, counts = np.unique(targets, return_counts=True)
        stats = {
            "n_total": n,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n - n_train - n_val,
            "n_features": features.shape[-1],
            "seq_length": features.shape[1],
            "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
        }
        
        print(f"\nDataset: {n} total ({n_train} train / {n_val} val / "
              f"{n - n_train - n_val} test)")
        print(f"Shape: ({features.shape[1]}, {features.shape[2]}) per sample")
        
        return train_ds, val_ds, test_ds, stats
