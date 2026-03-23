"""
Dataset generation: windowing, feature engineering, realisation-level split.

Features per timestep (5 total):
  0. Raw power (dBm) — absolute level for MCS threshold learning
  1. Raw SNR (dB) — same reason
  2. Normalised power — zero-mean, unit-variance within window (temporal pattern)
  3. Log scintillation index — backward-looking, regime context
  4. Power derivative — first difference in dB (rate of change)
"""

import numpy as np
from . import config
from .channel import generate_channel_realisation, snr_to_mcs


def compute_scintillation_index(power_linear, si_window=config.SI_WINDOW):
    """Causal SI = Var(I)/<I>^2 over sliding window. Returns log(1+SI)."""
    n = len(power_linear)
    si = np.zeros(n)
    for i in range(n):
        start = max(0, i - si_window + 1)
        w = power_linear[start:i+1]
        if len(w) < 10:
            continue
        mean_I = w.mean()
        if mean_I > 1e-20:
            si[i] = w.var() / mean_I**2
    return np.log1p(si)


def extract_windows(power_dbm, snr_db, window_size=config.WINDOW_SIZE,
                     stride=config.WINDOW_STRIDE, horizon=1):
    """
    Extract overlapping windows and compute features.
    Label = ideal MCS at (window_end - 1 + horizon).
    
    Returns: features (n, win, 5), labels (n,), raw_snr_windows (n, win)
    """
    n = len(power_dbm)
    power_linear = 10**(power_dbm / 10)
    log_si = compute_scintillation_index(power_linear)
    power_deriv = np.zeros_like(power_dbm)
    power_deriv[1:] = np.diff(power_dbm)
    ideal_mcs = snr_to_mcs(snr_db, hysteresis=config.HYSTERESIS_DB)
    
    max_start = n - window_size - horizon
    if max_start <= 0:
        return np.array([]), np.array([]), np.array([])
    
    starts = np.arange(0, max_start, stride)
    nw = len(starts)
    features = np.zeros((nw, window_size, config.N_FEATURES), dtype=np.float32)
    labels = np.zeros(nw, dtype=np.int64)
    raw_snr = np.zeros((nw, window_size), dtype=np.float32)
    
    for idx, s in enumerate(starts):
        e = s + window_size
        wp = power_dbm[s:e]
        ws = snr_db[s:e]
        
        # Normalised power within window
        wm, wsd = wp.mean(), wp.std()
        norm_p = (wp - wm) / max(wsd, 1e-8)
        
        features[idx, :, 0] = wp
        features[idx, :, 1] = ws
        features[idx, :, 2] = norm_p
        features[idx, :, 3] = log_si[s:e]
        features[idx, :, 4] = power_deriv[s:e]
        
        target = min(e - 1 + horizon, n - 1)
        labels[idx] = ideal_mcs[target]
        raw_snr[idx] = ws
    
    return features, labels, raw_snr


def generate_dataset(regimes=None, n_realisations=3, duration_s=300,
                     horizon=1, seed=42, verbose=True):
    """
    Generate full dataset across turbulence regimes with realisation-level split.
    
    Returns: dict with keys 'train', 'val', 'test' each containing
             {'features': array, 'labels': array, 'raw_snr': array}
             + 'class_weights' and 'metadata'
    """
    if regimes is None:
        regimes = list(config.TURBULENCE_REGIMES.keys())
    
    rng = np.random.default_rng(seed)
    splits = {s: {"features": [], "labels": [], "raw_snr": []}
              for s in ["train", "val", "test"]}
    metadata = {"regimes": {}, "horizon": horizon}
    
    for regime in regimes:
        if verbose:
            print(f"  Generating {regime} ({n_realisations} × {duration_s}s)...")
        metadata["regimes"][regime] = []
        
        for r in range(n_realisations):
            rseed = rng.integers(0, 2**31)
            ch = generate_channel_realisation(regime, duration_s, seed=rseed)
            feats, labels, raw_snr = extract_windows(
                ch["power_dbm"], ch["snr_db"], horizon=horizon)
            
            if len(feats) == 0:
                continue
            
            metadata["regimes"][regime].append({
                "realisation": r, "n_windows": len(labels),
                "v_wind": ch["v_wind"],
            })
            
            if n_realisations >= 3:
                if r < n_realisations - 1:
                    splits["train"]["features"].append(feats)
                    splits["train"]["labels"].append(labels)
                    splits["train"]["raw_snr"].append(raw_snr)
                else:
                    mid = len(labels) // 2
                    splits["val"]["features"].append(feats[:mid])
                    splits["val"]["labels"].append(labels[:mid])
                    splits["val"]["raw_snr"].append(raw_snr[:mid])
                    splits["test"]["features"].append(feats[mid:])
                    splits["test"]["labels"].append(labels[mid:])
                    splits["test"]["raw_snr"].append(raw_snr[mid:])
            else:
                n = len(labels)
                gap = config.WINDOW_SIZE // config.WINDOW_STRIDE + 2
                t_end = int(0.6 * n)
                v_start, v_end = t_end + gap, t_end + gap + int(0.2 * n)
                te_start = v_end + gap
                
                splits["train"]["features"].append(feats[:t_end])
                splits["train"]["labels"].append(labels[:t_end])
                splits["train"]["raw_snr"].append(raw_snr[:t_end])
                if v_start < n:
                    splits["val"]["features"].append(feats[v_start:min(v_end, n)])
                    splits["val"]["labels"].append(labels[v_start:min(v_end, n)])
                    splits["val"]["raw_snr"].append(raw_snr[v_start:min(v_end, n)])
                if te_start < n:
                    splits["test"]["features"].append(feats[te_start:])
                    splits["test"]["labels"].append(labels[te_start:])
                    splits["test"]["raw_snr"].append(raw_snr[te_start:])
    
    # Concatenate and shuffle
    for s in splits:
        for k in splits[s]:
            if splits[s][k]:
                splits[s][k] = np.concatenate(splits[s][k], axis=0)
            else:
                splits[s][k] = np.array([])
    
    # Shuffle train and val (keep test in order for plotting)
    for s in ["train", "val"]:
        if len(splits[s]["labels"]) > 0:
            idx = rng.permutation(len(splits[s]["labels"]))
            for k in splits[s]:
                splits[s][k] = splits[s][k][idx]
    
    # Class weights (inverse frequency)
    if len(splits["train"]["labels"]) > 0:
        cc = np.bincount(splits["train"]["labels"].astype(int),
                         minlength=config.NUM_CLASSES)
        total = cc.sum()
        cw = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        for c in range(config.NUM_CLASSES):
            cw[c] = total / (config.NUM_CLASSES * cc[c]) if cc[c] > 0 else 0.0
    else:
        cc = np.zeros(config.NUM_CLASSES)
        cw = np.zeros(config.NUM_CLASSES)
    
    if verbose:
        print(f"\n  Sizes: train={len(splits['train']['labels'])}, "
              f"val={len(splits['val']['labels'])}, test={len(splits['test']['labels'])}")
        print(f"  Class distribution (train): {dict(enumerate(cc.tolist()))}")
    
    metadata["class_counts"] = cc.tolist()
    
    return splits, cw, metadata
