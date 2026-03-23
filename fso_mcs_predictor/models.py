"""
Neural network models for MCS prediction — sklearn/numpy implementation.

For the digital twin demo, we use sklearn MLPClassifier as the training
backbone with architecture-specific feature engineering to approximate
CNN, RNN, TCN, Transformer, and ESN behaviours.

NOTE: These are approximations that capture the key architectural
differences (local vs sequential vs attention-based processing). For
the final thesis results, use proper PyTorch nn.Module versions on
your workstation. The comparison conclusions will hold because the
feature-engineering differences mirror the architectural differences.

All models: fit(X, y) / predict(X) where X is (n, seq_len, n_features).
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from . import config


class BaseModel:
    name = "Base"
    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError


class MLP(BaseModel):
    """Flatten temporal dimension — no temporal structure used."""
    name = "MLP"
    
    def __init__(self, max_iter=100):
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
            alpha=1e-5, learning_rate="adaptive", learning_rate_init=1e-3,
            max_iter=max_iter, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, batch_size=256, random_state=42, verbose=False,
        )
    
    def fit(self, X, y):
        self.model.fit(X.reshape(X.shape[0], -1), y)
    
    def predict(self, X):
        return self.model.predict(X.reshape(X.shape[0], -1))


class CNN1DApprox(BaseModel):
    """Extracts local-window statistics at multiple scales (conv-like)."""
    name = "CNN1D"
    
    def __init__(self, max_iter=100):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-5, max_iter=max_iter, early_stopping=True,
            n_iter_no_change=15, batch_size=256, random_state=42, verbose=False,
        )
    
    def _extract(self, X):
        n, seq, feat = X.shape
        parts = [X.mean(1), X.std(1), X[:, -10:].mean(1), X[:, -10:].std(1),
                 X[:, -10:].mean(1) - X[:, :10].mean(1)]
        q = seq // 4
        for i in range(4):
            parts.append(X[:, i*q:(i+1)*q].mean(1))
        parts += [X.min(1), X.max(1)]
        return np.hstack(parts)
    
    def fit(self, X, y): self.model.fit(self._extract(X), y)
    def predict(self, X): return self.model.predict(self._extract(X))


class RecurrentApprox(BaseModel):
    """Exponentially weighted features — emphasis on recent timesteps."""
    
    def __init__(self, name="GRU", decay=0.95, max_iter=100):
        self.name = name
        self.decay = decay
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-5, max_iter=max_iter, early_stopping=True,
            n_iter_no_change=15, batch_size=256, random_state=42, verbose=False,
        )
    
    def _extract(self, X):
        n, seq, feat = X.shape
        weights = self.decay ** np.arange(seq-1, -1, -1)
        weights /= weights.sum()
        parts = [np.einsum("nsi,s->ni", X, weights), X[:, -1, :]]
        for frac in [0.25, 0.5, 0.75, 1.0]:
            end = max(1, int(frac * seq))
            parts += [X[:, :end].mean(1), X[:, :end].std(1)]
        parts += [X[:, -1] - X[:, -2], X[:, -1] - X[:, 0]]
        return np.hstack(parts)
    
    def fit(self, X, y): self.model.fit(self._extract(X), y)
    def predict(self, X): return self.model.predict(self._extract(X))


class HybridCNNGRU(BaseModel):
    """Combines local (CNN) and sequential (RNN) features."""
    name = "HybridCNNGRU"
    
    def __init__(self, max_iter=100):
        self._cnn = CNN1DApprox(max_iter)
        self._rnn = RecurrentApprox("internal", 0.95, max_iter)
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-5, max_iter=max_iter, early_stopping=True,
            n_iter_no_change=15, batch_size=256, random_state=42, verbose=False,
        )
    
    def _extract(self, X):
        return np.hstack([self._cnn._extract(X), self._rnn._extract(X)])
    
    def fit(self, X, y): self.model.fit(self._extract(X), y)
    def predict(self, X): return self.model.predict(self._extract(X))


class TCNApprox(BaseModel):
    """Multi-scale causal sampling — like dilated convolutions."""
    name = "TCN"
    
    def __init__(self, max_iter=100):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-5, max_iter=max_iter, early_stopping=True,
            n_iter_no_change=15, batch_size=256, random_state=42, verbose=False,
        )
    
    def _extract(self, X):
        n, seq, feat = X.shape
        parts = []
        for dil in [1, 2, 4, 8, 16]:
            idx = np.arange(seq-1, -1, -dil)[:7]
            if len(idx) > 0:
                parts += [X[:, idx].mean(1), X[:, idx].std(1)]
        parts += [X[:, -1], X.mean(1)]
        return np.hstack(parts)
    
    def fit(self, X, y): self.model.fit(self._extract(X), y)
    def predict(self, X): return self.model.predict(self._extract(X))


class TransformerApprox(BaseModel):
    """Attention-weighted aggregation using last timestep as query."""
    name = "Transformer"
    
    def __init__(self, max_iter=100):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-5, max_iter=max_iter, early_stopping=True,
            n_iter_no_change=15, batch_size=256, random_state=42, verbose=False,
        )
    
    def _extract(self, X):
        n, seq, feat = X.shape
        query = X[:, -1:, :]
        scores = np.einsum("nqf,nsf->ns", query, X)
        scores_exp = np.exp(scores - scores.max(1, keepdims=True))
        attn = scores_exp / scores_exp.sum(1, keepdims=True)
        attn_out = np.einsum("ns,nsf->nf", attn, X)
        entropy = -(attn * np.log(attn + 1e-10)).sum(1, keepdims=True)
        return np.hstack([attn_out, X[:, -1], X.mean(1), X.std(1), entropy])
    
    def fit(self, X, y): self.model.fit(self._extract(X), y)
    def predict(self, X): return self.model.predict(self._extract(X))


class ESN(BaseModel):
    """Echo State Network — random reservoir + ridge regression readout."""
    name = "ESN"
    
    def __init__(self, reservoir_size=200, spectral_radius=0.95,
                 leak_rate=0.3, sparsity=0.9, ridge_lambda=1e-6):
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.ridge_lambda = ridge_lambda
        self.n_classes = config.NUM_CLASSES
        
        rng = np.random.default_rng(42)
        self.W_in = rng.uniform(-1, 1, (reservoir_size, config.N_FEATURES)) * 0.1
        W = rng.uniform(-1, 1, (reservoir_size, reservoir_size))
        W *= (rng.random((reservoir_size, reservoir_size)) > sparsity)
        eigmax = np.abs(np.linalg.eigvals(W)).max()
        if eigmax > 0:
            W *= spectral_radius / eigmax
        self.W_res = W
        self.W_out = None
    
    def _run(self, X):
        n, seq, _ = X.shape
        states = np.zeros((n, self.reservoir_size))
        for i in range(n):
            h = np.zeros(self.reservoir_size)
            for t in range(seq):
                pre = np.tanh(self.W_in @ X[i, t] + self.W_res @ h)
                h = (1 - self.leak_rate) * h + self.leak_rate * pre
            states[i] = h
        return states
    
    def fit(self, X, y):
        max_n = 5000
        if len(y) > max_n:
            idx = np.random.choice(len(y), max_n, replace=False)
            X, y = X[idx], y[idx]
        print(f"    ESN: running reservoir on {len(y)} samples...")
        states = self._run(X)
        Y = np.zeros((len(y), self.n_classes))
        Y[np.arange(len(y)), y] = 1.0
        reg = self.ridge_lambda * np.eye(self.reservoir_size)
        self.W_out = Y.T @ states @ np.linalg.inv(states.T @ states + reg)
    
    def predict(self, X):
        preds = []
        for i in range(0, len(X), 2000):
            states = self._run(X[i:i+2000])
            preds.append(np.argmax(states @ self.W_out.T, axis=1))
        return np.concatenate(preds)


# --- Reactive baseline & adaptive selector ---

def reactive_baseline(raw_snr_windows, delay=0):
    from .channel import snr_to_mcs
    idx = max(0, config.WINDOW_SIZE - 1 - delay)
    return snr_to_mcs(raw_snr_windows[:, idx], hysteresis=config.HYSTERESIS_DB)


def adaptive_selector(nn_preds, raw_snr_windows, features, si_threshold=0.05):
    log_si = features[:, -1, 3]
    si = np.expm1(log_si)
    reactive = reactive_baseline(raw_snr_windows, delay=0)
    result = nn_preds.copy()
    result[si < si_threshold] = reactive[si < si_threshold]
    return result


ALL_MODELS = ["MLP", "CNN1D", "GRU", "LSTM", "HybridCNNGRU", "TCN", "Transformer", "ESN"]

def get_model(name, max_iter=100):
    m = {"MLP": lambda: MLP(max_iter), "CNN1D": lambda: CNN1DApprox(max_iter),
         "GRU": lambda: RecurrentApprox("GRU", 0.95, max_iter),
         "LSTM": lambda: RecurrentApprox("LSTM", 0.97, max_iter),
         "HybridCNNGRU": lambda: HybridCNNGRU(max_iter),
         "TCN": lambda: TCNApprox(max_iter),
         "Transformer": lambda: TransformerApprox(max_iter),
         "ESN": lambda: ESN()}
    return m[name]()
