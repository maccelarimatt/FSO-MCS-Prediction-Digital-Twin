"""
Model Registry
==============
Factory for creating MCS predictor instances by name.
Makes it easy to loop over architectures for comparison.
"""

from typing import Dict, Type
from .base import BaseMCSPredictor
from .mlp import MLPPredictor
from .cnn1d import CNN1DPredictor
from .lstm import LSTMPredictor
from .gru import GRUPredictor
from .hybrid import HybridCNNGRUPredictor
from .transformer import TransformerPredictor
from .esn import EchoStateNetwork
from .tcn import TCNPredictor
from .baseline import ReactiveAMCBaseline, run_baseline_sweep

# Registry mapping model names to classes
# Ordered roughly by complexity for clean comparison tables
MODEL_REGISTRY: Dict[str, Type[BaseMCSPredictor]] = {
    "mlp": MLPPredictor,
    "cnn1d": CNN1DPredictor,
    "lstm": LSTMPredictor,
    "gru": GRUPredictor,
    "hybrid_cnn_gru": HybridCNNGRUPredictor,
    "tcn": TCNPredictor,
    "transformer": TransformerPredictor,
    "esn": EchoStateNetwork,
}


def create_model(
    name: str,
    seq_length: int,
    n_features: int,
    n_classes: int,
    **kwargs,
) -> BaseMCSPredictor:
    """
    Create a model by name.
    
    Args:
        name: One of 'lstm', 'gru', 'transformer', 'esn', 'tcn'.
        seq_length: Input sequence length.
        n_features: Number of features per timestep.
        n_classes: Number of MCS classes (15: outage + 14 MCS levels).
        **kwargs: Architecture-specific config.
    
    Returns:
        Instantiated model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](seq_length, n_features, n_classes, **kwargs)


def list_models():
    """Return available model names."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "BaseMCSPredictor", "MLPPredictor", "CNN1DPredictor",
    "LSTMPredictor", "GRUPredictor", "HybridCNNGRUPredictor",
    "TransformerPredictor", "EchoStateNetwork", "TCNPredictor",
    "ReactiveAMCBaseline", "run_baseline_sweep",
    "MODEL_REGISTRY", "create_model", "list_models",
]
