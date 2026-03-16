"""
Base Predictor Interface
========================
All MCS predictor architectures inherit from this base class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseMCSPredictor(nn.Module, ABC):
    """
    Abstract base for MCS prediction models.
    
    Input:  (batch, seq_length, n_features)
    Output: (batch, n_classes) — logits over MCS classes
    """
    
    def __init__(self, seq_length: int, n_features: int, n_classes: int):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_classes = n_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def model_summary(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "parameters": self.count_parameters(),
            "seq_length": self.seq_length,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
        }
