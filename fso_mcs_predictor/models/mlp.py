"""
MLP (Feed-Forward) MCS Predictor
=================================
Flattens the input window into a single vector and passes it through
fully connected layers. Has NO built-in temporal structure.

This is a critical baseline because:
  - If MLP performs nearly as well as LSTM/GRU, it means the SI context
    feature carries most of the information and temporal modelling adds
    little value. That's an important finding either way.
  - If MLP is significantly worse, it validates that sequence models
    capture temporal dynamics the MLP cannot.

Ref: Your document Section 1.1 — "works well if the right inputs are
provided (short history + SI estimate + maybe wind/context)"
"""

import torch
import torch.nn as nn
from .base import BaseMCSPredictor


class MLPPredictor(BaseMCSPredictor):
    """
    Feed-forward network that flattens the (window_size × n_features)
    input into a single vector.
    
    Architecture:
      flatten -> FC -> ReLU -> Dropout -> FC -> ReLU -> Dropout -> FC -> classes
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        hidden_sizes: tuple = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__(seq_length, n_features, n_classes)
        
        input_dim = seq_length * n_features
        layers = []
        prev_dim = input_dim
        
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> flatten to (batch, seq_len * n_features)
        x = x.reshape(x.size(0), -1)
        return self.network(x)
