"""
1D Convolutional Neural Network MCS Predictor
==============================================
Applies 1D convolutions over the time axis to learn local temporal
patterns (fade signatures, scintillation bursts) without recurrence.

Sits between MLP (no temporal structure) and TCN (dilated causal convs)
in the complexity hierarchy. Tests whether simple local pattern 
extraction is enough, or whether you need the larger receptive field
of TCN or the explicit memory of LSTM/GRU.

Ref: Your document Section 1.2 — "fast at inference, parallelisable,
and stable to train"
"""

import torch
import torch.nn as nn
from .base import BaseMCSPredictor


class CNN1DPredictor(BaseMCSPredictor):
    """
    1D CNN for MCS classification.
    
    Architecture:
      Conv1D -> BN -> ReLU -> Conv1D -> BN -> ReLU -> 
      Conv1D -> BN -> ReLU -> GlobalAvgPool -> FC head
    
    Uses standard (non-causal) convolutions with same-padding.
    This is acceptable here because the entire window is a past
    observation — there's no future leakage at the window level.
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        channels: tuple = (32, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__(seq_length, n_features, n_classes)
        
        conv_layers = []
        in_ch = n_features
        for out_ch in channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Global average pooling over the time dimension, then classify
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)              # (batch, channels[-1], seq_len)
        x = x.mean(dim=2)             # global average pool -> (batch, channels[-1])
        return self.fc(x)
