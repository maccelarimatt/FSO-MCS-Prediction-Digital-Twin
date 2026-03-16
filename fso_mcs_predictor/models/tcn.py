"""
Temporal Convolutional Network (TCN) MCS Predictor
===================================================
Uses dilated causal convolutions to capture temporal patterns
without recurrence. Often faster to train than RNNs and can
match or exceed their performance on many sequence tasks.

Architecture follows Bai et al., "An Empirical Evaluation of 
Generic Convolutional and Recurrent Networks for Sequence 
Modeling," 2018.
"""

import torch
import torch.nn as nn
from typing import List
from .base import BaseMCSPredictor
from ..config import TCNConfig


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding to prevent future leakage."""
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove right padding to enforce causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Residual block with two causal convolutions, weight norm, 
    and dropout.
    """
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, dilation: int, dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual connection (with 1x1 conv if channels change)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TCNPredictor(BaseMCSPredictor):
    """
    TCN for MCS classification.
    
    Stacks temporal blocks with exponentially increasing dilation
    to achieve a large receptive field with few layers.
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        config: TCNConfig = None,
    ):
        super().__init__(seq_length, n_features, n_classes)
        cfg = config or TCNConfig()
        
        layers = []
        num_levels = len(cfg.num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = n_features if i == 0 else cfg.num_channels[i - 1]
            out_ch = cfg.num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, cfg.kernel_size, dilation, cfg.dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        self.fc = nn.Sequential(
            nn.Linear(cfg.num_channels[-1], cfg.num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.num_channels[-1] // 2, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> need (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)     # (batch, channels, seq_len)
        x = x[:, :, -1]         # last time step
        return self.fc(x)
