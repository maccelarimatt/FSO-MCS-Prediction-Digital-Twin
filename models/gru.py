"""GRU-based MCS Predictor — lighter alternative to LSTM."""

import torch
import torch.nn as nn
from .base import BaseMCSPredictor
from ..config import GRUConfig


class GRUPredictor(BaseMCSPredictor):
    """
    GRU for sequence classification.
    Fewer parameters than LSTM (no separate cell state), often
    comparable performance on shorter sequences.
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        config: GRUConfig = None,
    ):
        super().__init__(seq_length, n_features, n_classes)
        cfg = config or GRUConfig()
        
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        
        hidden_out = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_out, hidden_out // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_out // 2, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        return self.fc(self.dropout(last_output))
