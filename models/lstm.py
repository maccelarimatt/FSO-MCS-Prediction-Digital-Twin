"""LSTM-based MCS Predictor — the baseline recurrent architecture."""

import torch
import torch.nn as nn
from .base import BaseMCSPredictor
from ..config import LSTMConfig


class LSTMPredictor(BaseMCSPredictor):
    """
    Standard LSTM for sequence classification.
    
    Uses the final hidden state (or mean-pooled outputs if bidirectional)
    to classify the upcoming MCS level.
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        config: LSTMConfig = None,
    ):
        super().__init__(seq_length, n_features, n_classes)
        cfg = config or LSTMConfig()
        
        self.lstm = nn.LSTM(
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
        # x: (batch, seq_len, n_features)
        output, (h_n, _) = self.lstm(x)
        # Use last time step output
        last_output = output[:, -1, :]
        return self.fc(self.dropout(last_output))
