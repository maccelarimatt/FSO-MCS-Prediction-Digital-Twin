"""
Hybrid CNN-GRU MCS Predictor
==============================
Two-stage architecture:
  1. CNN front-end extracts local temporal features (fade bursts, 
     scintillation signatures) from short windows
  2. GRU back-end models the temporal evolution of those features
     to predict the upcoming MCS

This directly implements the architecture described in your document
Section 1.4 (rated "Excellent"). Uses GRU over LSTM for the recurrent
part because:
  - Fewer parameters (no separate cell state)
  - Comparable performance on moderate-length sequences
  - More realistic for GNU Radio deployment (Phase 3)

The CNN-GRU hybrid tests a specific hypothesis: that separating 
"what patterns are present" (CNN) from "how patterns evolve over 
time" (GRU) gives better performance than either alone.
"""

import torch
import torch.nn as nn
from .base import BaseMCSPredictor


class HybridCNNGRUPredictor(BaseMCSPredictor):
    """
    CNN feature extractor followed by GRU temporal modeller.
    
    The CNN operates on local windows within the sequence using
    1D convolutions, producing a feature sequence. The GRU then
    processes this feature sequence and outputs from its final 
    hidden state.
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        cnn_channels: tuple = (32, 64),
        cnn_kernel: int = 5,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__(seq_length, n_features, n_classes)
        
        # --- CNN front-end ---
        conv_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)
        
        # --- GRU back-end ---
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        
        # --- Classification head ---
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        
        # CNN expects (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        x_conv = self.cnn(x_conv)           # (batch, cnn_out, seq_len)
        
        # Back to (batch, seq_len, cnn_out) for GRU
        x_seq = x_conv.transpose(1, 2)
        
        # GRU processes the CNN feature sequence
        output, _ = self.gru(x_seq)         # (batch, seq_len, gru_hidden)
        last = output[:, -1, :]             # final hidden state
        
        return self.fc(self.dropout(last))
