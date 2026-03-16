"""
Transformer-based MCS Predictor
================================
Uses self-attention to capture temporal dependencies in the
channel observation sequence. Potentially better at capturing
long-range correlations than recurrent models.
"""

import torch
import torch.nn as nn
import math
from .base import BaseMCSPredictor
from ..config import TransformerConfig


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(BaseMCSPredictor):
    """
    Transformer encoder for MCS classification.
    
    Architecture:
      1. Linear projection from n_features to d_model
      2. Positional encoding
      3. Transformer encoder layers
      4. Mean pooling over sequence
      5. Classification head
    
    Uses a causal attention mask so the model can only attend
    to past and present observations (no future leakage).
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        config: TransformerConfig = None,
    ):
        super().__init__(seq_length, n_features, n_classes)
        cfg = config or TransformerConfig()
        
        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, seq_length, cfg.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_encoder_layers,
        )
        
        self.fc = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, n_classes),
        )
        
        # Causal mask: each position can only attend to itself and earlier
        mask = nn.Transformer.generate_square_subsequent_mask(seq_length)
        self.register_buffer('causal_mask', mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)        # -> (batch, seq_len, d_model)
        x = self.pos_enc(x)
        
        # Adjust mask size if sequence is shorter than expected
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        
        x = self.encoder(x, mask=mask)  # -> (batch, seq_len, d_model)
        
        # Use last position (most informed under causal mask)
        x = x[:, -1, :]
        return self.fc(x)
