"""
Echo State Network (ESN) MCS Predictor
=======================================
Reservoir computing approach where only the output layer is trained.

This is particularly relevant for your research because:
  - Very fast training (ridge regression, not backprop)
  - Potentially deployable on resource-constrained hardware
  - The fixed reservoir acts as a nonlinear feature expander
  
The ESN processes the sequence through a large random recurrent
network (reservoir), then trains a linear readout on the reservoir
states. For fair comparison with gradient-based models, we wrap
it in the same PyTorch interface but use a separate training path.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .base import BaseMCSPredictor
from ..config import ESNConfig


class EchoStateNetwork(BaseMCSPredictor):
    """
    Echo State Network for MCS classification.
    
    The reservoir weights are fixed (not trained). Only the 
    readout layer is optimised, via ridge regression or SGD.
    
    NOTE: For the comparison study, we provide both:
      1. Ridge regression training (classic ESN, very fast)
      2. SGD-compatible forward() for the unified training loop
    """
    
    def __init__(
        self, seq_length: int, n_features: int, n_classes: int,
        config: ESNConfig = None,
    ):
        super().__init__(seq_length, n_features, n_classes)
        cfg = config or ESNConfig()
        self.cfg = cfg
        
        # --- Reservoir (fixed, not trained) ---
        # Input weights
        W_in = np.random.uniform(
            -cfg.input_scaling, cfg.input_scaling,
            size=(cfg.reservoir_size, n_features)
        )
        self.register_buffer(
            'W_in', torch.FloatTensor(W_in)
        )
        
        # Recurrent weights (sparse, scaled to spectral radius)
        W_res = np.random.randn(cfg.reservoir_size, cfg.reservoir_size)
        # Apply sparsity
        mask = np.random.rand(*W_res.shape) < cfg.sparsity
        W_res[mask] = 0.0
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        spectral_radius_actual = np.max(np.abs(eigenvalues))
        if spectral_radius_actual > 0:
            W_res *= cfg.spectral_radius / spectral_radius_actual
        self.register_buffer(
            'W_res', torch.FloatTensor(W_res)
        )
        
        self.leaking_rate = cfg.leaking_rate
        
        # --- Trainable readout ---
        self.readout = nn.Linear(cfg.reservoir_size, n_classes)
    
    def _run_reservoir(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run input through the reservoir to get state trajectory.
        
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            states: (batch, reservoir_size) — final reservoir state
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(
            batch_size, self.cfg.reservoir_size, 
            device=x.device, dtype=x.dtype
        )
        
        for t in range(seq_len):
            u = x[:, t, :]  # (batch, n_features)
            # Reservoir update with leaky integration
            pre_activation = (
                torch.mm(u, self.W_in.T) + torch.mm(h, self.W_res.T)
            )
            h = (
                (1 - self.leaking_rate) * h 
                + self.leaking_rate * torch.tanh(pre_activation)
            )
        
        return h
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (for SGD-based training loop)."""
        h = self._run_reservoir(x)
        return self.readout(h)
    
    def train_ridge(
        self,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
        reg_lambda: Optional[float] = None,
    ):
        """
        Train the readout via ridge regression (classic ESN training).
        
        Much faster than SGD — processes the entire dataset in one shot.
        
        Args:
            train_features: (n_samples, seq_len, n_features)
            train_targets: (n_samples,) — class indices
        """
        reg = reg_lambda or self.cfg.regularization
        
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = 512
            n = len(train_features)
            all_states = []
            
            for i in range(0, n, batch_size):
                batch = train_features[i:i+batch_size]
                states = self._run_reservoir(batch)
                all_states.append(states)
            
            H = torch.cat(all_states, dim=0)  # (n, reservoir_size)
            
            # One-hot encode targets
            Y = torch.zeros(n, self.n_classes, device=H.device)
            Y.scatter_(1, train_targets.unsqueeze(1), 1.0)
            
            # Ridge regression: W = (H^T H + λI)^{-1} H^T Y
            I = torch.eye(H.shape[1], device=H.device)
            W = torch.linalg.solve(
                H.T @ H + reg * I,
                H.T @ Y,
            )
            
            # Set readout weights
            self.readout.weight.data = W.T
            self.readout.bias.data.zero_()
