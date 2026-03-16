"""
Training Engine
===============
Unified training loop that works with all model architectures.
Handles class imbalance weighting, early stopping, LR scheduling,
and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .config import TrainingConfig, NUM_MCS_CLASSES
from .models.base import BaseMCSPredictor
from .models.esn import EchoStateNetwork
from .dataset.generator import FSODataset


class Trainer:
    """
    Trains and validates MCS predictor models.
    
    Handles:
      - Class-weighted cross-entropy (for imbalanced MCS distribution)
      - Early stopping on validation loss
      - Learning rate scheduling
      - Special ESN ridge regression path
    """
    
    def __init__(
        self,
        model: BaseMCSPredictor,
        train_dataset: FSODataset,
        val_dataset: FSODataset,
        config: TrainingConfig = None,
        output_dir: str = "./results",
    ):
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)
        
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )
        
        # Class weights to handle imbalanced MCS distribution
        class_weights = self._compute_class_weights(train_dataset)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }
    
    def _compute_class_weights(self, dataset: FSODataset) -> torch.Tensor:
        """
        Compute inverse-frequency class weights.
        Classes that appear less often get higher weight so the model
        doesn't just predict the most common MCS level.
        """
        targets = dataset.targets.numpy()
        class_counts = np.bincount(targets, minlength=NUM_MCS_CLASSES)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts.astype(np.float32)
        weights /= weights.sum()  # normalise
        weights *= len(weights)   # scale so mean weight ≈ 1
        return torch.FloatTensor(weights)
    
    def train(self) -> Dict:
        """
        Run the full training loop.
        
        Returns:
            History dict with losses, accuracies, and timings.
        """
        # Special path for ESN: ridge regression, not SGD
        if isinstance(self.model, EchoStateNetwork):
            return self._train_esn()
        
        return self._train_sgd()
    
    def _train_sgd(self) -> Dict:
        """Standard SGD training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        start_time = time.time()
        
        print(f"\nTraining {self.model.__class__.__name__} "
              f"({self.model.count_parameters():,} params) on {self.device}")
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Train Acc':>9} | {'Val Acc':>9} | {'LR':>10}")
        print("-" * 65)
        
        for epoch in range(self.config.max_epochs):
            # --- Train ---
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for features, targets in self.train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                loss.backward()
                
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item() * len(targets)
                train_correct += (logits.argmax(1) == targets).sum().item()
                train_total += len(targets)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # --- Validate ---
            val_loss, val_acc = self._evaluate(self.val_loader)
            
            # --- LR schedule ---
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            
            # --- Log ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            
            print(f"{epoch+1:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
                  f"{train_acc:>8.1%} | {val_acc:>8.1%} | {current_lr:>10.2e}")
            
            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() 
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")
        
        self.history["training_time_s"] = elapsed
        self.history["best_val_loss"] = best_val_loss
        self.history["epochs_trained"] = len(self.history["train_loss"])
        
        return self.history
    
    def _train_esn(self) -> Dict:
        """Train ESN via ridge regression (one-shot)."""
        print(f"\nTraining {self.model.__class__.__name__} via ridge regression "
              f"({self.model.count_parameters():,} trainable params)")
        
        start_time = time.time()
        
        self.model.train_ridge(
            self.train_dataset.features.to(self.device),
            self.train_dataset.targets.to(self.device),
        )
        
        elapsed = time.time() - start_time
        
        # Evaluate
        _, train_acc = self._evaluate(self.train_loader)
        val_loss, val_acc = self._evaluate(self.val_loader)
        
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Val accuracy:   {val_acc:.1%}")
        print(f"  Training time:  {elapsed:.1f}s")
        
        self.history = {
            "train_loss": [0.0],
            "val_loss": [val_loss],
            "train_acc": [train_acc],
            "val_acc": [val_acc],
            "lr": [0.0],
            "training_time_s": elapsed,
            "best_val_loss": val_loss,
            "epochs_trained": 1,
        }
        
        return self.history
    
    @torch.no_grad()
    def _evaluate(
        self, loader: DataLoader
    ) -> Tuple[float, float]:
        """Compute loss and accuracy on a data loader."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            logits = self.model(features)
            loss = self.criterion(logits, targets)
            
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)
        
        return total_loss / max(total, 1), correct / max(total, 1)
