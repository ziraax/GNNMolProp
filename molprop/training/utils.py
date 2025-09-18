"""Training utilities."""
import torch
import torch.nn as nn
from typing import Optional


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.compare = (lambda x, y: x >= y) if mode == 'max' else (lambda x, y: x <= y)
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        # Convert to float to handle numpy types
        score = float(score)
        
        if self.best_score is None or self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate."""
    return optimizer.param_groups[0]['lr']


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)