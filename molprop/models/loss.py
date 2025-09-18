"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Multi-task BCE loss with masking."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked BCE loss."""
        # Handle shape mismatch
        if logits.dim() == 2 and targets.dim() == 1:
            batch_size, num_tasks = logits.size()
            targets = targets.view(batch_size, num_tasks)
            mask = mask.view(batch_size, num_tasks)
        
        loss_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        masked_loss = loss_raw * mask
        return masked_loss.sum() / (mask.sum() + 1e-8)