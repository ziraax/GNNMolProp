"""Main trainer class."""
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import logging

from .utils import EarlyStopping, get_lr, count_parameters
from ..config import ExperimentConfig
from ..data import DataModule
from ..models import create_model, create_loss
from ..utils import set_seed, get_device

logger = logging.getLogger(__name__)


class Trainer:
    """Simplified trainer class."""
    
    def __init__(self, config: ExperimentConfig, data_module: DataModule):
        self.config = config
        self.data_module = data_module
        
        # Setup device and model
        self.device = get_device()
        atom_feat_dim, _ = data_module.get_feature_dims()
        num_tasks = data_module.get_num_tasks()
        
        self.model = create_model(config.model, atom_feat_dim, num_tasks).to(self.device)
        self.loss_fn = create_loss()
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience, mode='max'
        )
        
        # Setup directories
        self.save_dir = Path("results") / config.experiment_name / config.experiment_version
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        logger.info(f"Save directory: {self.save_dir}")
    
    def train_epoch(self, train_loader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            logits = self.model(batch)
            loss = self.loss_fn(logits, batch.y, batch.y_mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []
        all_masks = []
        
        for batch in val_loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            
            # Handle shape mismatch in loss calculation
            batch_size, num_tasks = logits.size()
            y = batch.y
            y_mask = batch.y_mask
            
            if y.dim() == 1:
                y = y.view(batch_size, num_tasks)
            if y_mask.dim() == 1:
                y_mask = y_mask.view(batch_size, num_tasks)
            
            loss = self.loss_fn(logits, y, y_mask)
            
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(y.cpu())
            all_masks.append(y_mask.cpu())
        
        # Calculate AUC
        probs = torch.cat(all_probs)
        targets = torch.cat(all_targets)
        masks = torch.cat(all_masks)
        
        auc_scores = []
        for task_idx in range(probs.shape[1]):
            mask = masks[:, task_idx].bool()
            if mask.sum() > 0 and len(torch.unique(targets[mask, task_idx])) > 1:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(targets[mask, task_idx], probs[mask, task_idx])
                    auc_scores.append(auc)
                except:
                    pass
        
        mean_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'auc': mean_auc
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'hidden_dim': self.config.model.hidden_dim,
                'num_layers': self.config.model.num_layers,
                'dropout': self.config.model.dropout,
                'pool_type': self.config.model.pool_type,
                'experiment_name': self.config.experiment_name
            },
            'task_names': self.data_module.get_task_names(),
            'num_tasks': self.data_module.get_num_tasks(),
            'atom_feat_dim': self.data_module.get_feature_dims()[0]
        }
        
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        set_seed(self.config.seed)
        train_loader, val_loader, test_loader = self.data_module.get_dataloaders()
        
        logger.info("Starting training...")
        best_auc = 0.0
        
        for epoch in range(1, self.config.training.epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation  
            val_metrics = self.validate(val_loader)
            val_loss, val_auc = val_metrics['loss'], val_metrics['auc']
            
            # Update scheduler
            old_lr = get_lr(self.optimizer)
            self.scheduler.step(val_auc)
            new_lr = get_lr(self.optimizer)
            
            # Check if best model
            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc
                self.save_checkpoint(epoch, is_best=True)
            
            # Logging
            lr_str = f" | LR: {old_lr:.2e}"
            if new_lr < old_lr:
                lr_str += f" -> {new_lr:.2e}"
            
            logger.info(
                f"Epoch {epoch:03d} | "
                f"TrainLoss: {train_loss:.4f} | "
                f"ValLoss: {val_loss:.4f} | "
                f"ValAUC: {val_auc:.4f}{lr_str} | "
                f"{'⭐' if is_best else ''}"
            )
            
            # Early stopping
            if self.early_stopping(val_auc, self.model):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        logger.info("Final evaluation...")
        if (self.save_dir / "best_model.pt").exists():
            checkpoint = torch.load(self.save_dir / "best_model.pt", weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.validate(test_loader)
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        
        return {
            'best_val_auc': best_auc,
            'test_auc': test_metrics['auc']
        }