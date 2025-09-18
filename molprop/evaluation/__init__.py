"""Evaluation metrics."""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_mask: np.ndarray, 
                     task_names: List[str]) -> Dict[str, float]:
    """Calculate classification metrics."""
    metrics = {}
    auc_scores = []
    acc_scores = []
    
    # Ensure arrays are 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, len(task_names))
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, len(task_names))
    if y_mask.ndim == 1:
        y_mask = y_mask.reshape(-1, len(task_names))
    
    for i, task_name in enumerate(task_names):
        mask = y_mask[:, i].astype(bool)
        if not mask.any() or len(np.unique(y_true[mask, i])) < 2:
            continue
            
        y_t = y_true[mask, i]
        y_p = y_prob[mask, i]
        y_pred = (y_p > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(y_t, y_p)
            acc = accuracy_score(y_t, y_pred)
            
            metrics[f"{task_name}_auc"] = auc
            metrics[f"{task_name}_acc"] = acc
            auc_scores.append(auc)
            acc_scores.append(acc)
        except:
            continue
    
    if auc_scores:
        metrics['mean_auc'] = np.mean(auc_scores)
        metrics['mean_acc'] = np.mean(acc_scores)
    
    return metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device, task_names: List[str]) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    
    all_probs = []
    all_targets = []
    all_masks = []
    total_loss = 0.0
    
    from ..models import create_loss
    loss_fn = create_loss()
    
    for batch in dataloader:
        batch = batch.to(device)
        logits = model(batch)
        loss = loss_fn(logits, batch.y, batch.y_mask)
        
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
        all_masks.append(batch.y_mask.cpu().numpy())
    
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)
    
    metrics = calculate_metrics(targets, probs, masks, task_names)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics