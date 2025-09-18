"""GIN model implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from typing import Optional

from ..config import ModelConfig


class GINLayer(nn.Module):
    """GIN layer with MLP."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.conv = GINConv(mlp)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        return self.dropout(x)


class GINModel(nn.Module):
    """GIN model for molecular property prediction."""
    
    def __init__(self, config: ModelConfig, atom_feat_dim: int, num_tasks: int):
        super().__init__()
        self.config = config
        
        # Layers
        self.layers = nn.ModuleList()
        dims = [atom_feat_dim] + [config.hidden_dim] * config.num_layers
        
        for i in range(config.num_layers):
            self.layers.append(GINLayer(dims[i], dims[i+1], config.dropout))
        
        # Pooling
        self.pool_fn = self._get_pool_fn(config.pool_type)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_tasks)
        )
    
    def _get_pool_fn(self, pool_type: str):
        """Get pooling function."""
        if pool_type == "add":
            return global_add_pool
        elif pool_type == "mean":
            return global_mean_pool
        elif pool_type == "max":
            return global_max_pool
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
    
    def forward(self, batch) -> torch.Tensor:
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        for layer in self.layers:
            x = layer(x, edge_index)
        
        x = self.pool_fn(x, batch_idx)
        return self.predictor(x)