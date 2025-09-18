"""Dataset classes."""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from typing import List, Optional
import numpy as np
import logging

from .featurizer import MolecularFeaturizer

logger = logging.getLogger(__name__)


class MolecularDataset(Dataset):
    """Molecular dataset."""
    
    def __init__(self, smiles_list: List[str], labels: Optional[np.ndarray] = None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.featurizer = MolecularFeaturizer()
        self.valid_indices = []
        self._process()
    
    def _process(self):
        """Process and validate molecules."""
        for idx, smiles in enumerate(self.smiles_list):
            labels = self.labels[idx] if self.labels is not None else None
            if self.featurizer.mol_to_graph(smiles, labels) is not None:
                self.valid_indices.append(idx)
        
        logger.info(f"Valid molecules: {len(self.valid_indices)}/{len(self.smiles_list)}")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        original_idx = self.valid_indices[idx]
        smiles = self.smiles_list[original_idx]
        labels = self.labels[original_idx] if self.labels is not None else None
        return self.featurizer.mol_to_graph(smiles, labels)