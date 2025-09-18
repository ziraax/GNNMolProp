"""Molecular featurization."""
import torch
from rdkit import Chem
from typing import List, Optional
import numpy as np


class MolecularFeaturizer:
    """Molecular featurizer with essential features."""
    
    def __init__(self):
        self.atom_features = list(range(1, 119))  # Atomic numbers
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2, 
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
    
    @property
    def atom_feature_dim(self) -> int:
        return len(self.atom_features) + len(self.hybridizations) + 7  # +7 for other features
    
    def _one_hot(self, value, choices: List) -> List[float]:
        """One-hot encode value."""
        encoding = [0.0] * len(choices)
        try:
            encoding[choices.index(value)] = 1.0
        except ValueError:
            encoding[0] = 1.0  # Default to first
        return encoding
    
    def get_atom_features(self, atom: Chem.Atom) -> torch.Tensor:
        """Get atom features."""
        features = []
        features.extend(self._one_hot(atom.GetAtomicNum(), self.atom_features))
        features.extend(self._one_hot(atom.GetHybridization(), self.hybridizations))
        features.extend([
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalDegree()),
            float(atom.GetTotalNumHs()),
            float(atom.GetTotalValence()),
            float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        ])
        return torch.tensor(features, dtype=torch.float32)
    
    def mol_to_graph(self, smiles: str, labels: Optional[np.ndarray] = None):
        """Convert SMILES to graph data."""
        from torch_geometric.data import Data
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Atom features
        atom_features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_features:
            return None
        x = torch.stack(atom_features)
        
        # Edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Labels
        y = y_mask = None
        if labels is not None:
            y = torch.tensor(labels, dtype=torch.float32)
            y_mask = (y != -1.0).float()
            y = torch.where(y == -1.0, torch.zeros_like(y), y)
        
        return Data(x=x, edge_index=edge_index, y=y, y_mask=y_mask, smiles=smiles)