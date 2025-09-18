"""Data loading utilities."""
import deepchem as dc
from torch_geometric.data import DataLoader
from rdkit import RDLogger
import logging

from .dataset import MolecularDataset
from ..config import DataConfig
from ..utils import set_seed

RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)


class DataModule:
    """Data module for loading and preparing datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.task_names = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Load and prepare data."""
        if self.config.dataset_name.lower() == "tox21":
            self._load_tox21()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
    
    def _load_tox21(self):
        """Load Tox21 dataset."""
        tasks, datasets, _ = dc.molnet.load_tox21(featurizer='Raw')
        train_dc, val_dc, test_dc = datasets
        self.task_names = tasks
        
        def extract_data(dc_dataset):
            smiles = [dc_dataset.ids[i] for i in range(len(dc_dataset))]
            labels = dc_dataset.y
            if self.config.max_molecules:
                n = min(self.config.max_molecules, len(smiles))
                smiles, labels = smiles[:n], labels[:n]
            return smiles, labels
        
        train_smiles, train_labels = extract_data(train_dc)
        val_smiles, val_labels = extract_data(val_dc)
        test_smiles, test_labels = extract_data(test_dc)
        
        self.train_dataset = MolecularDataset(train_smiles, train_labels)
        self.val_dataset = MolecularDataset(val_smiles, val_labels)
        self.test_dataset = MolecularDataset(test_smiles, test_labels)
        
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Datasets: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
    
    def get_dataloaders(self):
        """Get data loaders."""
        set_seed(self.config.random_seed)
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, 
                                shuffle=True, num_workers=self.config.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size,
                              shuffle=False, num_workers=self.config.num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size,
                               shuffle=False, num_workers=self.config.num_workers)
        
        return train_loader, val_loader, test_loader
    
    def get_feature_dims(self):
        """Get feature dimensions."""
        featurizer = MolecularDataset(["C"], None).featurizer
        return featurizer.atom_feature_dim, 0
    
    def get_num_tasks(self) -> int:
        return len(self.task_names)
    
    def get_task_names(self) -> list:
        return self.task_names.copy()