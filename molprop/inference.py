"""Inference utilities for molecular property prediction."""
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
import logging

from .data.featurizer import MolecularFeaturizer
from .models import create_model
from .config import ModelConfig, DataConfig

logger = logging.getLogger(__name__)


class MolecularPredictor:
    """Molecular property predictor for inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """Initialize predictor with trained model.
        
        Args:
            checkpoint_path: Path to saved model checkpoint
            device: Device to run inference on ("cpu", "cuda", or "auto")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._get_device(device)
        self.featurizer = MolecularFeaturizer()
        
        # Load model and config
        self.model, self.config, self.task_names = self._load_model()
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract config and task info
        config = checkpoint.get('config', {})
        task_names = checkpoint.get('task_names', [])
        num_tasks = checkpoint.get('num_tasks', len(task_names))
        atom_feat_dim = checkpoint.get('atom_feat_dim', self.featurizer.atom_feature_dim)
        
        # Create model config from checkpoint
        model_config = ModelConfig(
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 5),
            dropout=config.get('dropout', 0.2),
            pool_type=config.get('pool_type', 'add')
        )
        
        # Create and load model
        model = create_model(model_config, atom_feat_dim, num_tasks)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model with {num_tasks} tasks: {task_names}")
        
        return model, config, task_names
    
    def predict_smiles(self, smiles: str) -> Dict[str, float]:
        """Predict properties for a single SMILES string.
        
        Args:
            smiles: SMILES string of molecule
            
        Returns:
            Dictionary mapping task names to prediction probabilities
        """
        return self.predict_batch([smiles])[0]
    
    def predict_batch(self, smiles_list: List[str]) -> List[Dict[str, float]]:
        """Predict properties for multiple SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of dictionaries mapping task names to prediction probabilities
        """
        from torch_geometric.loader import DataLoader
        
        # Convert all SMILES to graphs
        graphs = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            graph_data = self.featurizer.mol_to_graph(smiles, labels=None)
            if graph_data is not None:
                graphs.append(graph_data)
                valid_indices.append(i)
            else:
                logger.warning(f"Invalid SMILES: {smiles}")
        
        # Create results list
        results = []
        
        if not graphs:
            # All molecules invalid
            results = [{task: None for task in self.task_names} for _ in smiles_list]
        else:
            # Create dataloader for batch processing
            dataloader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    logits = self.model(batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    
                    # Process results
                    batch_results = []
                    for prob_vec in probs:
                        result = {task: float(prob) for task, prob in zip(self.task_names, prob_vec)}
                        batch_results.append(result)
            
            # Map results back to original order
            results = [{task: None for task in self.task_names} for _ in smiles_list]
            for valid_idx, result in zip(valid_indices, batch_results):
                results[valid_idx] = result
        
        return results
    
    def predict_file(self, input_file: str, output_file: Optional[str] = None) -> None:
        """Predict properties for molecules in a file.
        
        Args:
            input_file: Path to file with SMILES strings (one per line)
            output_file: Path to save results (optional)
        """
        import pandas as pd
        
        # Read SMILES
        with open(input_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Predicting for {len(smiles_list)} molecules from {input_file}")
        
        # Get predictions
        predictions = self.predict_batch(smiles_list)
        
        # Create DataFrame
        data = []
        for smiles, pred in zip(smiles_list, predictions):
            row = {'smiles': smiles}
            row.update(pred)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return df
    
    def get_task_names(self) -> List[str]:
        """Get list of prediction task names."""
        return self.task_names.copy()
    
    def get_model_info(self) -> Dict:
        """Get model configuration information."""
        return {
            'num_tasks': len(self.task_names),
            'task_names': self.task_names,
            'hidden_dim': self.config.get('hidden_dim', 'unknown'),
            'num_layers': self.config.get('num_layers', 'unknown'),
            'device': str(self.device),
            'checkpoint_path': str(self.checkpoint_path)
        }


def load_predictor(checkpoint_path: str, device: str = "auto") -> MolecularPredictor:
    """Convenience function to load a molecular predictor.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        device: Device to run inference on
        
    Returns:
        MolecularPredictor instance
    """
    return MolecularPredictor(checkpoint_path, device)