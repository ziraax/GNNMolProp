"""Core configuration classes."""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "tox21"
    batch_size: int = 32
    num_workers: int = 4
    max_molecules: Optional[int] = None
    random_seed: int = 42


@dataclass 
class ModelConfig:
    """Model configuration."""
    model_type: str = "gin"
    hidden_dim: int = 128
    num_layers: int = 5
    dropout: float = 0.2
    pool_type: str = "add"


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 20
    scheduler_type: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    experiment_name: str = "gin_tox21"
    seed: int = 42
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        from datetime import datetime
        self.experiment_version = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def get_quick_test_config() -> ExperimentConfig:
    """Get quick test configuration."""
    config = ExperimentConfig()
    config.training.epochs = 3
    config.data.batch_size = 16
    config.data.max_molecules = 1000
    config.experiment_name = "quick_test"
    return config


def get_big_model_config() -> ExperimentConfig:
    """Get big model configuration with more parameters."""
    config = ExperimentConfig()
    
    # Larger model architecture
    config.model.hidden_dim = 512          # 4x larger than default (128)
    config.model.num_layers = 8            # More layers (vs 5 default)
    config.model.dropout = 0.3             # Higher dropout for regularization
    
    # Optimized training for larger model
    config.training.epochs = 200           # More epochs
    config.training.learning_rate = 5e-4   # Lower LR for stability
    config.training.weight_decay = 1e-4    # More regularization
    config.training.early_stopping_patience = 30  # More patience
    config.training.scheduler_patience = 15
    
    # Larger batch size for better gradients
    config.data.batch_size = 64
    
    config.experiment_name = "gin_tox21_big"
    return config