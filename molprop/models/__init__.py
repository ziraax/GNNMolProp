"""Model factory and registry."""
from .gin import GINModel
from .loss import MultiTaskLoss
from ..config import ModelConfig


def create_model(config: ModelConfig, atom_feat_dim: int, num_tasks: int):
    """Create model based on config."""
    if config.model_type.lower() == "gin":
        return GINModel(config, atom_feat_dim, num_tasks)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def create_loss():
    """Create loss function."""
    return MultiTaskLoss()