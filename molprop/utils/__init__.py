"""Core utilities."""
import random
import numpy as np
import torch
import logging


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get computation device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device


def setup_logging(level: str = "INFO"):
    """Setup logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )