# Molecular Property Prediction - Modular Package

## Overview
A professional-grade, modular implementation of molecular property prediction using Graph Isomorphism Networks (GIN) for multi-task binary classification on the Tox21 dataset.

## Package Structure

```
molprop/                        # Main package
├── __init__.py                 # Package initialization
├── config/                     # Configuration management
│   └── __init__.py            # DataConfig, ModelConfig, TrainingConfig classes
├── data/                       # Data handling and featurization
│   ├── __init__.py            # DataModule class
│   ├── dataset.py             # TorchGeometric dataset implementation
│   └── featurizer.py          # Molecular graph featurization
├── models/                     # Model architectures and loss functions
│   ├── __init__.py            # Model factory and utilities
│   ├── gin.py                 # GIN model implementation
│   └── loss.py                # Multi-task loss with masking
├── training/                   # Training framework
│   ├── __init__.py            # Trainer class
│   └── utils.py               # Training utilities (EarlyStopping)
├── evaluation/                 # Evaluation and metrics
│   └── __init__.py            # Evaluation utilities
└── utils/                      # Core utilities
    └── __init__.py            # Utility functions
```

## Entry Points

- `train_simple.py` - Simple training script with default configs
- `eval_simple.py` - Simple evaluation script for saved models

## Key Features

### Configuration System
- Dataclass-based configuration with type hints
- Factory functions for different experiment setups
- Automatic directory creation and logging setup

### Data Pipeline
- RDKit-based molecular featurization
- PyTorch Geometric graph data handling
- Automatic train/val/test splitting
- Proper masking for missing labels in multi-task learning

### Model Architecture
- Graph Isomorphism Network (GIN) with configurable layers
- Multiple pooling strategies (mean, max, sum)
- Dropout and batch normalization options
- Multi-task binary classification head

### Training Framework
- Early stopping with configurable patience
- Learning rate scheduling
- Automatic model checkpointing
- Comprehensive logging and metrics tracking

### Evaluation
- Per-task AUC calculation with proper masking
- Test set evaluation with best validation model
- Metric visualization and result saving

## Usage

### Quick Training
```bash
python train_simple.py --quick-test  # Fast test run with 5 epochs
python train_simple.py              # Full training with default config
```

### Evaluation
```bash
python eval_simple.py --checkpoint results/gin_tox21/timestamp/best_model.pt
```

### Custom Configuration
```python
from molprop.config import get_default_config, ExperimentConfig
from molprop.training import Trainer
from molprop.data import DataModule

# Create custom config
config = get_default_config()
config.training.learning_rate = 0.001
config.training.epochs = 50

# Initialize components
data_module = DataModule(config.data)
trainer = Trainer(config)

# Run training
results = trainer.train()
```

## Dependencies

- PyTorch & PyTorch Geometric
- RDKit for molecular handling
- DeepChem for Tox21 dataset
- scikit-learn for metrics
- Standard scientific Python stack (numpy, pandas, etc.)

## Results

The model achieves competitive performance on Tox21 molecular toxicity prediction tasks:
- Multi-task AUC scores across 12 toxicity endpoints
- Proper handling of missing labels through masking
- Robust training with early stopping and checkpointing

## Design Principles

1. **Modularity**: Clean separation of concerns across packages
2. **Configurability**: Dataclass-based configuration system
3. **Reusability**: Factory patterns and dependency injection
4. **Maintainability**: Type hints, logging, and comprehensive testing
5. **Performance**: Efficient batching and GPU utilization
6. **Reproducibility**: Seed setting and deterministic operations