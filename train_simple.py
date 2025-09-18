"""Simplified training script."""
import argparse
import sys
from pathlib import Path

# Add molprop to path
sys.path.insert(0, str(Path(__file__).parent))

from molprop import (
    ExperimentConfig,
    get_default_config,
    get_quick_test_config,
    get_big_model_config,
    DataModule,
    Trainer,
    setup_logging
)


def main():
    parser = argparse.ArgumentParser(description="Train molecular property prediction model")
    parser.add_argument("--quick-test", action="store_true", help="Quick test run")
    parser.add_argument("--big-model", action="store_true", help="Train big model with more parameters")
    parser.add_argument("--experiment-name", type=str, default="gin_tox21")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-molecules", type=int, help="Max molecules to use")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Get config
    if args.quick_test:
        config = get_quick_test_config()
    elif args.big_model:
        config = get_big_model_config()
    else:
        config = get_default_config()
    
    # Override with CLI args
    config.experiment_name = args.experiment_name
    config.training.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.model.dropout = args.dropout
    
    if args.max_molecules:
        config.data.max_molecules = args.max_molecules
    
    print(f"Config: {config.experiment_name}, epochs={config.training.epochs}")
    
    # Setup data
    data_module = DataModule(config.data)
    data_module.prepare_data()
    
    # Train
    trainer = Trainer(config, data_module)
    results = trainer.train()
    
    print(f"Training completed! Best val AUC: {results['best_val_auc']:.4f}")
    return 0


if __name__ == "__main__":
    exit(main())