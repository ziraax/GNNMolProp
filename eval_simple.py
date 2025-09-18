"""Simplified evaluation script."""
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from molprop import DataModule, create_model, evaluate_model, get_device, setup_logging
from molprop.config import DataConfig, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Setup data
    data_config = DataConfig()
    data_config.batch_size = args.batch_size
    data_module = DataModule(data_config)
    data_module.prepare_data()
    
    # Create model
    model_config = ModelConfig()
    atom_feat_dim, _ = data_module.get_feature_dims()
    num_tasks = data_module.get_num_tasks()
    
    model = create_model(model_config, atom_feat_dim, num_tasks)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Evaluate
    _, _, test_loader = data_module.get_dataloaders()
    results = evaluate_model(model, test_loader, device, data_module.get_task_names())
    
    print(f"Test Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Mean AUC: {results.get('mean_auc', 'N/A'):.4f}")
    print(f"  Mean Accuracy: {results.get('mean_acc', 'N/A'):.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())