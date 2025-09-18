"""Simple inference script for molecular property prediction."""
import argparse
import sys
import json
from pathlib import Path

# Add molprop to path
sys.path.insert(0, str(Path(__file__).parent))

from molprop.inference import MolecularPredictor
from molprop import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Predict molecular properties from SMILES")
    parser.add_argument("checkpoint", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--smiles", type=str, help="Single SMILES string to predict")
    parser.add_argument("--input-file", type=str, help="File with SMILES strings (one per line)")
    parser.add_argument("--output-file", type=str, help="Output file for predictions")
    parser.add_argument("--batch-smiles", type=str, nargs="+", help="Multiple SMILES strings")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Device for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--format", type=str, default="table", choices=["table", "json", "csv"],
                       help="Output format")
    
    args = parser.parse_args()
    
    if not any([args.smiles, args.input_file, args.batch_smiles]):
        parser.error("Must provide --smiles, --input-file, or --batch-smiles")
    
    setup_logging()
    
    # Load predictor
    print(f"Loading model from {args.checkpoint}...")
    predictor = MolecularPredictor(args.checkpoint, device=args.device)
    
    # Print model info
    info = predictor.get_model_info()
    print(f"Model loaded: {info['num_tasks']} tasks, device: {info['device']}")
    print(f"Tasks: {', '.join(info['task_names'])}")
    print()
    
    # Get predictions
    if args.smiles:
        # Single SMILES
        print(f"Predicting for SMILES: {args.smiles}")
        result = predictor.predict_smiles(args.smiles)
        print_results([args.smiles], [result], args.format, args.threshold)
        
    elif args.batch_smiles:
        # Multiple SMILES from command line
        print(f"Predicting for {len(args.batch_smiles)} molecules...")
        results = predictor.predict_batch(args.batch_smiles)
        print_results(args.batch_smiles, results, args.format, args.threshold)
        
    elif args.input_file:
        # File input
        print(f"Predicting for molecules from {args.input_file}...")
        df = predictor.predict_file(args.input_file, args.output_file)
        
        if args.format == "table":
            print("\nResults:")
            print(df.to_string(index=False))
        elif args.format == "json":
            print(df.to_json(orient="records", indent=2))
        
        if args.output_file:
            print(f"\nResults saved to {args.output_file}")
    
    return 0


def print_results(smiles_list, results, format_type="table", threshold=0.5):
    """Print prediction results in specified format."""
    
    if format_type == "json":
        output = []
        for smiles, result in zip(smiles_list, results):
            output.append({"smiles": smiles, "predictions": result})
        print(json.dumps(output, indent=2))
        
    elif format_type == "csv":
        print("smiles," + ",".join(results[0].keys()))
        for smiles, result in zip(smiles_list, results):
            values = [str(v) if v is not None else "NA" for v in result.values()]
            print(f"{smiles}," + ",".join(values))
            
    else:  # table format
        print("\nPrediction Results:")
        print("-" * 80)
        
        for smiles, result in zip(smiles_list, results):
            print(f"SMILES: {smiles}")
            
            if any(v is None for v in result.values()):
                print("  ERROR: Invalid molecule")
            else:
                print("  Predictions (probability):")
                for task, prob in result.items():
                    status = "POSITIVE" if prob > threshold else "NEGATIVE"
                    print(f"    {task:15s}: {prob:.4f} ({status})")
            print()


if __name__ == "__main__":
    exit(main())