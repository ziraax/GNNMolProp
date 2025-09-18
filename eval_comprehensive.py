"""Comprehensive evaluation script with detailed analysis and visualizations."""
import argparse
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Add molprop to path
sys.path.insert(0, str(Path(__file__).parent))

from molprop.inference import MolecularPredictor
from molprop import setup_logging, DataModule, get_default_config


class ComprehensiveEvaluator:
    """Comprehensive model evaluation with detailed metrics and visualizations."""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "evaluation_results"):
        """Initialize evaluator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            output_dir: Directory to save evaluation results
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model and data
        self.predictor = MolecularPredictor(checkpoint_path)
        self.task_names = self.predictor.get_task_names()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_test_data(self):
        """Load test dataset for evaluation."""
        # Create data module to get test data
        config = get_default_config()
        data_module = DataModule(config.data)
        data_module.prepare_data()
        
        # Get test loader
        _, _, test_loader = data_module.get_dataloaders()
        
        # Extract test data
        test_smiles = []
        test_labels = []
        test_masks = []
        
        for batch in test_loader:
            # Handle batch processing properly
            batch_size = batch.num_graphs
            
            for i in range(batch_size):
                test_smiles.append(batch.smiles[i])
                
                # Extract labels and masks for this sample
                if batch.y.dim() == 1:
                    # If y is flattened, reshape it
                    y_reshaped = batch.y.view(batch_size, -1)
                    y_mask_reshaped = batch.y_mask.view(batch_size, -1)
                else:
                    y_reshaped = batch.y
                    y_mask_reshaped = batch.y_mask
                
                test_labels.append(y_reshaped[i].numpy())
                test_masks.append(y_mask_reshaped[i].numpy())
        
        test_labels = np.array(test_labels)
        test_masks = np.array(test_masks)
        
        print(f"Loaded {len(test_smiles)} test samples")
        print(f"Label shape: {test_labels.shape}, Mask shape: {test_masks.shape}")
        
        return test_smiles, test_labels, test_masks
    
    def evaluate_model(self, save_predictions=True) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        print("Loading test data...")
        test_smiles, test_labels, test_masks = self.load_test_data()
        
        print(f"Evaluating {len(test_smiles)} test molecules...")
        
        # Get predictions
        predictions = self.predictor.predict_batch(test_smiles)
        
        # Convert to arrays
        pred_probs = np.array([[pred[task] for task in self.task_names] for pred in predictions])
        
        # Ensure shapes are correct
        print(f"Initial shapes: labels={test_labels.shape}, predictions={pred_probs.shape}, masks={test_masks.shape}")
        
        # Labels and masks should be (n_samples, n_tasks)
        if test_labels.ndim == 1:
            # If somehow still flat, this is an error in our data loading
            raise ValueError(f"Labels still flat after processing: {test_labels.shape}")
        
        if test_masks.ndim == 1:
            # If somehow still flat, this is an error in our data loading  
            raise ValueError(f"Masks still flat after processing: {test_masks.shape}")
        
        print(f"Final shapes: labels={test_labels.shape}, predictions={pred_probs.shape}, masks={test_masks.shape}")
        
        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(
            test_labels, pred_probs, test_masks, test_smiles
        )
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(test_smiles, test_labels, pred_probs, test_masks)
        
        # Generate visualizations
        self._create_visualizations(test_labels, pred_probs, test_masks, results)
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, masks, smiles) -> Dict[str, Any]:
        """Calculate comprehensive metrics for each task and overall."""
        results = {
            'overall': {},
            'per_task': {},
            'confusion_matrices': {},
            'roc_data': {},
            'pr_data': {}
        }
        
        all_aucs = []
        all_aps = []  # Average Precision
        overall_metrics = []
        
        for i, task in enumerate(self.task_names):
            task_mask = masks[:, i].astype(bool)
            
            if not task_mask.any():
                print(f"Warning: No valid samples for task {task}")
                continue
            
            y_task = y_true[task_mask, i]
            pred_task = y_pred[task_mask, i]
            
            # Skip if only one class present
            if len(np.unique(y_task)) < 2:
                print(f"Warning: Only one class present for task {task}")
                continue
            
            # Calculate metrics
            task_results = self._calculate_task_metrics(y_task, pred_task, task)
            results['per_task'][task] = task_results
            
            # Collect for overall metrics
            all_aucs.append(task_results['auc'])
            all_aps.append(task_results['average_precision'])
            
            # Store confusion matrix
            pred_binary = (pred_task > 0.5).astype(int)
            cm = confusion_matrix(y_task, pred_binary)
            results['confusion_matrices'][task] = cm
            
            # Store ROC and PR curve data
            fpr, tpr, _ = roc_curve(y_task, pred_task)
            precision, recall, _ = precision_recall_curve(y_task, pred_task)
            
            results['roc_data'][task] = {'fpr': fpr, 'tpr': tpr}
            results['pr_data'][task] = {'precision': precision, 'recall': recall}
        
        # Overall metrics
        results['overall'] = {
            'mean_auc': np.mean(all_aucs),
            'std_auc': np.std(all_aucs),
            'mean_ap': np.mean(all_aps),
            'std_ap': np.std(all_aps),
            'num_tasks': len(all_aucs),
            'total_samples': len(smiles),
            'tasks_evaluated': list(results['per_task'].keys())
        }
        
        return results
    
    def _calculate_task_metrics(self, y_true, y_pred, task_name) -> Dict[str, float]:
        """Calculate detailed metrics for a single task."""
        # Continuous metrics
        auc = roc_auc_score(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        
        # Binary predictions (threshold = 0.5)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Class distribution
        pos_samples = np.sum(y_true)
        neg_samples = len(y_true) - pos_samples
        class_balance = pos_samples / len(y_true)
        
        return {
            'auc': auc,
            'average_precision': avg_precision,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'positive_samples': int(pos_samples),
            'negative_samples': int(neg_samples),
            'total_samples': len(y_true),
            'class_balance': class_balance
        }
    
    def _save_predictions(self, smiles, y_true, y_pred, masks):
        """Save detailed predictions to CSV."""
        data = []
        
        for i, smi in enumerate(smiles):
            row = {'smiles': smi}
            
            for j, task in enumerate(self.task_names):
                if masks[i, j]:
                    row[f'{task}_true'] = int(y_true[i, j])
                    row[f'{task}_pred'] = y_pred[i, j]
                    row[f'{task}_pred_binary'] = int(y_pred[i, j] > 0.5)
                else:
                    row[f'{task}_true'] = np.nan
                    row[f'{task}_pred'] = np.nan
                    row[f'{task}_pred_binary'] = np.nan
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_dir / 'detailed_predictions.csv', index=False)
        print(f"Saved detailed predictions to {self.output_dir / 'detailed_predictions.csv'}")
    
    def _create_visualizations(self, y_true, y_pred, masks, results):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # 1. ROC Curves
        self._plot_roc_curves(results['roc_data'])
        
        # 2. Precision-Recall Curves  
        self._plot_pr_curves(results['pr_data'])
        
        # 3. Performance Overview
        self._plot_performance_overview(results['per_task'])
        
        # 4. Confusion Matrices
        self._plot_confusion_matrices(results['confusion_matrices'])
        
        # 5. Prediction Distributions
        self._plot_prediction_distributions(y_true, y_pred, masks)
        
        # 6. Class Balance Analysis
        self._plot_class_balance(results['per_task'])
        
        # 7. Correlation Matrix
        self._plot_task_correlations(y_pred, masks)
    
    def _plot_roc_curves(self, roc_data):
        """Plot ROC curves for all tasks."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (task, data) in enumerate(roc_data.items()):
            if i >= len(axes):
                break
                
            axes[i].plot(data['fpr'], data['tpr'], linewidth=2, 
                        label=f'AUC = {roc_auc_score([0, 1], [data["fpr"][0], data["tpr"][-1]]):.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'ROC Curve - {task}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(roc_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, pr_data):
        """Plot Precision-Recall curves for all tasks."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (task, data) in enumerate(pr_data.items()):
            if i >= len(axes):
                break
                
            axes[i].plot(data['recall'], data['precision'], linewidth=2)
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f'Precision-Recall - {task}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(pr_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_overview(self, per_task_results):
        """Plot performance overview with multiple metrics."""
        tasks = list(per_task_results.keys())
        metrics = ['auc', 'average_precision', 'accuracy', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [per_task_results[task][metric] for task in tasks]
            
            bars = axes[i].bar(range(len(tasks)), values, alpha=0.7)
            axes[i].set_xlabel('Tasks')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Task')
            axes[i].set_xticks(range(len(tasks)))
            axes[i].set_xticklabels(tasks, rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, confusion_matrices):
        """Plot confusion matrices for all tasks."""
        n_tasks = len(confusion_matrices)
        cols = 4
        rows = (n_tasks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i, (task, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {task}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(confusion_matrices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distributions(self, y_true, y_pred, masks):
        """Plot prediction score distributions."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, task in enumerate(self.task_names):
            if i >= len(axes):
                break
                
            task_mask = masks[:, i].astype(bool)
            if not task_mask.any():
                continue
            
            y_task = y_true[task_mask, i]
            pred_task = y_pred[task_mask, i]
            
            # Separate by true class
            pos_preds = pred_task[y_task == 1]
            neg_preds = pred_task[y_task == 0]
            
            axes[i].hist(neg_preds, bins=20, alpha=0.7, label='Negative', color='blue')
            axes[i].hist(pos_preds, bins=20, alpha=0.7, label='Positive', color='red')
            axes[i].axvline(0.5, color='black', linestyle='--', alpha=0.8, label='Threshold')
            axes[i].set_xlabel('Prediction Score')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'Prediction Distributions - {task}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.task_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_balance(self, per_task_results):
        """Plot class balance for each task."""
        tasks = list(per_task_results.keys())
        pos_counts = [per_task_results[task]['positive_samples'] for task in tasks]
        neg_counts = [per_task_results[task]['negative_samples'] for task in tasks]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(15, 8))
        bars1 = ax.bar(x - width/2, neg_counts, width, label='Negative', alpha=0.7)
        bars2 = ax.bar(x + width/2, pos_counts, width, label='Positive', alpha=0.7)
        
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Sample Count')
        ax.set_title('Class Distribution by Task')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_balance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_task_correlations(self, y_pred, masks):
        """Plot correlation matrix of prediction scores between tasks."""
        # Create DataFrame with valid predictions only
        valid_data = {}
        for i, task in enumerate(self.task_names):
            task_mask = masks[:, i].astype(bool)
            valid_preds = np.full(len(y_pred), np.nan)
            valid_preds[task_mask] = y_pred[task_mask, i]
            valid_data[task] = valid_preds
        
        df = pd.DataFrame(valid_data)
        corr_matrix = df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=ax)
        ax.set_title('Task Correlation Matrix (Prediction Scores)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'task_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, results):
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Model info
            model_info = self.predictor.get_model_info()
            f.write("MODEL INFORMATION:\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Tasks: {model_info['num_tasks']}\n")
            f.write(f"Hidden Dimensions: {model_info.get('hidden_dim', 'Unknown')}\n")
            f.write(f"Device: {model_info['device']}\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE:\n")
            overall = results['overall']
            f.write(f"Mean AUC: {overall['mean_auc']:.4f} ± {overall['std_auc']:.4f}\n")
            f.write(f"Mean Average Precision: {overall['mean_ap']:.4f} ± {overall['std_ap']:.4f}\n")
            f.write(f"Tasks Evaluated: {overall['num_tasks']}/{len(self.task_names)}\n")
            f.write(f"Total Test Samples: {overall['total_samples']}\n\n")
            
            # Per-task results
            f.write("PER-TASK PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Task':<15} {'AUC':<8} {'AP':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Pos':<6} {'Neg':<6}\n")
            f.write("-" * 80 + "\n")
            
            for task, metrics in results['per_task'].items():
                f.write(f"{task:<15} "
                       f"{metrics['auc']:<8.3f} "
                       f"{metrics['average_precision']:<8.3f} "
                       f"{metrics['accuracy']:<8.3f} "
                       f"{metrics['precision']:<8.3f} "
                       f"{metrics['recall']:<8.3f} "
                       f"{metrics['f1_score']:<8.3f} "
                       f"{metrics['positive_samples']:<6d} "
                       f"{metrics['negative_samples']:<6d}\n")
            
            f.write("\n\nLEGEND:\n")
            f.write("AUC: Area Under ROC Curve\n")
            f.write("AP: Average Precision\n")
            f.write("Acc: Accuracy\n")
            f.write("Prec: Precision\n")
            f.write("Rec: Recall\n")
            f.write("F1: F1 Score\n")
            f.write("Pos: Positive Samples\n")
            f.write("Neg: Negative Samples\n")
        
        print(f"Generated evaluation report: {report_path}")
        
        # Also save as JSON
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        print(f"Saved evaluation results: {json_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--no-predictions", action="store_true",
                       help="Skip saving detailed predictions")
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("Starting comprehensive evaluation...")
    evaluator = ComprehensiveEvaluator(args.checkpoint, args.output_dir)
    
    results = evaluator.evaluate_model(save_predictions=not args.no_predictions)
    
    print(f"\\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Mean AUC: {results['overall']['mean_auc']:.4f}")
    print(f"Tasks evaluated: {results['overall']['num_tasks']}")


if __name__ == "__main__":
    exit(main())