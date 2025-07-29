import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm

from config import *
from models.cnn_lstm_model import CNNLSTMActionRecognizer
from utils.data_utils import ActionRecognitionDataset

class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = CNNLSTMActionRecognizer(
            num_classes=len(ACTION_CLASSES),
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=512,
            num_lstm_layers=2,
            dropout=0.4,
            use_attention=True
        ).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    def evaluate_dataset(self, data_loader, dataset_name="Test"):
        """Comprehensive evaluation of the model"""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_action_names = []
        
        print(f"Evaluating on {dataset_name} set...")
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'{dataset_name} Evaluation'):
                frames = batch['frames'].to(self.device)
                targets = batch['action'].to(self.device)
                action_names = batch['action_name']
                
                outputs = self.model(frames)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_action_names.extend(action_names)
        
        return self._compute_metrics(all_predictions, all_targets, all_probabilities, dataset_name)
    
    def _compute_metrics(self, predictions, targets, probabilities, dataset_name):
        """Compute comprehensive metrics"""
        # Overall accuracy
        accuracy = 100. * sum(p == t for p, t in zip(predictions, targets)) / len(predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Classification report
        report = classification_report(
            targets, predictions,
            target_names=ACTION_CLASSES,
            output_dict=True,
            zero_division=0
        )
        
        # Create detailed results
        results = {
            'dataset_name': dataset_name,
            'overall_accuracy': accuracy,
            'macro_precision': macro_precision * 100,
            'macro_recall': macro_recall * 100,
            'macro_f1': macro_f1 * 100,
            'weighted_precision': weighted_precision * 100,
            'weighted_recall': weighted_recall * 100,
            'weighted_f1': weighted_f1 * 100,
            'per_class_metrics': {},
            'classification_report': report
        }
        
        # Per-class detailed metrics
        for i, action in enumerate(ACTION_CLASSES):
            results['per_class_metrics'][action] = {
                'precision': precision[i] * 100,
                'recall': recall[i] * 100,
                'f1_score': f1[i] * 100,
                'support': int(support[i])
            }
        
        # Print results
        self._print_results(results)
        
        # Generate visualizations
        self._generate_visualizations(targets, predictions, probabilities, dataset_name)
        
        return results
    
    def _print_results(self, results):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print(f"{results['dataset_name']} SET EVALUATION RESULTS")
        print(f"{'='*60}")
        
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"\nMacro Averages:")
        print(f"  Precision: {results['macro_precision']:.2f}%")
        print(f"  Recall: {results['macro_recall']:.2f}%")
        print(f"  F1-Score: {results['macro_f1']:.2f}%")
        
        print(f"\nWeighted Averages:")
        print(f"  Precision: {results['weighted_precision']:.2f}%")
        print(f"  Recall: {results['weighted_recall']:.2f}%")
        print(f"  F1-Score: {results['weighted_f1']:.2f}%")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Action':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for action, metrics in results['per_class_metrics'].items():
            print(f"{action:<12} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} "
                  f"{metrics['f1_score']:<10.2f} {metrics['support']:<10}")
        
        # Identify problematic classes
        print(f"\nClasses needing improvement (Recall < 70%):")
        low_recall_classes = []
        for action, metrics in results['per_class_metrics'].items():
            if metrics['recall'] < 70.0 and metrics['support'] > 0:
                low_recall_classes.append((action, metrics['recall']))
                print(f"  {action}: {metrics['recall']:.2f}% recall")
        
        if not low_recall_classes:
            print("  All classes have recall >= 70%")
    
    def _generate_visualizations(self, targets, predictions, probabilities, dataset_name):
        """Generate visualization plots"""
        # Confusion Matrix
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=ACTION_CLASSES, yticklabels=ACTION_CLASSES,
                   cbar_kws={'label': 'Count'})
        plt.title(f'{dataset_name} Set - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/{dataset_name.lower()}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=ACTION_CLASSES, yticklabels=ACTION_CLASSES,
                   cbar_kws={'label': 'Proportion'})
        plt.title(f'{dataset_name} Set - Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/{dataset_name.lower()}_confusion_matrix_normalized.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class Performance Bar Plot
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i in range(len(ACTION_CLASSES)):
            mask = np.array(targets) == i
            if mask.sum() > 0:
                pred_mask = np.array(predictions) == i
                tp = (mask & pred_mask).sum()
                fp = (~mask & pred_mask).sum()
                fn = (mask & ~pred_mask).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_scores.append(precision * 100)
                recall_scores.append(recall * 100)
                f1_scores.append(f1 * 100)
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
        
        x = np.arange(len(ACTION_CLASSES))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(16, 8))
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Action Classes', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title(f'{dataset_name} Set - Per-Class Performance', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_CLASSES, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/{dataset_name.lower()}_per_class_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_with_targets(self, target_metrics):
        """Compare current performance with target metrics"""
        print(f"\n{'='*60}")
        print("COMPARISON WITH TARGET PERFORMANCE")
        print(f"{'='*60}")
        
        target_overall_precision = target_metrics.get('overall_precision', 90.0)
        target_overall_recall = target_metrics.get('overall_recall', 85.0)
        target_min_recall = target_metrics.get('min_recall_per_class', 70.0)
        
        # Load latest results
        results_file = f'{RESULTS_DIR}/evaluation_results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                current_results = json.load(f)
            
            current_precision = current_results.get('weighted_precision', 0)
            current_recall = current_results.get('weighted_recall', 0)
            
            print(f"Target vs Current Performance:")
            print(f"  Overall Precision: {target_overall_precision:.1f}% (target) vs {current_precision:.1f}% (current)")
            print(f"  Overall Recall: {target_overall_recall:.1f}% (target) vs {current_recall:.1f}% (current)")
            
            # Check per-class recall
            min_recall_achieved = min([metrics['recall'] for metrics in current_results['per_class_metrics'].values()])
            print(f"  Min Class Recall: {target_min_recall:.1f}% (target) vs {min_recall_achieved:.1f}% (current)")
            
            # Status
            precision_met = current_precision >= target_overall_precision
            recall_met = current_recall >= target_overall_recall
            min_recall_met = min_recall_achieved >= target_min_recall
            
            print(f"\nTarget Achievement Status:")
            print(f"  Overall Precision: {'✓' if precision_met else '✗'}")
            print(f"  Overall Recall: {'✓' if recall_met else '✗'}")
            print(f"  Min Class Recall: {'✓' if min_recall_met else '✗'}")
            
            if precision_met and recall_met and min_recall_met:
                print(f"\n🎉 ALL TARGETS ACHIEVED! 🎉")
            else:
                print(f"\n⚠️  Some targets not yet achieved. Consider model improvements.")

def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(CNN_LSTM_MODEL_PATH, device=DEVICE)
    
    # Create test dataset
    test_dataset = ActionRecognitionDataset(
        TEST_DIR, sequence_length=SEQUENCE_LENGTH,
        overlap_ratio=OVERLAP_RATIO, img_size=CNN_IMG_SIZE, augment=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate model
    results = evaluator.evaluate_dataset(test_loader, "Test")
    
    # Save results
    with open(f'{RESULTS_DIR}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compare with targets
    target_metrics = {
        'overall_precision': 90.0,
        'overall_recall': 85.0,
        'min_recall_per_class': 70.0
    }
    
    evaluator.compare_with_targets(target_metrics)
    
    print(f"\nEvaluation completed! Results saved in {RESULTS_DIR}/")

if __name__ == "__main__":
    main()