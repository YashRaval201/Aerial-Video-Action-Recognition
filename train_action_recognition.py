import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import os
import json
from collections import defaultdict

from config import *
from models.final_architecture import FinalCNNLSTM
from utils.minimal_dataset import MinimalActionDataset

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ActionRecognitionTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Simple cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Differential learning rates
        backbone_params = list(self.model.cnn.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
            {'params': other_params, 'lr': LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
        
        # Enhanced scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=2, eta_min=1e-7
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=PATIENCE)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def compute_loss(self, outputs, targets, epoch):
        """Enhanced loss with label smoothing"""
        # Label smoothing
        smoothed_targets = targets * 0.9 + 0.1 / len(ACTION_CLASSES)
        return F.cross_entropy(outputs, targets, label_smoothing=0.1)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            targets = batch['action'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(frames)
            loss = self.compute_loss(outputs, targets, epoch)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                frames = batch['frames'].to(self.device)
                targets = batch['action'].to(self.device)
                
                outputs = self.model(frames)
                loss = self.ce_loss(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self):
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0
        
        for epoch in range(NUM_EPOCHS):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch()
            
            # Scheduler step
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), CNN_LSTM_MODEL_PATH)
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load(CNN_LSTM_MODEL_PATH))
        
        return best_val_acc
    
    def evaluate(self, data_loader, split_name="Test"):
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_action_names = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'{split_name} Evaluation'):
                frames = batch['frames'].to(self.device)
                targets = batch['action'].to(self.device)
                action_names = batch['action_name']
                
                outputs = self.model(frames)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_action_names.extend(action_names)
        
        # Calculate metrics
        accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_predictions)
        
        # Get unique classes present in the data
        unique_classes = sorted(list(set(all_targets + all_predictions)))
        present_class_names = [ACTION_CLASSES[i] for i in unique_classes]
        
        # Classification report
        report = classification_report(
            all_targets, all_predictions,
            labels=unique_classes,
            target_names=present_class_names,
            output_dict=True,
            zero_division=0
        )
        
        print(f'\n{split_name} Results:')
        print(f'Overall Accuracy: {accuracy:.2f}%')
        print('\nPer-class Performance:')
        
        for i, class_idx in enumerate(unique_classes):
            action = ACTION_CLASSES[class_idx]
            if str(i) in report:
                precision = report[str(i)]['precision']
                recall = report[str(i)]['recall']
                f1 = report[str(i)]['f1-score']
                print(f'{action:12}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=unique_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=present_class_names, yticklabels=present_class_names)
        plt.title(f'{split_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/{split_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy, report
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CNN_LSTM_MODEL_PATH), exist_ok=True)
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MinimalActionDataset(
        TRAIN_DIR, sequence_length=SEQUENCE_LENGTH, img_size=CNN_IMG_SIZE
    )
    
    val_dataset = MinimalActionDataset(
        VAL_DIR, sequence_length=SEQUENCE_LENGTH, img_size=CNN_IMG_SIZE
    )
    
    test_dataset = MinimalActionDataset(
        TEST_DIR, sequence_length=SEQUENCE_LENGTH, img_size=CNN_IMG_SIZE
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model with deep FC layers
    model = FinalCNNLSTM(
        num_classes=len(ACTION_CLASSES),
        sequence_length=SEQUENCE_LENGTH,
        hidden_size=HIDDEN_SIZE
    )
    
    # Create trainer
    trainer = ActionRecognitionTrainer(model, train_loader, val_loader, test_loader, device)
    
    # Train model
    print("Starting training...")
    best_val_acc = trainer.train()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    val_acc, val_report = trainer.evaluate(val_loader, "Validation")
    test_acc, test_report = trainer.evaluate(test_loader, "Test")
    
    # Calculate F1 score
    val_f1 = val_report['weighted avg']['f1-score'] * 100
    test_f1 = test_report['weighted avg']['f1-score'] * 100
    
    # Save results
    results = {
        'train_acc': trainer.train_accuracies[-1] if trainer.train_accuracies else 0,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'f1_score': test_f1,
        'best_validation_accuracy': best_val_acc,
        'validation_report': val_report,
        'test_report': test_report,
        'model_config': {
            'num_classes': len(ACTION_CLASSES),
            'sequence_length': SEQUENCE_LENGTH,
            'hidden_size': HIDDEN_SIZE,
            'num_lstm_layers': 3,
            'dropout': 0.2,
            'use_attention': True
        }
    }
    
    with open(f'{RESULTS_DIR}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"train_acc[-1] = {trainer.train_accuracies[-1]:.2f}")
    print(f"val_acc[-1] = {val_acc:.2f}")
    print(f"test_acc[-1] = {test_acc:.2f}")
    print(f"f1_score[-1] = {test_f1:.1f}")
    print(f"Results saved in: {RESULTS_DIR}/")

if __name__ == "__main__":
    main()