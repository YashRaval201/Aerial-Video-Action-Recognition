import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

from config import *
from models.cnn_lstm_model import CNNLSTMActionRecognizer
from utils.minimal_dataset import MinimalActionDataset

def test_model():
    """Test model on test dataset only"""
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = MinimalActionDataset(
        TEST_DIR, 
        sequence_length=SEQUENCE_LENGTH, 
        img_size=CNN_IMG_SIZE
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load existing model with deep FC layers
    model = CNNLSTMActionRecognizer(
        num_classes=len(ACTION_CLASSES),
        sequence_length=SEQUENCE_LENGTH,
        hidden_size=512,
        num_lstm_layers=2,
        dropout=0.4,
        use_attention=True
    ).to(device)
    
    if not os.path.exists(CNN_LSTM_MODEL_PATH):
        print(f"❌ Model not found: {CNN_LSTM_MODEL_PATH}")
        print("Please train the model first using: python main.py --mode train_action")
        return
    
    model.load_state_dict(torch.load(CNN_LSTM_MODEL_PATH, map_location=device))
    print(f"✅ Model loaded successfully from {CNN_LSTM_MODEL_PATH}")
    model.eval()
    
    # Test evaluation
    print("Running test evaluation...")
    all_predictions = []
    all_targets = []
    all_action_names = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            frames = batch['frames'].to(device)
            targets = batch['action'].to(device)
            action_names = batch['action_name']
            
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_action_names.extend(action_names)
    
    # Calculate accuracy
    test_accuracy = 100. * correct / total
    
    # Print results
    print("\n" + "="*60)
    print("TEST DATASET RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Correct Predictions: {correct}/{total}")
    
    # Detailed classification report
    print("\nPer-Class Performance:")
    report = classification_report(
        all_targets, all_predictions,
        target_names=ACTION_CLASSES,
        output_dict=True,
        zero_division=0
    )
    
    for i, action in enumerate(ACTION_CLASSES):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            print(f"{action:12}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.3f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.3f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    print(f"\nConfusion Matrix:")
    print("Predicted →")
    print("Actual ↓")
    
    # Print confusion matrix with action names
    header = "".join([f"{ACTION_CLASSES[i][:3]:>4}" for i in range(len(ACTION_CLASSES))])
    print(f"     {header}")
    
    for i, action in enumerate(ACTION_CLASSES):
        row = "".join([f"{cm[i][j]:4d}" for j in range(len(ACTION_CLASSES))])
        print(f"{action[:3]:>3}: {row}")
    
    return test_accuracy, report

if __name__ == "__main__":
    try:
        test_accuracy, report = test_model()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()