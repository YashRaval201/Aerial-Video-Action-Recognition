import os
import torch

# Dataset Configuration
DATA_ROOT = "data"
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
VAL_DIR = os.path.join(DATA_ROOT, "Validation")
TEST_DIR = os.path.join(DATA_ROOT, "Test")

# Action Classes
ACTION_CLASSES = [
    'Carrying', 'Drinking', 'Handshaking', 'Hugging', 'Kicking',
    'Lying', 'Punching', 'Reading', 'Running', 'Sitting',
    'Standing', 'Walking', 'Waving'
]

ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTION_CLASSES)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}

# Model Configuration
YOLO_IMG_SIZE = 640
CNN_IMG_SIZE = 224
SEQUENCE_LENGTH = 16  # Longer sequences
OVERLAP_RATIO = 0.3  # Less overlap

# Training Configuration - Optimized for target performance
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 80
PATIENCE = 25
HIDDEN_SIZE = 768

# Model Paths
YOLO_MODEL_PATH = "models/yolo11_person_detection.pt"
CNN_LSTM_MODEL_PATH = "models/cnn_lstm_action_recognition.pth"
RESULTS_DIR = "results"

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Augmentation
AUGMENTATION_PROB = 0.5