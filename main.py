#!/usr/bin/env python3
"""
YOLO11-CNN-LSTM Multi-Person Action Recognition
Main execution script for training and evaluation pipeline
"""

import argparse
import os
import sys
import torch
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_yolo():
    """Train YOLO11 person detection model"""
    print("\n" + "="*60)
    print("TRAINING YOLO11 PERSON DETECTION MODEL")
    print("="*60)
    
    from train_yolo import convert_annotations_to_yolo_format, train_yolo_model, validate_yolo_model
    
    # First convert annotations and extract frames
    print("Converting video data to YOLO format...")
    convert_annotations_to_yolo_format()
    
    # Train YOLO model
    train_results = train_yolo_model()
    print("YOLO training completed!")
    
    # Validate YOLO model
    val_results = validate_yolo_model()
    print("YOLO validation completed!")
    
    return train_results, val_results

def train_action_recognition():
    """Train CNN-LSTM action recognition model"""
    print("\n" + "="*60)
    print("TRAINING CNN-LSTM ACTION RECOGNITION MODEL")
    print("="*60)
    
    from train_action_recognition import main as train_main
    train_main()

def evaluate_model():
    """Evaluate the trained model"""
    print("\n" + "="*60)
    print("EVALUATING TRAINED MODEL")
    print("="*60)
    
    from evaluate import main as eval_main
    eval_main()

def run_inference(video_path, output_path=None, show_video=False):
    """Run inference on a video"""
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    from inference import MultiPersonActionRecognizer
    from config import YOLO_MODEL_PATH, CNN_LSTM_MODEL_PATH
    
    # Initialize recognizer
    recognizer = MultiPersonActionRecognizer(
        yolo_model_path=YOLO_MODEL_PATH,
        cnn_lstm_model_path=CNN_LSTM_MODEL_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process video
    recognizer.process_video(
        video_path=video_path,
        output_path=output_path,
        show_video=show_video
    )

def check_data_structure():
    """Check if data structure is correct"""
    print("Checking data structure...")
    
    required_dirs = ['data/Train', 'data/Validation', 'data/Test']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        
        # Check for video files and annotations
        video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        if not video_files:
            print(f"❌ No video files found in {dir_path}")
            return False
        
        # Check for corresponding annotation directories
        for video_file in video_files:
            video_name = video_file.replace('.mp4.mp4', '.mp4').replace('.mp4', '')
            annotation_dir = os.path.join(dir_path, f"{video_name}_annotations")
            if not os.path.exists(annotation_dir):
                print(f"❌ Missing annotation directory: {annotation_dir}")
                return False
    
    print("✅ Data structure is correct!")
    return True

def main():
    parser = argparse.ArgumentParser(description='YOLO11-CNN-LSTM Multi-Person Action Recognition')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['install', 'train_yolo', 'train_action', 'train_all', 'evaluate', 'test_only', 'inference'],
                       help='Mode to run')
    parser.add_argument('--video', type=str, help='Video path for inference')
    parser.add_argument('--output', type=str, help='Output video path for inference')
    parser.add_argument('--show', action='store_true', help='Show video during inference')
    parser.add_argument('--check_data', action='store_true', help='Check data structure')
    
    args = parser.parse_args()
    
    # Check data structure if requested
    if args.check_data:
        if not check_data_structure():
            print("Please fix data structure issues before proceeding.")
            return
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        if args.mode == 'install':
            install_requirements()
            print("✅ Installation completed!")
            
        elif args.mode == 'train_yolo':
            if not check_data_structure():
                print("❌ Data structure check failed. Please fix issues before training.")
                return
            train_yolo()
            
        elif args.mode == 'train_action':
            if not check_data_structure():
                print("❌ Data structure check failed. Please fix issues before training.")
                return
            train_action_recognition()
            
        elif args.mode == 'train_all':
            if not check_data_structure():
                print("❌ Data structure check failed. Please fix issues before training.")
                return
            
            print("🚀 Starting complete training pipeline...")
            
            # Step 1: Train YOLO
            train_yolo()
            
            # Step 2: Train Action Recognition
            train_action_recognition()
            
            # Step 3: Evaluate
            evaluate_model()
            
            print("\n🎉 Complete training pipeline finished!")
            print("Check the 'results' directory for evaluation metrics and visualizations.")
            
        elif args.mode == 'evaluate':
            if not os.path.exists('models/cnn_lstm_action_recognition.pth'):
                print("❌ Trained action recognition model not found. Please train first.")
                return
            evaluate_model()
            
        elif args.mode == 'test_only':
            if not os.path.exists('models/cnn_lstm_action_recognition.pth'):
                print("❌ Trained action recognition model not found. Please train first.")
                return
            from test_only import test_model
            test_model()
            
        elif args.mode == 'inference':
            if not args.video:
                print("❌ Please provide video path with --video argument")
                return
            
            if not os.path.exists(args.video):
                print(f"❌ Video file not found: {args.video}")
                return
            
            if not os.path.exists('models/cnn_lstm_action_recognition.pth'):
                print("❌ Trained action recognition model not found. Please train first.")
                return
            
            run_inference(args.video, args.output, args.show)
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    print("1. Install requirements:")
    print("   python main.py --mode install")
    print()
    print("2. Check data structure:")
    print("   python main.py --mode train_all --check_data")
    print()
    print("3. Train complete pipeline:")
    print("   python main.py --mode train_all")
    print()
    print("4. Train only YOLO:")
    print("   python main.py --mode train_yolo")
    print()
    print("5. Train only action recognition:")
    print("   python main.py --mode train_action")
    print()
    print("6. Evaluate model:")
    print("   python main.py --mode evaluate")
    print()
    print("7. Run inference:")
    print("   python main.py --mode inference --video path/to/video.mp4 --output output.mp4")
    print()
    print("8. Run inference with live display:")
    print("   python main.py --mode inference --video path/to/video.mp4 --show")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage_examples()
    else:
        main()