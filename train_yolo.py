import os
import torch
from ultralytics import YOLO
import yaml
from utils.data_utils import AnnotationParser, VideoProcessor
import cv2
import numpy as np
from tqdm import tqdm

def convert_annotations_to_yolo_format():
    """Convert custom annotations to YOLO format"""
    
    def process_split(split_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
        for video_file in tqdm(os.listdir(split_dir), desc=f"Processing {split_dir}"):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(split_dir, video_file)
                # Handle .mp4.mp4 files - remove one .mp4 for annotation directory
                video_name = video_file.replace('.mp4.mp4', '.mp4').replace('.mp4', '')
                annotations_dir = os.path.join(split_dir, f"{video_name}_annotations")
                
                if os.path.exists(annotations_dir):
                    print(f"Processing {video_file}...")
                    annotations = AnnotationParser.get_video_annotations(video_path, annotations_dir)
                    
                    # Process frames one by one to save memory
                    saved_frames = VideoProcessor.process_video_frames(
                        video_path, annotations, 
                        os.path.join(output_dir, 'images'),
                        os.path.join(output_dir, 'labels'),
                        video_name
                    )
                    
                    print(f"Saved {saved_frames} frames from {video_file}")
    
    # Process all splits
    print("Converting video annotations to YOLO format...")
    process_split('data/Train', 'yolo_data/train')
    process_split('data/Validation', 'yolo_data/val')
    process_split('data/Test', 'yolo_data/test')
    print("Frame extraction and annotation conversion completed!")

def create_yolo_config():
    """Create YOLO dataset configuration"""
    config = {
        'path': os.path.abspath('yolo_data'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['person']
    }
    
    with open('yolo_person_dataset.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return 'yolo_person_dataset.yaml'

def train_yolo_model():
    """Train YOLO11 model for person detection"""
    
    # Check if YOLO data exists
    if not os.path.exists('yolo_data/train/images') or len(os.listdir('yolo_data/train/images')) == 0:
        print("❌ No training images found. Frame extraction may have failed.")
        return None
    
    # Create config file
    config_path = create_yolo_config()
    
    # Initialize YOLO model
    model = YOLO('yolo11n.pt')  # Start with nano model for faster training
    
    # Train the model
    results = model.train(
        data=config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        patience=20,
        save=True,
        plots=True,
        val=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=4,
        project='runs/detect',
        name='yolo11_person'
    )
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    model.save('models/yolo11_person_detection.pt')
    
    return results

def validate_yolo_model():
    """Validate trained YOLO model"""
    if not os.path.exists('models/yolo11_person_detection.pt'):
        print("❌ No trained YOLO model found. Skipping validation.")
        return None
    
    model = YOLO('models/yolo11_person_detection.pt')
    
    # Validate on test set
    results = model.val(
        data='yolo_person_dataset.yaml',
        split='test',
        save_json=True,
        plots=True
    )
    
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    return results

if __name__ == "__main__":
    print("Starting YOLO11 training for person detection...")
    
    # Train model
    train_results = train_yolo_model()
    print("Training completed!")
    
    # Validate model
    print("Validating model...")
    val_results = validate_yolo_model()
    print("Validation completed!")
    
    print(f"Best model saved at: models/yolo11_person_detection.pt")