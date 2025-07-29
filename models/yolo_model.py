import torch
import torch.nn as nn
from ultralytics import YOLO
import yaml

class YOLOPersonDetector:
    def __init__(self, model_size='n', pretrained=True):
        """
        Initialize YOLO11 model for person detection
        Args:
            model_size: 'n', 's', 'm', 'l', 'x' for nano, small, medium, large, extra-large
            pretrained: Whether to use pretrained weights
        """
        self.model_size = model_size
        self.model = YOLO(f'yolo11{model_size}.pt' if pretrained else f'yolo11{model_size}.yaml')
        
    def create_config(self, save_path='yolo11_person.yaml'):
        """Create YOLO configuration for person detection"""
        config = {
            'path': '../data',
            'train': 'Train',
            'val': 'Validation',
            'test': 'Test',
            'nc': 1,  # number of classes (person only)
            'names': ['person']
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
        
        return save_path
    
    def train(self, data_config, epochs=100, imgsz=640, batch=16, lr0=0.01, 
              weight_decay=0.0005, momentum=0.937, patience=50):
        """Train YOLO model"""
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            weight_decay=weight_decay,
            momentum=momentum,
            patience=patience,
            save=True,
            plots=True,
            val=True
        )
        return results
    
    def predict(self, source, conf=0.25, iou=0.7, save=False):
        """Run inference"""
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            show_labels=True,
            show_conf=True
        )
        return results
    
    def validate(self, data_config):
        """Validate model"""
        results = self.model.val(data=data_config)
        return results
    
    def export(self, format='onnx'):
        """Export model"""
        self.model.export(format=format)
    
    def save(self, path):
        """Save model"""
        self.model.save(path)
    
    def load(self, path):
        """Load model"""
        self.model = YOLO(path)

def create_yolo_dataset_yaml():
    """Create YAML configuration for YOLO training"""
    config = {
        'path': '/media/cvbl-ag/Data/Mtech/Yash_Raval_Thesis/Human_Action_Recognition/AERIAL_VIDEO_ACTION_RECOGNITION/data',
        'train': 'Train',
        'val': 'Validation', 
        'test': 'Test',
        'nc': 1,
        'names': ['person']
    }
    
    yaml_path = 'yolo_person_config.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    return yaml_path