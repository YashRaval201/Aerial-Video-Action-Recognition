import torch
import cv2
import numpy as np
from collections import defaultdict, deque
import argparse
import os

from config import *
from models.yolo_model import YOLOPersonDetector
from models.cnn_lstm_model import CNNLSTMActionRecognizer
from utils.data_utils import VideoProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultiPersonActionRecognizer:
    def __init__(self, yolo_model_path, cnn_lstm_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model for person detection
        self.yolo_model = YOLOPersonDetector()
        if os.path.exists(yolo_model_path):
            self.yolo_model.load(yolo_model_path)
        else:
            print(f"YOLO model not found at {yolo_model_path}, using pretrained YOLO11")
            self.yolo_model = YOLOPersonDetector(pretrained=True)
        
        # Load CNN-LSTM model for action recognition
        self.action_model = CNNLSTMActionRecognizer(
            num_classes=len(ACTION_CLASSES),
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=512,
            num_lstm_layers=2,
            dropout=0.4,
            use_attention=True
        ).to(self.device)
        
        if os.path.exists(cnn_lstm_model_path):
            self.action_model.load_state_dict(torch.load(cnn_lstm_model_path, map_location=self.device))
            self.action_model.eval()
        else:
            print(f"Action recognition model not found at {cnn_lstm_model_path}")
        
        # Transform for action recognition
        self.transform = A.Compose([
            A.Resize(CNN_IMG_SIZE, CNN_IMG_SIZE),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Person tracking
        self.person_tracks = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
        self.person_actions = defaultdict(str)
        self.track_id_counter = 0
        
    def detect_persons(self, frame, conf_threshold=0.5):
        """Detect persons in frame using YOLO"""
        results = self.yolo_model.predict(frame, conf=conf_threshold, save=False)
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    if box.cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf)
                        })
        
        return detections   # ← Returns list of ALL detected persons
    
    def simple_tracker(self, detections, frame_idx, iou_threshold=0.3):
        """Simple tracking based on IoU overlap"""
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        tracked_detections = []
        
        # Get last frame detections for tracking
        last_frame_detections = []
        for track_id, track in self.person_tracks.items():
            if len(track) > 0:
                last_frame_detections.append((track_id, track[-1]['bbox']))
                # Result: [(ID:0, bbox_A), (ID:1, bbox_B), (ID:2, bbox_C)]
        
        # Match current detections with existing tracks
        used_track_ids = set()
        for detection in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, last_bbox in last_frame_detections:
                if track_id not in used_track_ids:
                    iou = calculate_iou(detection['bbox'], last_bbox)
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                track_id = best_track_id
                used_track_ids.add(track_id)
            else:
                track_id = self.track_id_counter
                self.track_id_counter += 1
            
            tracked_detections.append({
                'track_id': track_id,
                'bbox': detection['bbox'],
                'confidence': detection['confidence']
            })
        
        return tracked_detections
    
    def crop_person(self, frame, bbox, padding=0.1):
        """Crop person from frame"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        width = x2 - x1
        height = y2 - y1
        pad_w = int(width * padding)
        pad_h = int(height * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return frame[y1:y2, x1:x2]
    
    def recognize_action(self, person_crops):
        """Recognize action from sequence of person crops"""
        if len(person_crops) < SEQUENCE_LENGTH:
            return "Unknown", 0.0
        
        # Prepare sequence
        frames = []
        for crop in person_crops[-SEQUENCE_LENGTH:]:    # ← Last 16 frames of THIS person
            if crop is not None and crop.size > 0:
                transformed = self.transform(image=crop)    # ← Resize to 224x224
                frames.append(transformed['image'])
            else:
                # Dummy frame
                dummy = np.zeros((CNN_IMG_SIZE, CNN_IMG_SIZE, 3), dtype=np.uint8)
                transformed = self.transform(image=dummy)
                frames.append(transformed['image'])
        
        # Stack frames
        frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device)
        # ↑ Shape: (1, 16, 3, 224, 224) - One person's sequence
    
        # Predict
        with torch.no_grad():
            outputs = self.action_model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            action_idx = predicted.item()
            confidence_score = confidence.item()
            action_name = IDX_TO_ACTION.get(action_idx, "Unknown")
        
        return action_name, confidence_score
    
    def process_video(self, video_path, output_path=None, show_video=False):
        """Process video for multi-person action recognition"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect persons
            detections = self.detect_persons(frame) # ← Multiple persons
            
            # Track persons
            tracked_detections = self.simple_tracker(detections, frame_idx)
            
            # Process each tracked person
            for detection in tracked_detections:
                track_id = detection['track_id']    # ← Person's unique ID
                bbox = detection['bbox']
                
                # Crop person
                person_crop = self.crop_person(frame, bbox)
                
                # Add to track
                self.person_tracks[track_id].append({   # ← Person-specific sequence
                    'frame_idx': frame_idx,
                    'bbox': bbox,
                    'crop': person_crop
                })
                
                # Recognize action if we have enough frames
                if len(self.person_tracks[track_id]) >= SEQUENCE_LENGTH:
                    crops = [item['crop'] for item in self.person_tracks[track_id]]
                    action, confidence = self.recognize_action(crops)
                    self.person_actions[track_id] = f"{action} ({confidence:.2f})"
                else:
                    self.person_actions[track_id] = "Collecting..."
                
                # Draw bounding box and action
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw action label
                label = f"ID:{track_id} {self.person_actions[track_id]}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Show frame
            if show_video:
                cv2.imshow('Multi-Person Action Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_video:
            cv2.destroyAllWindows()
        
        print(f"Video processing completed!")
        if output_path:
            print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Multi-Person Action Recognition')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    parser.add_argument('--yolo_model', type=str, default=YOLO_MODEL_PATH, help='YOLO model path')
    parser.add_argument('--action_model', type=str, default=CNN_LSTM_MODEL_PATH, help='Action recognition model path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = MultiPersonActionRecognizer(
        yolo_model_path=args.yolo_model,
        cnn_lstm_model_path=args.action_model,
        device=args.device
    )
    
    # Process video
    recognizer.process_video(
        video_path=args.video,
        output_path=args.output,
        show_video=args.show
    )

if __name__ == "__main__":
    main()