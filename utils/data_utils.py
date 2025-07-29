import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AnnotationParser:
    @staticmethod
    def parse_annotation_file(file_path):
        """Parse annotation file and return list of annotations"""
        annotations = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 7:
                            try:
                                annotations.append({
                                    'person': parts[0],
                                    'action': parts[1],
                                    'person_id': int(parts[2]),
                                    'x_center': float(parts[3]),
                                    'y_center': float(parts[4]),
                                    'width': float(parts[5]),
                                    'height': float(parts[6])
                                })
                            except ValueError as e:
                                print(f"Error parsing line: {line} - {e}")
                                continue
        return annotations

    @staticmethod
    def get_video_annotations(video_path, annotations_dir):
        """Get all annotations for a video"""
        video_annotations = {}
        if os.path.exists(annotations_dir):
            for file_name in sorted(os.listdir(annotations_dir)):
                if file_name.endswith('.txt'):
                    frame_idx = int(file_name.split('.')[0])
                    file_path = os.path.join(annotations_dir, file_name)
                    video_annotations[frame_idx] = AnnotationParser.parse_annotation_file(file_path)
        return video_annotations

class VideoProcessor:
    @staticmethod
    def extract_frames(video_path, max_frames=100):
        """Extract minimal frames - ultra memory efficient"""
        cap = cv2.VideoCapture(video_path)
        frames = {}
        frame_idx = 0
        
        # Only extract every 10th frame to save memory
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:  # Every 10th frame only
                frames[frame_idx] = frame
            frame_idx += 1
        
        cap.release()
        return frames
    
    @staticmethod
    def process_video_frames(video_path, annotations, output_images_dir, output_labels_dir, video_name):
        """Process video frames one by one to save memory"""
        cap = cv2.VideoCapture(video_path)
        saved_frames = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process frames that have annotations
            if frame_idx in annotations and len(annotations[frame_idx]) > 0:
                # Save frame
                frame_name = f"{video_name}_{frame_idx:06d}.jpg"
                frame_path = os.path.join(output_images_dir, frame_name)
                
                success = cv2.imwrite(frame_path, frame)
                if success:
                    saved_frames += 1
                    # Save YOLO format labels
                    label_path = os.path.join(output_labels_dir, f"{video_name}_{frame_idx:06d}.txt")
                    h, w = frame.shape[:2]
                    
                    with open(label_path, 'w') as f:
                        for ann in annotations[frame_idx]:
                            # Convert to YOLO format (normalized)
                            x_center = ann['x_center'] / w
                            y_center = ann['y_center'] / h
                            width = ann['width'] / w
                            height = ann['height'] / h
                            
                            # Class 0 for person
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            frame_idx += 1
            
            # Limit processing to prevent excessive runtime
            if frame_idx > 2000:
                break
        
        cap.release()
        return saved_frames

    @staticmethod
    def crop_person(frame, annotation, padding=0.1):
        """Crop person from frame using bounding box"""
        h, w = frame.shape[:2]
        x_center, y_center = annotation['x_center'], annotation['y_center']
        width, height = annotation['width'], annotation['height']
        
        # Add padding
        width = int(width * (1 + padding))
        height = int(height * (1 + padding))
        
        # Calculate crop coordinates
        x1 = max(0, int(x_center - width // 2))
        y1 = max(0, int(y_center - height // 2))
        x2 = min(w, int(x_center + width // 2))
        y2 = min(h, int(y_center + height // 2))
        
        return frame[y1:y2, x1:x2]

class YOLODataset(Dataset):
    def __init__(self, data_dir, img_size=640, augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.samples = self._load_samples()
        
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def _load_samples(self):
        samples = []
        for video_file in os.listdir(self.data_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(self.data_dir, video_file)
                # Handle .mp4.mp4 files
                video_name = video_file.replace('.mp4.mp4', '.mp4').replace('.mp4', '')
                annotations_dir = os.path.join(self.data_dir, f"{video_name}_annotations")
                
                if os.path.exists(annotations_dir):
                    frames = VideoProcessor.extract_frames(video_path)
                    annotations = AnnotationParser.get_video_annotations(video_path, annotations_dir)
                    
                    for frame_idx in frames.keys():
                        if frame_idx in annotations:
                            frame = frames[frame_idx]
                            samples.append({
                                'frame': frame,
                                'annotations': annotations[frame_idx],
                                'video_name': video_file,
                                'frame_idx': frame_idx
                            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame = sample['frame']
        annotations = sample['annotations']
        
        h, w = frame.shape[:2]
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            # Convert to YOLO format (normalized)
            x_center = ann['x_center'] / w
            y_center = ann['y_center'] / h
            width = ann['width'] / w
            height = ann['height'] / h
            
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(0)  # Person class
        
        if len(bboxes) == 0:
            bboxes = [[0, 0, 0, 0]]
            class_labels = [0]
        
        transformed = self.transform(image=frame, bboxes=bboxes, class_labels=class_labels)
        
        return {
            'image': transformed['image'],
            'bboxes': torch.tensor(transformed['bboxes'], dtype=torch.float32),
            'labels': torch.tensor(class_labels, dtype=torch.long)
        }

class ActionRecognitionDataset(Dataset):
    def __init__(self, data_dir, sequence_length=16, overlap_ratio=0.5, img_size=224, augment=False):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size
        self.augment = augment
        self.sequences = self._create_sequences()
        
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(),
                ToTensorV2()
            ])

    def _create_sequences(self):
        sequences = []
        
        for video_file in os.listdir(self.data_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(self.data_dir, video_file)
                # Handle .mp4.mp4 files
                video_name = video_file.replace('.mp4.mp4', '.mp4').replace('.mp4', '')
                annotations_dir = os.path.join(self.data_dir, f"{video_name}_annotations")
                
                if os.path.exists(annotations_dir):
                    frames = VideoProcessor.extract_frames(video_path)
                    annotations = AnnotationParser.get_video_annotations(video_path, annotations_dir)
                    
                    # Group annotations by person_id
                    person_tracks = defaultdict(list)
                    for frame_idx, frame_annotations in annotations.items():
                        for ann in frame_annotations:
                            person_tracks[ann['person_id']].append({
                                'frame_idx': frame_idx,
                                'frame': frames.get(frame_idx),
                                'annotation': ann
                            })
                    
                    # Create sequences for each person
                    for person_id, track in person_tracks.items():
                        track.sort(key=lambda x: x['frame_idx'])
                        
                        step = int(self.sequence_length * (1 - self.overlap_ratio))
                        for start_idx in range(0, len(track) - self.sequence_length + 1, step):
                            sequence_data = track[start_idx:start_idx + self.sequence_length]
                            
                            # Get most frequent action in sequence
                            actions = [item['annotation']['action'] for item in sequence_data]
                            most_common_action = Counter(actions).most_common(1)[0][0]
                            
                            sequences.append({
                                'sequence': sequence_data,
                                'action': most_common_action,
                                'person_id': person_id,
                                'video_name': video_file
                            })
        
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        sequence = sequence_data['sequence']
        action = sequence_data['action']
        
        frames = []
        for item in sequence:
            if item['frame'] is not None:
                person_crop = VideoProcessor.crop_person(item['frame'], item['annotation'])
                if person_crop.size > 0:
                    transformed = self.transform(image=person_crop)
                    frames.append(transformed['image'])
                else:
                    # Create dummy frame if crop failed
                    dummy_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                    transformed = self.transform(image=dummy_frame)
                    frames.append(transformed['image'])
        
        # Pad sequence if needed
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros_like(frames[0]) if frames else torch.zeros(3, self.img_size, self.img_size))
        
        frames_tensor = torch.stack(frames[:self.sequence_length])
        
        from config import ACTION_TO_IDX
        action_idx = ACTION_TO_IDX.get(action, 0)
        
        return {
            'frames': frames_tensor,
            'action': torch.tensor(action_idx, dtype=torch.long),
            'action_name': action
        }