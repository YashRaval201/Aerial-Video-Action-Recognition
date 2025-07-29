import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import ACTION_TO_IDX

class MinimalActionDataset(Dataset):
    def __init__(self, data_dir, sequence_length=8, img_size=224):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Process ALL videos for better accuracy
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
        self.samples = self._create_samples()
    
    def _create_samples(self):
        samples = []
        
        for video_file in self.video_files:
            video_path = os.path.join(self.data_dir, video_file)
            video_name = video_file.replace('.mp4.mp4', '.mp4').replace('.mp4', '')
            annotations_dir = os.path.join(self.data_dir, f"{video_name}_annotations")
            
            if os.path.exists(annotations_dir):
                ann_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.txt')])[:100]
                
                for i in range(0, len(ann_files) - self.sequence_length, 4):
                    frame_indices = list(range(i, i + self.sequence_length))
                    samples.append({
                        'video_path': video_path,
                        'annotations_dir': annotations_dir,
                        'frame_indices': frame_indices
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        annotations_dir = sample['annotations_dir']
        frame_indices = sample['frame_indices']
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        actions = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                ann_file = os.path.join(annotations_dir, f"{frame_idx:04d}.txt")
                if os.path.exists(ann_file):
                    with open(ann_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split(',')
                            if len(parts) >= 7:
                                action = parts[1]
                                actions.append(action)
                                
                                h, w = frame.shape[:2]
                                x_center = int(float(parts[3]))
                                y_center = int(float(parts[4]))
                                crop_size = 150
                                
                                x1 = max(0, x_center - crop_size//2)
                                y1 = max(0, y_center - crop_size//2)
                                x2 = min(w, x_center + crop_size//2)
                                y2 = min(h, y_center + crop_size//2)
                                
                                person_crop = frame[y1:y2, x1:x2]
                                
                                if person_crop.size > 0:
                                    transformed = self.transform(image=person_crop)
                                    frames.append(transformed['image'])
                                else:
                                    dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                                    transformed = self.transform(image=dummy)
                                    frames.append(transformed['image'])
            
            if len(frames) == 0:
                dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                transformed = self.transform(image=dummy)
                frames.append(transformed['image'])
                actions.append('Standing')
        
        cap.release()
        
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else torch.zeros(3, self.img_size, self.img_size))
            actions.append(actions[-1] if actions else 'Standing')
        
        if actions:
            most_common_action = max(set(actions), key=actions.count)
        else:
            most_common_action = 'Standing'
        
        frames_tensor = torch.stack(frames[:self.sequence_length])
        action_idx = ACTION_TO_IDX.get(most_common_action, 0)
        
        return {
            'frames': frames_tensor,
            'action': torch.tensor(action_idx, dtype=torch.long),
            'action_name': most_common_action
        }