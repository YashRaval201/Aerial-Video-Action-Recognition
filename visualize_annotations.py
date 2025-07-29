#!/usr/bin/env python3
import cv2
import os
import sys
import argparse
from pathlib import Path

def visualize_annotations(video_path, output_path):
    """Overlay annotations on video and save output"""
    
    # Find annotation folder
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    # Try different annotation folder patterns
    annotation_folder = None
    # Remove .mp4 extension if it appears twice
    clean_name = video_name.replace('.mp4', '') if video_name.endswith('.mp4') else video_name
    patterns = [
        f"{video_path}_annotations",  # full path + _annotations
        os.path.join(video_path_obj.parent, f"{video_name}_annotations"),  # video_name + _annotations
        os.path.join(video_path_obj.parent, f"{clean_name}_annotations"),  # clean name + _annotations
        os.path.join(video_path_obj.parent, f"{video_path_obj.name}_annotations")  # full filename + _annotations
    ]
    
    for pattern in patterns:
        if os.path.exists(pattern):
            annotation_folder = pattern
            break
    
    if annotation_folder is None:
        print(f"Error: Annotation folder not found. Tried patterns:")
        for pattern in patterns:
            print(f"  - {pattern}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Load annotations for current frame
        annotation_file = os.path.join(annotation_folder, f"{frame_idx:04d}.txt")
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 7:
                        person, action, person_id, x_center, y_center, w, h = parts[:7]
                        
                        # Convert normalized coordinates to pixel coordinates
                        x_center = float(x_center) * width
                        y_center = float(y_center) * height
                        w = float(w) * width
                        h = float(h) * height
                        
                        # Calculate bounding box corners
                        x1 = int(x_center - w/2)
                        y1 = int(y_center - h/2)
                        x2 = int(x_center + w/2)
                        y2 = int(y_center + h/2)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"ID:{person_id} {action}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")
    
    cap.release()
    out.release()
    print(f"Output saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Visualize annotations on video')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--output', '-o', help='Output video path', default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Generate output path if not provided
    if args.output is None:
        video_path = Path(args.video_path)
        args.output = str(video_path.parent / f"{video_path.stem}_annotated.mp4")
    
    success = visualize_annotations(args.video_path, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()