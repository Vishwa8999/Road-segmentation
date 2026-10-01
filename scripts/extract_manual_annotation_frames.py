import cv2
import os
import glob
from pathlib import Path

def extract_frames_for_annotation(video_dir, output_dir, frames_per_video=15):
    """
    Extract evenly spaced frames from all videos in the given directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(video_dir, "*.MOV"))
    
    total_extracted = 0
    
    for video_path in video_files:
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            continue
            
        print(f"Processing {video_name} (Total frames: {total_frames})")
        
        # Calculate evenly spaced frame indices
        step = max(1, total_frames // frames_per_video)
        frame_indices = [i * step for i in range(frames_per_video)]
        
        # Make sure we don't go out of bounds
        frame_indices = [idx for idx in frame_indices if idx < total_frames]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Save the frame
                out_name = f"{video_name}_frame_{idx:04d}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, frame)
                total_extracted += 1
            else:
                print(f"Failed to read frame {idx} from {video_name}")
                
        cap.release()
        
    print(f"\nDone! Extracted {total_extracted} frames to {output_dir}")

if __name__ == "__main__":
    video_dir = r"C:\Users\svish\Downloads\files (1)"
    output_dir = r"C:\Users\svish\Downloads\Testing model 1\deepglobe-road-segmentation\data\manual_annotation\images"
    
    # 7 videos * 15 frames = 105 total frames
    extract_frames_for_annotation(video_dir, output_dir, frames_per_video=15)
