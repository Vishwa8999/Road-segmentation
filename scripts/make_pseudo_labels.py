"""Generate a pseudo-labeled dataset from raw drone videos using a trained model.

Usage:
    python scripts/make_pseudo_labels.py \
        --input-dir "C:/Users/svish/Downloads/files (1)" \
        --output-dir data/drone_pseudo_train \
        --interval-sec 3 \
        --tta
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model
from src.utils import get_device, load_checkpoint, load_config
from src.predict_video import (
    preprocess_frame_letterbox,
    predict_prob,
    mask_from_prob,
)

def parse_args():
    p = argparse.ArgumentParser(description="Generate pseudo-labels from drone videos.")
    p.add_argument("--config", default="configs/config.uavid_finetune.yaml")
    p.add_argument("--checkpoint", default="outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth")
    p.add_argument("--input-dir", required=True, help="Directory containing raw drone videos")
    p.add_argument("--output-dir", required=True, help="Output directory for generated dataset")
    p.add_argument("--interval-sec", type=float, default=3.0, help="Seconds between extracted frames")
    p.add_argument("--resolution", type=int, default=512, help="Inference resolution")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    p.add_argument("--min-area", type=int, default=500, help="Min connected component area")
    p.add_argument("--tta", action="store_true", help="Enable TTA for higher quality pseudo-labels")
    return p.parse_args()

def main():
    args = parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    config = load_config(args.config)
    device = get_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(model, Path(args.checkpoint).expanduser().resolve(), device)
    model.eval()
    
    videos = list(in_dir.glob("*.mp4")) + list(in_dir.glob("*.MOV")) + list(in_dir.glob("*.mov"))
    if not videos:
        print(f"No videos found in {in_dir}")
        return
        
    print(f"Found {len(videos)} videos. Outputting to {out_dir}")
    print(f"Extracting 1 frame every {args.interval_sec} seconds. TTA={args.tta}")
    
    global_idx = 0
    
    for video_path in videos:
        print(f"\nProcessing {video_path.name}...")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Failed to open, skipping.")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_interval = max(1, int(fps * args.interval_sec))
        
        frame_idx = 0
        pbar = tqdm(total=total_frames // frame_interval, unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Inference
                tensor, pad_info = preprocess_frame_letterbox(frame, args.resolution, device)
                prob = predict_prob(model, tensor, tta=args.tta)
                mask = mask_from_prob(
                    prob,
                    threshold=args.threshold,
                    orig_shape=(height, width),
                    smooth=True,
                    min_area=args.min_area,
                    pad_info=pad_info,
                )
                
                # Save pair
                # Format: vidName_frameIdx_sat.jpg -> vidName_frameIdx_mask.png
                prefix = f"{video_path.stem}_{frame_idx:05d}"
                sat_path = out_dir / f"{prefix}_sat.jpg"
                mask_path = out_dir / f"{prefix}_mask.png"
                
                cv2.imwrite(str(sat_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(mask_path), mask)
                
                global_idx += 1
                pbar.update(1)
                
            frame_idx += 1
            
        cap.release()
        pbar.close()
        
    print(f"\nDone! Generated {global_idx} training pairs in {out_dir}")

if __name__ == "__main__":
    main()
