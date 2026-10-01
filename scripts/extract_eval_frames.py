"""Extract frames from a video for manual annotation as an evaluation set.

Saves each frame as ``data/eval/<frame_idx>_sat.jpg``.
Optionally generates a draft pseudo-mask from the model so you only need to
*correct* it instead of drawing from scratch.

Usage
-----
    # Extract 15 evenly-spaced frames + pseudo-masks:
    python scripts/extract_eval_frames.py \
        --input "C:/Users/svish/Downloads/files (1)/DJI_0529.MOV" \
        --count 15 --pseudo-masks

    # Extract specific frame indices:
    python scripts/extract_eval_frames.py \
        --input "C:/Users/svish/Downloads/files (1)/DJI_0529.MOV" \
        --frames 100 300 600 900 1200 1500 1800 2100 2400 2700
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_frames(
    video_path: str,
    output_dir: Path,
    frame_indices: list[int],
) -> list[Path]:
    """Read *video_path* and write the requested frames as JPEGs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total} frames")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for target_idx in sorted(frame_indices):
        if target_idx >= total:
            print(f"  SKIP frame {target_idx} (only {total} frames)")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  SKIP frame {target_idx} (read failed)")
            continue
        out_path = output_dir / f"{target_idx}_sat.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(out_path)
        print(f"  Saved {out_path.name}  ({frame.shape[1]}x{frame.shape[0]})")

    cap.release()
    return saved


def generate_pseudo_masks(
    frame_paths: list[Path],
    config_path: str,
    checkpoint_path: str,
    resolution: int = 512,
    threshold: float = 0.50,
    min_area: int = 500,
) -> None:
    """Run the model on each extracted frame and save draft masks."""
    import torch

    from src.model import build_model
    from src.predict_video import predict_mask, preprocess_frame
    from src.utils import get_device, load_checkpoint, load_config

    config = load_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(model, Path(checkpoint_path).expanduser().resolve(), device)
    model.eval()

    for fp in frame_paths:
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"  Cannot read {fp}, skipping pseudo-mask")
            continue
        h, w = frame.shape[:2]
        tensor = preprocess_frame(frame, resolution, device)
        mask = predict_mask(
            model, tensor, threshold,
            orig_shape=(h, w), smooth=True, min_area=min_area,
        )
        mask_path = fp.with_name(fp.stem.replace("_sat", "_mask") + ".png")
        cv2.imwrite(str(mask_path), mask)
        road_pct = 100.0 * np.count_nonzero(mask) / mask.size
        print(f"  Pseudo-mask {mask_path.name}  (road {road_pct:.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract frames from a video for the evaluation set.",
    )
    p.add_argument("--input", required=True, help="Path to the source video")
    p.add_argument("--output-dir", default="data/eval",
                   help="Where to save frames (default: data/eval)")
    p.add_argument("--count", type=int, default=15,
                   help="Number of evenly-spaced frames to extract (ignored if --frames given)")
    p.add_argument("--frames", type=int, nargs="+",
                   help="Explicit frame indices to extract")
    p.add_argument("--pseudo-masks", action="store_true",
                   help="Also generate draft pseudo-masks from the model")
    p.add_argument("--config", default="configs/config.uavid_finetune.yaml")
    p.add_argument("--checkpoint",
                   default="outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--min-area", type=int, default=500)
    args = p.parse_args()

    # Determine frame indices
    if args.frames:
        indices = args.frames
    else:
        cap = cv2.VideoCapture(args.input)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Evenly spaced, skipping first and last 5%
        start = int(total * 0.05)
        end = int(total * 0.95)
        step = max(1, (end - start) // args.count)
        indices = list(range(start, end, step))[: args.count]

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    print(f"Extracting {len(indices)} frames -> {out_dir}")
    saved = extract_frames(args.input, out_dir, indices)
    print(f"\n{len(saved)} frames saved.\n")

    if args.pseudo_masks and saved:
        print("Generating pseudo-masks from model predictions...")
        generate_pseudo_masks(
            saved,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            resolution=args.resolution,
            threshold=args.threshold,
            min_area=args.min_area,
        )
        print("\nDone. Review masks in an image editor, correct errors, and re-save.")
        print("Naming convention:  <idx>_sat.jpg  ->  <idx>_mask.png")


if __name__ == "__main__":
    main()
