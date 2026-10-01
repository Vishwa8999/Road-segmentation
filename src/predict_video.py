"""Process a video file frame-by-frame with the U-Net road segmentation model.

Reads every frame (or every Nth frame) from a video, runs inference through the
road segmentation U-Net, and writes an output MP4 with a translucent green road
overlay.  Optionally saves the binary mask video as well.

Usage
-----
    python src/predict_video.py \
        --config configs/config.uavid_finetune.yaml \
        --checkpoint outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth \
        --input "C:/Users/svish/Downloads/files (1)/DJI_0529.MOV" \
        --output-dir outputs/uavid/video_output

Optional flags:
    --alpha         Overlay opacity 0.0–1.0 (default 0.40)
    --color         Overlay BGR colour as "B,G,R" (default "0,255,0" = green)
    --save-masks    Also write a mask-only greyscale video
    --every-n       Process every Nth frame (default 1 = all frames)
    --no-overlay    Skip overlay video, only produce masks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure the project root is importable so we can reuse model / utils.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model          # noqa: E402
from src.utils import get_device, load_checkpoint, load_config  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────


def preprocess_frame(
    frame_bgr: np.ndarray,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize an OpenCV BGR frame to *image_size* and return a batch tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0            # H×W×3, float32
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))   # 3×H×W
    return tensor.unsqueeze(0).to(device)                # 1×3×H×W


def preprocess_frame_letterbox(
    frame_bgr: np.ndarray,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Resize preserving aspect ratio, padding with zeros. Returns tensor and (top, bottom, left, right) padding."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = image_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    
    padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    arr = padded.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor.unsqueeze(0).to(device), (pad_t, pad_b, pad_l, pad_r)


def predict_prob(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    tta: bool = False,
) -> np.ndarray:
    """Run inference (with optional horizontal TTA) and return probability map."""
    with torch.inference_mode():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        
        if tta:
            tensor_flipped = torch.flip(tensor, dims=[3])
            logits_flipped = model(tensor_flipped)
            prob_flipped = torch.sigmoid(logits_flipped)[0, 0].cpu().numpy()
            prob_flipped = np.fliplr(prob_flipped)
            prob = (prob + prob_flipped) / 2.0
            
    return prob


def mask_from_prob(
    prob: np.ndarray,
    threshold: float,
    orig_shape: tuple[int, int] | None = None,
    smooth: bool = True,
    min_area: int = 500,
    hysteresis_low: float | None = None,
    pad_info: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Convert probability map to final mask, handling resizing, padding, hysteresis and noise filtering."""
    # 1. Unpad if needed
    if pad_info is not None:
        pad_t, pad_b, pad_l, pad_r = pad_info
        h_pad, w_pad = prob.shape
        prob = prob[pad_t:h_pad-pad_b, pad_l:w_pad-pad_r]

    # 2. Resize to original shape
    if orig_shape is not None:
        h, w = orig_shape
        interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
        prob = cv2.resize(prob, (w, h), interpolation=interp)
        
    # 3. Thresholding (standard or hysteresis)
    if hysteresis_low is not None:
        high_mask = (prob >= threshold).astype(np.uint8)
        low_mask = (prob >= hysteresis_low).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low_mask)
        
        # Vectorized check: find all components that intersect with high_mask
        valid_labels = np.unique(labels[high_mask > 0])
        valid_labels = valid_labels[valid_labels > 0]  # Exclude background label 0
        
        mask = np.isin(labels, valid_labels).astype(np.uint8) * 255
    else:
        mask = (prob >= threshold).astype(np.uint8) * 255
        
    # 4. Minimum area filtering
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        # Vectorized area filtering
        valid_labels = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
        valid_labels = valid_labels[valid_labels > 0]  # Exclude background label 0
        
        mask = np.isin(labels, valid_labels).astype(np.uint8) * 255
        
    return mask


def predict_mask(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    threshold: float,
    orig_shape: tuple[int, int] | None = None,
    smooth: bool = True,
    min_area: int = 500,
) -> np.ndarray:
    """Run inference and return a uint8 binary mask at full or model resolution."""
    prob = predict_prob(model, tensor)
    return mask_from_prob(prob, threshold, orig_shape, smooth, min_area)


def overlay_mask_on_frame(
    frame_bgr: np.ndarray,
    mask_full: np.ndarray,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.40,
) -> np.ndarray:
    """Blend a coloured overlay from *mask_full* onto *frame_bgr*."""
    overlay = frame_bgr.copy()
    overlay[mask_full > 0] = color_bgr
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)


# ── main pipeline ─────────────────────────────────────────────────────────


def process_video(
    config_path: str,
    checkpoint_path: str,
    video_path: str,
    output_dir: str,
    alpha: float = 0.40,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    save_masks: bool = False,
    every_n: int = 1,
    no_overlay: bool = False,
    resolution: int | None = 512,
    threshold: float | None = None,
    smooth: bool = True,
    min_area: int = 500,
    letterbox: bool = False,
    tta: bool = False,
    ema_alpha: float | None = None,
    hysteresis_low: float | None = None,
) -> Path:
    """End-to-end: read video → segment → write overlay MP4.

    Returns the path to the primary output video.
    """
    # ── load config & model ────────────────────────────────────────────
    config = load_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    print(f"Device : {device}")

    model = build_model(config).to(device)
    load_checkpoint(model, Path(checkpoint_path).expanduser().resolve(), device)
    model.eval()

    image_size: int = resolution if resolution is not None else config["dataset"].get("image_size", 256)
    thresh: float = threshold if threshold is not None else config["training"].get("threshold", 0.5)

    print(f"Inference resolution : {image_size}×{image_size}")
    print(f"Threshold            : {thresh}")
    print(f"Smoothing enabled    : {smooth}")
    print(f"Min noise area filter: {min_area} px")
    print(f"Letterboxing         : {letterbox}")
    print(f"TTA (horiz flip)     : {tta}")
    print(f"Temporal EMA alpha   : {ema_alpha}")
    print(f"Hysteresis low thresh: {hysteresis_low}")

    # ── open source video ──────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input  : {video_path}")
    print(f"         {width}×{height} @ {fps_in:.3f} FPS, {total_frames} frames")

    # ── prepare writers ────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    fps_out = fps_in / every_n          # keep real-time playback speed

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    overlay_path = out_dir / f"{stem}_road_overlay.mp4"
    mask_path = out_dir / f"{stem}_road_masks.mp4"

    overlay_writer: cv2.VideoWriter | None = None
    mask_writer: cv2.VideoWriter | None = None

    if not no_overlay:
        overlay_writer = cv2.VideoWriter(
            str(overlay_path), fourcc, fps_out, (width, height),
        )
        if not overlay_writer.isOpened():
            raise RuntimeError(f"Failed to create overlay writer: {overlay_path}")

    if save_masks:
        mask_writer = cv2.VideoWriter(
            str(mask_path), fourcc, fps_out, (width, height), isColor=False,
        )

    # ── frame loop ─────────────────────────────────────────────────────
    frames_processed = 0
    t_start = time.perf_counter()

    pbar = tqdm(total=total_frames // every_n, desc="Segmenting", unit="frame")
    
    prev_prob_ema = None
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % every_n != 0:
            continue

        if letterbox:
            tensor, pad_info = preprocess_frame_letterbox(frame, image_size, device)
        else:
            tensor = preprocess_frame(frame, image_size, device)
            pad_info = None

        prob = predict_prob(model, tensor, tta=tta)
        
        if ema_alpha is not None:
            if prev_prob_ema is None:
                prev_prob_ema = prob
            else:
                prev_prob_ema = ema_alpha * prob + (1.0 - ema_alpha) * prev_prob_ema
            prob_to_use = prev_prob_ema
        else:
            prob_to_use = prob

        mask_full = mask_from_prob(
            prob_to_use,
            threshold=thresh,
            orig_shape=(height, width),
            smooth=smooth,
            min_area=min_area,
            hysteresis_low=hysteresis_low,
            pad_info=pad_info,
        )

        # overlay video
        if overlay_writer is not None:
            blended = overlay_mask_on_frame(frame, mask_full, color_bgr, alpha)
            overlay_writer.write(blended)

        # mask video
        if mask_writer is not None:
            mask_writer.write(mask_full)

        frames_processed += 1
        pbar.update(1)

    pbar.close()

    elapsed = time.perf_counter() - t_start
    throughput = frames_processed / elapsed if elapsed > 0 else 0

    # ── cleanup ────────────────────────────────────────────────────────
    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if mask_writer is not None:
        mask_writer.release()

    # ── summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Frames processed : {frames_processed}")
    print(f"  Elapsed          : {elapsed:.1f} s")
    print(f"  Throughput       : {throughput:.1f} frames/s")
    if overlay_writer is not None:
        size_mb = overlay_path.stat().st_size / (1024 * 1024)
        print(f"  Overlay video    : {overlay_path}  ({size_mb:.1f} MB)")
    if mask_writer is not None:
        size_mb = mask_path.stat().st_size / (1024 * 1024)
        print(f"  Mask video       : {mask_path}  ({size_mb:.1f} MB)")
    print("=" * 60)

    return overlay_path if overlay_writer is not None else mask_path


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run U-Net road segmentation on a video and write an overlay MP4.",
    )
    p.add_argument("--config", default="configs/config.uavid_finetune.yaml",
                   help="YAML config (default: configs/config.uavid_finetune.yaml)")
    p.add_argument("--checkpoint",
                   default="outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth",
                   help="Model checkpoint (.pth)")
    p.add_argument("--input", required=True,
                   help="Path to the source video file (e.g. DJI_0529.MOV)")
    p.add_argument("--output-dir", default=None,
                   help="Directory for output videos (default: <output_dir>/video_output)")
    p.add_argument("--alpha", type=float, default=0.40,
                   help="Overlay opacity 0.0–1.0 (default 0.40)")
    p.add_argument("--color", default="0,255,0",
                   help='Overlay colour as "B,G,R" (default "0,255,0" = green)')
    p.add_argument("--save-masks", action="store_true",
                   help="Also write a greyscale mask-only video")
    p.add_argument("--every-n", type=int, default=1,
                   help="Process every Nth frame (default 1 = all)")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip overlay, only produce mask video (requires --save-masks)")
    p.add_argument("--resolution", type=int, default=512,
                   help="Inference resolution width/height, multiple of 16 (default: 512 for sharp HD)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Probability threshold (default: config value or 0.50)")
    p.add_argument("--min-area", type=int, default=500,
                   help="Minimum connected component area in 4K pixels to filter speckle noise (default: 500)")
    p.add_argument("--no-smooth", action="store_true",
                   help="Disable bicubic probability upsampling")
    p.add_argument("--letterbox", action="store_true",
                   help="Preserve aspect ratio by padding during resize")
    p.add_argument("--tta", action="store_true",
                   help="Enable horizontal flip Test-Time Augmentation")
    p.add_argument("--ema-alpha", type=float, default=None,
                   help="Temporal EMA alpha factor (e.g. 0.3). Lower is smoother but lags.")
    p.add_argument("--hysteresis-low", type=float, default=None,
                   help="Lower threshold for hysteresis (e.g. 0.2). Requires components to hit --threshold.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_overlay and not args.save_masks:
        print("ERROR: --no-overlay requires --save-masks", file=sys.stderr)
        sys.exit(1)

    color_bgr = tuple(int(c) for c in args.color.split(","))
    if len(color_bgr) != 3:
        print("ERROR: --color must be three comma-separated ints", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        config = load_config(args.config)
        output_dir = str(Path(config["paths"]["output_dir"]) / "video_output")

    process_video(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        video_path=args.input,
        output_dir=output_dir,
        alpha=args.alpha,
        color_bgr=color_bgr,
        save_masks=args.save_masks,
        every_n=args.every_n,
        no_overlay=args.no_overlay,
        resolution=args.resolution,
        threshold=args.threshold,
        smooth=not args.no_smooth,
        min_area=args.min_area,
        letterbox=args.letterbox,
        tta=args.tta,
        ema_alpha=args.ema_alpha,
        hysteresis_low=args.hysteresis_low,
    )


if __name__ == "__main__":
    main()
