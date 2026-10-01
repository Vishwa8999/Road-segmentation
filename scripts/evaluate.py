"""Evaluate a road segmentation checkpoint on hand-annotated drone frames.

Runs inference at the specified resolution (default 512×512), computes
per-frame and aggregate IoU, Dice, Precision, Recall, and background
false-positive rate. Prints a comparison table and saves results as JSON.

The evaluation set lives in ``data/eval/`` with the standard naming:
    <id>_sat.jpg   — input frame
    <id>_mask.png  — ground-truth binary mask (white = road)

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint path/to/other.pth --resolution 768
    python scripts/evaluate.py --eval-dir data/eval_v2 --tag round2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model  # noqa: E402
from src.predict_video import predict_mask, preprocess_frame  # noqa: E402
from src.utils import get_device, load_checkpoint, load_config, save_json  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────


def discover_eval_pairs(eval_dir: Path) -> List[Dict[str, Path]]:
    """Find all *_sat.jpg / *_mask.png pairs in *eval_dir*."""
    sat_files = sorted(eval_dir.glob("*_sat.jpg"))
    pairs = []
    for sat in sat_files:
        mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
        if mask.is_file():
            pairs.append({"image": sat, "mask": mask})
        else:
            print(f"  WARNING: no mask for {sat.name}, skipping")
    return pairs


def load_gt_mask(mask_path: Path, threshold: int = 128) -> np.ndarray:
    """Load a ground-truth mask as a binary uint8 array (0 or 255)."""
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    return (raw >= threshold).astype(np.uint8) * 255


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute IoU, Dice, Precision, Recall, BG-FPR from binary masks."""
    p = pred > 0
    g = gt > 0

    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(np.logical_and(~p, ~g).sum())

    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    bg_fpr = fp / (fp + tn + eps)  # false-positive rate on background pixels

    return {
        "iou": round(iou, 6),
        "dice": round(dice, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "bg_fpr": round(bg_fpr, 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ── main pipeline ─────────────────────────────────────────────────────────


def evaluate(
    config_path: str,
    checkpoint_path: str,
    eval_dir: Path,
    output_dir: Path,
    resolution: int = 512,
    threshold: float = 0.50,
    min_area: int = 500,
    smooth: bool = True,
    tag: str = "baseline",
) -> Dict:
    """Run full evaluation and return results dict."""

    # ── load model ────────────────────────────────────────────────────
    config = load_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    print(f"Device            : {device}")

    model = build_model(config).to(device)
    ckpt = load_checkpoint(model, Path(checkpoint_path).expanduser().resolve(), device)
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_val_iou = ckpt.get("best_score", ckpt.get("validation_metrics", {}).get("iou", "?"))
    print(f"Checkpoint        : {checkpoint_path}")
    print(f"  epoch={ckpt_epoch}  val_iou={ckpt_val_iou}")
    print(f"Inference res     : {resolution}×{resolution}")
    print(f"Threshold         : {threshold}")
    print(f"Min area filter   : {min_area} px")
    print()

    # ── discover pairs ────────────────────────────────────────────────
    pairs = discover_eval_pairs(eval_dir)
    if not pairs:
        print(f"ERROR: No *_sat.jpg + *_mask.png pairs found in {eval_dir}")
        sys.exit(1)
    print(f"Found {len(pairs)} annotated frames in {eval_dir}\n")

    # ── per-frame evaluation ──────────────────────────────────────────
    per_frame: List[Dict] = []
    agg_tp = agg_fp = agg_fn = agg_tn = 0.0
    t_start = time.perf_counter()

    for pair in pairs:
        img = cv2.imread(str(pair["image"]))
        if img is None:
            print(f"  Cannot read {pair['image'].name}, skipping")
            continue

        h, w = img.shape[:2]
        gt = load_gt_mask(pair["mask"])
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        tensor = preprocess_frame(img, resolution, device)
        pred = predict_mask(
            model, tensor, threshold,
            orig_shape=(h, w), smooth=smooth, min_area=min_area,
        )

        m = compute_metrics(pred, gt)
        m["frame"] = pair["image"].stem.replace("_sat", "")
        per_frame.append(m)

        agg_tp += m["tp"]
        agg_fp += m["fp"]
        agg_fn += m["fn"]
        agg_tn += m["tn"]

    elapsed = time.perf_counter() - t_start

    # ── aggregate ─────────────────────────────────────────────────────
    eps = 1e-7
    aggregate = {
        "iou": round(agg_tp / (agg_tp + agg_fp + agg_fn + eps), 6),
        "dice": round((2 * agg_tp) / (2 * agg_tp + agg_fp + agg_fn + eps), 6),
        "precision": round(agg_tp / (agg_tp + agg_fp + eps), 6),
        "recall": round(agg_tp / (agg_tp + agg_fn + eps), 6),
        "bg_fpr": round(agg_fp / (agg_fp + agg_tn + eps), 6),
    }

    # ── print table ───────────────────────────────────────────────────
    header = f"{'Frame':>12}  {'IoU':>8}  {'Dice':>8}  {'Prec':>8}  {'Recall':>8}  {'BG-FPR':>8}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in per_frame:
        print(
            f"{row['frame']:>12}  {row['iou']:8.4f}  {row['dice']:8.4f}  "
            f"{row['precision']:8.4f}  {row['recall']:8.4f}  {row['bg_fpr']:8.6f}"
        )
    print("-" * len(header))
    print(
        f"{'AGGREGATE':>12}  {aggregate['iou']:8.4f}  {aggregate['dice']:8.4f}  "
        f"{aggregate['precision']:8.4f}  {aggregate['recall']:8.4f}  {aggregate['bg_fpr']:8.6f}"
    )
    print("=" * len(header))
    print(f"\n{len(per_frame)} frames evaluated in {elapsed:.1f}s")

    # ── save JSON ─────────────────────────────────────────────────────
    result = {
        "tag": tag,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_val_iou": float(ckpt_val_iou) if isinstance(ckpt_val_iou, (int, float)) else ckpt_val_iou,
        "resolution": resolution,
        "threshold": threshold,
        "min_area": min_area,
        "smooth": smooth,
        "num_frames": len(per_frame),
        "aggregate": aggregate,
        "per_frame": per_frame,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{tag}.json"
    save_json(result, out_path)
    print(f"Results saved -> {out_path}\n")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate road segmentation on annotated drone frames.",
    )
    p.add_argument("--config", default="configs/config.uavid_finetune.yaml")
    p.add_argument("--checkpoint",
                   default="outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth")
    p.add_argument("--eval-dir", default="data/eval",
                   help="Directory with *_sat.jpg + *_mask.png pairs")
    p.add_argument("--output-dir", default="outputs/eval",
                   help="Where to save JSON results (default: outputs/eval)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--min-area", type=int, default=500)
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--tag", default="baseline",
                   help="Name tag for the JSON output (e.g. baseline, round1, segformer)")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        eval_dir=eval_dir,
        output_dir=out_dir,
        resolution=args.resolution,
        threshold=args.threshold,
        min_area=args.min_area,
        smooth=not args.no_smooth,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
