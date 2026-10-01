"""Convert the Drone_Road_Dataset side-by-side composites into paired files.

Each source PNG is 512x256: left half is the RGB image, right half is the
binary road mask.  This script splits them into the format expected by the
existing training pipeline:

    <id>_sat.jpg   (256x256 RGB image)
    <id>_mask.png  (256x256 greyscale mask)

Usage
-----
    python src/convert_drone_dataset.py \
        --input "C:/Users/svish/Downloads/Drone_Road_Dataset/Drone_Road_Dataset" \
        --output data/drone_roads \
        --subsets dataset256_100cm dataset256_150cm dataset256_200cm \
                 dataset256_300cm dataset256_500cm dataset256_50cm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# Subsets that contain road segmentation masks (not vehicle detection)
ROAD_SUBSETS = [
    "dataset256_50cm",
    "dataset256_100cm",
    "dataset256_150cm",
    "dataset256_200cm",
    "dataset256_300cm",
    "dataset256_500cm",
]


def split_composite(src: Path, dst_dir: Path, prefix: str) -> None:
    """Split a 512x256 side-by-side composite into image + mask files."""
    composite = Image.open(src)
    w, h = composite.size
    if w != 2 * h:
        print(f"  SKIP {src.name}: unexpected size {w}x{h}", file=sys.stderr)
        return

    half = w // 2
    image = composite.crop((0, 0, half, h)).convert("RGB")
    mask = composite.crop((half, 0, w, h)).convert("L")

    image.save(dst_dir / f"{prefix}_sat.jpg", quality=95)
    mask.save(dst_dir / f"{prefix}_mask.png")


def convert_subset(
    subset_dir: Path,
    output_dir: Path,
    subset_name: str,
) -> dict[str, int]:
    """Convert one subset (e.g. dataset256_100cm) into paired files."""
    counts: dict[str, int] = {}

    for split in ("train", "valid", "test"):
        split_dir = subset_dir / split
        if not split_dir.is_dir():
            continue

        files = sorted(split_dir.glob("*.png"))
        if not files:
            continue

        dst = output_dir
        dst.mkdir(parents=True, exist_ok=True)

        for f in files:
            # Prefix: <subset>_<split>_<stem> to ensure uniqueness
            prefix = f"{subset_name}_{split}_{f.stem}"
            split_composite(f, dst, prefix)

        counts[split] = len(files)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Drone_Road_Dataset composites to paired sat/mask files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root of Drone_Road_Dataset (contains dataset256_* subdirs)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for paired files",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=ROAD_SUBSETS,
        help="Subset directories to convert (default: all road subsets)",
    )
    args = parser.parse_args()

    input_root = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for subset_name in args.subsets:
        subset_dir = input_root / subset_name
        if not subset_dir.is_dir():
            print(f"WARNING: Subset not found, skipping: {subset_dir}")
            continue

        print(f"\nConverting {subset_name}...")
        counts = convert_subset(subset_dir, output_dir, subset_name)
        subset_total = sum(counts.values())
        total += subset_total
        print(f"  {counts}  (subtotal: {subset_total})")

    print(f"\n{'=' * 50}")
    print(f"Total paired files written: {total}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
