#!/usr/bin/env python3
"""
07_prepare_stage2_digit_crops.py

Builds a SECOND dataset for a two-stage detector:

    Stage 1 (already trained) : meter, window        -> full image
    Stage 2 (this script builds the data for it)      : 0-9, u  -> cropped window only

Why: digits occupy a tiny fraction of a full water-meter photo, so even a
"correct" detection (IoU >= 0.5) has a lot of room to be loose, which is why
mAP50-95 is much lower than mAP50 for every digit class. Cropping to the
window region and upscaling the crop makes each digit occupy far more of the
frame, which should tighten box localization (raise mAP50-95) without
needing a single new labeled image.

INPUT (existing dataset produced by 02_prepare_dataset.py):
    dataset/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt      (OBB format: class x1 y1 x2 y2 x3 y3 x4 y4, normalized)
        data.yaml

OUTPUT:
    dataset_stage2/
        images/{train,val,test}/*.jpg      (cropped + letterboxed to --imgsz)
        labels/{train,val,test}/*.txt      (digit classes only, remapped 0-10)
        data.yaml
        skipped.txt                        (images with no window box -> can't crop, logged not silently dropped)

Usage:
    python3 07_prepare_stage2_digit_crops.py dataset dataset_stage2 --imgsz 416 --pad 0.12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Must match CLASS_NAMES order in 02_prepare_dataset.py
STAGE1_CLASSES = ["meter", "window", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "u"]
WINDOW_CLASS_ID = STAGE1_CLASSES.index("window")   # 1
METER_CLASS_ID = STAGE1_CLASSES.index("meter")     # 0

# Digit-only classes for stage 2, in their NEW index order
STAGE2_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "u"]
# old stage-1 class_id -> new stage-2 class_id
OLD_TO_NEW = {STAGE1_CLASSES.index(name): i for i, name in enumerate(STAGE2_CLASSES)}


def parse_args():
    p = argparse.ArgumentParser(description="Crop to window region and remap digit OBB labels for stage-2 training.")
    p.add_argument("input_dataset", type=Path, help="Existing dataset dir (output of 02_prepare_dataset.py)")
    p.add_argument("output_dataset", type=Path, help="Where to write the stage-2 dataset")
    p.add_argument("--imgsz", type=int, default=416, help="Square size to letterbox each crop to")
    p.add_argument("--pad", type=float, default=0.12, help="Padding around window box, as fraction of window size")
    p.add_argument("--min-visible-frac", type=float, default=0.5,
                   help="A digit label is kept only if this fraction of its bounding area survives the crop")
    return p.parse_args()


def read_obb_labels(label_path: Path) -> List[Tuple[int, List[float]]]:
    """Returns list of (class_id, [x1,y1,x2,y2,x3,y3,x4,y4]) all normalized 0-1."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        out.append((class_id, coords))
    return out


def norm_to_abs(coords: List[float], w: int, h: int) -> np.ndarray:
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts


def find_window_box(labels: List[Tuple[int, List[float]]], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """Returns axis-aligned (x0,y0,x1,y1) in pixels for the largest window box, or None."""
    best = None
    best_area = -1
    for class_id, coords in labels:
        if class_id != WINDOW_CLASS_ID:
            continue
        pts = norm_to_abs(coords, w, h)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best_area = area
            best = (x0, y0, x1, y1)
    if best is None:
        return None
    x0, y0, x1, y1 = best
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def pad_and_clip_box(box, w, h, pad_frac):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def letterbox_resize(img: np.ndarray, target: int) -> Tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio, pad to target x target. Returns (canvas, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, 3), dtype=img.dtype)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def remap_digit_labels(labels, crop_box, orig_w, orig_h, target, min_visible_frac):
    """Remap digit-class OBB points from original image space into the letterboxed crop space."""
    cx0, cy0, cx1, cy1 = crop_box
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    if crop_w <= 0 or crop_h <= 0:
        return []

    scale = min(target / crop_w, target / crop_h)
    pad_x = (target - crop_w * scale) / 2
    pad_y = (target - crop_h * scale) / 2

    new_lines = []
    for class_id, coords in labels:
        if class_id not in OLD_TO_NEW:
            continue  # skip meter/window in stage-2 labels
        pts = norm_to_abs(coords, orig_w, orig_h)  # abs px in ORIGINAL image

        orig_poly_area = cv2.contourArea(pts.astype(np.float32))

        # shift into crop space, then scale+pad into the letterboxed canvas
        pts_crop = pts.copy()
        pts_crop[:, 0] = (pts_crop[:, 0] - cx0) * scale + pad_x
        pts_crop[:, 1] = (pts_crop[:, 1] - cy0) * scale + pad_y

        # clip polygon to the canvas bounds and measure surviving area
        clipped = pts_crop.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, target)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, target)
        clipped_area = cv2.contourArea(clipped.astype(np.float32))
        scaled_orig_area = orig_poly_area * (scale ** 2)
        visible_frac = clipped_area / scaled_orig_area if scaled_orig_area > 0 else 0

        if visible_frac < min_visible_frac:
            continue  # digit mostly cropped out, drop it

        norm_pts = clipped / target  # normalize to [0,1] against the square canvas
        new_class_id = OLD_TO_NEW[class_id]
        flat = " ".join(f"{v:.6f}" for v in norm_pts.flatten())
        new_lines.append(f"{new_class_id} {flat}")

    return new_lines


def process_split(input_dataset: Path, output_dataset: Path, split: str, imgsz: int, pad: float,
                   min_visible_frac: float, skipped_log: List[str]):
    img_dir = input_dataset / "images" / split
    lbl_dir = input_dataset / "labels" / split
    out_img_dir = output_dataset / "images" / split
    out_lbl_dir = output_dataset / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept, skipped_no_window, skipped_no_digits = 0, 0, 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = lbl_dir / (img_path.stem + ".txt")
        labels = read_obb_labels(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            skipped_log.append(f"UNREADABLE: {img_path}")
            continue
        h, w = img.shape[:2]

        window_box = find_window_box(labels, w, h)
        if window_box is None:
            skipped_no_window += 1
            skipped_log.append(f"NO_WINDOW_BOX: {img_path}")
            continue

        crop_box = pad_and_clip_box(window_box, w, h, pad)
        cx0, cy0, cx1, cy1 = crop_box
        crop_img = img[cy0:cy1, cx0:cx1]
        if crop_img.size == 0:
            skipped_log.append(f"EMPTY_CROP: {img_path}")
            continue

        new_labels = remap_digit_labels(labels, crop_box, w, h, imgsz, min_visible_frac)
        if not new_labels:
            skipped_no_digits += 1
            skipped_log.append(f"NO_DIGITS_SURVIVED_CROP: {img_path}")
            continue

        canvas, _, _, _ = letterbox_resize(crop_img, imgsz)

        out_stem = img_path.stem
        cv2.imwrite(str(out_img_dir / f"{out_stem}.jpg"), canvas)
        (out_lbl_dir / f"{out_stem}.txt").write_text("\n".join(new_labels) + "\n", encoding="utf-8")
        kept += 1

    print(f"[{split}] kept={kept}  skipped_no_window={skipped_no_window}  skipped_no_digits={skipped_no_digits}")
    return kept


def write_data_yaml(output_dataset: Path):
    lines = [
        f"path: {output_dataset.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for i, name in enumerate(STAGE2_CLASSES):
        lines.append(f"  {i}: {name}")
    (output_dataset / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.input_dataset.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input_dataset}")

    args.output_dataset.mkdir(parents=True, exist_ok=True)
    skipped_log: List[str] = []
    total_kept = 0
    for split in ["train", "val", "test"]:
        total_kept += process_split(
            args.input_dataset, args.output_dataset, split,
            args.imgsz, args.pad, args.min_visible_frac, skipped_log,
        )

    write_data_yaml(args.output_dataset)
    (args.output_dataset / "skipped.txt").write_text("\n".join(skipped_log) + "\n", encoding="utf-8")

    print("\n========== STAGE-2 DATASET BUILD COMPLETE ==========")
    print(f"Output: {args.output_dataset}")
    print(f"Total crops kept: {total_kept}")
    print(f"Skipped entries logged in: {args.output_dataset / 'skipped.txt'}")
    print(f"Classes: {STAGE2_CLASSES}")
    print("\nNext: run 08_oversample_rare_classes.py on the train split, then 09_train_stage2.py")


if __name__ == "__main__":
    main()
