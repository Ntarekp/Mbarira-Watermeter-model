#!/usr/bin/env python3
"""
08_oversample_rare_classes.py

You can't add new labeled images right now, so this squeezes more training
signal out of the images you already have. It finds every TRAIN image that
contains at least one instance of a rare class (default: 'u', '9', '8', or
any class under a count threshold) and creates several augmented copies of
just those images -- rotation, scale jitter, and HSV jitter, all with the
OBB polygon points transformed to match. Only the train split is touched;
val/test are never duplicated or augmented, to avoid leaking near-duplicates
into evaluation.

This does NOT invent new information the model hasn't seen -- it just
increases how many gradient updates the rare classes get per epoch, and
gives the model some pose/lighting variety for those classes it wouldn't
otherwise see. It is a stopgap for genuinely new labeled data, not a
replacement for it.

Usage (run AFTER 07_prepare_stage2_digit_crops.py):
    python3 08_oversample_rare_classes.py dataset_stage2 --copies 4 --threshold-frac 0.5
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "u"]


def parse_args():
    p = argparse.ArgumentParser(description="Oversample images containing rare classes via label-aware augmentation.")
    p.add_argument("dataset_dir", type=Path, help="Stage-2 dataset dir (output of 07_prepare_stage2_digit_crops.py)")
    p.add_argument("--copies", type=int, default=4, help="Augmented copies to generate per rare-class source image")
    p.add_argument("--threshold-frac", type=float, default=0.5,
                   help="Classes with count < threshold_frac * max_class_count are considered rare")
    p.add_argument("--max-rotate", type=float, default=12.0, help="Max +/- rotation in degrees")
    p.add_argument("--scale-jitter", type=float, default=0.15, help="Max +/- scale fraction")
    p.add_argument("--hsv-s", type=float, default=0.4, help="Max saturation jitter fraction")
    p.add_argument("--hsv-v", type=float, default=0.3, help="Max value/brightness jitter fraction")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_labels(label_path: Path) -> List[Tuple[int, np.ndarray]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        class_id = int(parts[0])
        pts = np.array(parts[1:], dtype=np.float32).reshape(4, 2)
        out.append((class_id, pts))
    return out


def count_classes(label_dir: Path) -> Counter:
    counts = Counter()
    for lbl_path in label_dir.glob("*.txt"):
        for class_id, _ in read_labels(lbl_path):
            counts[class_id] += 1
    return counts


def hsv_jitter(img: np.ndarray, max_s: float, max_v: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_factor = 1.0 + random.uniform(-max_s, max_s)
    v_factor = 1.0 + random.uniform(-max_v, max_v)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_factor, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def rotate_and_scale(img: np.ndarray, labels: List[Tuple[int, np.ndarray]],
                      max_rotate: float, scale_jitter: float):
    h, w = img.shape[:2]
    angle = random.uniform(-max_rotate, max_rotate)
    scale = 1.0 + random.uniform(-scale_jitter, scale_jitter)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    new_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    new_labels = []
    for class_id, pts_norm in labels:
        pts_abs = pts_norm.copy()
        pts_abs[:, 0] *= w
        pts_abs[:, 1] *= h
        ones = np.ones((4, 1), dtype=np.float32)
        pts_h = np.hstack([pts_abs, ones])
        transformed = (M @ pts_h.T).T  # (4,2)
        transformed[:, 0] = np.clip(transformed[:, 0], 0, w)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, h)
        norm = transformed.copy()
        norm[:, 0] /= w
        norm[:, 1] /= h
        # drop degenerate polygons that got squashed to near-zero area by clipping
        if cv2.contourArea(transformed.astype(np.float32)) < 4.0:
            continue
        new_labels.append((class_id, norm))
    return new_img, new_labels


def write_label(path: Path, labels: List[Tuple[int, np.ndarray]]):
    lines = []
    for class_id, pts in labels:
        flat = " ".join(f"{v:.6f}" for v in pts.flatten())
        lines.append(f"{class_id} {flat}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)

    train_img_dir = args.dataset_dir / "images" / "train"
    train_lbl_dir = args.dataset_dir / "labels" / "train"

    before_counts = count_classes(train_lbl_dir)
    max_count = max(before_counts.values()) if before_counts else 0
    threshold = max_count * args.threshold_frac
    rare_classes = {cid for cid, cnt in before_counts.items() if cnt < threshold}

    print("========== CLASS COUNTS BEFORE OVERSAMPLING (train split) ==========")
    for cid in sorted(before_counts):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        flag = "  <- RARE, will oversample" if cid in rare_classes else ""
        print(f"  {name:>3}: {before_counts[cid]:4d}{flag}")

    if not rare_classes:
        print("\nNo class fell under the rarity threshold. Nothing to do.")
        return

    # find every train image containing at least one rare-class instance
    target_images = []
    for lbl_path in sorted(train_lbl_dir.glob("*.txt")):
        labels = read_labels(lbl_path)
        if any(cid in rare_classes for cid, _ in labels):
            img_path = train_img_dir / (lbl_path.stem + ".jpg")
            if img_path.exists():
                target_images.append((img_path, lbl_path))

    print(f"\nFound {len(target_images)} train images containing a rare class.")
    print(f"Generating {args.copies} augmented copies each ({len(target_images) * args.copies} new images)...")

    added = 0
    for img_path, lbl_path in target_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        labels = read_labels(lbl_path)

        for i in range(args.copies):
            aug_img, aug_labels = rotate_and_scale(img, labels, args.max_rotate, args.scale_jitter)
            if not aug_labels:
                continue
            aug_img = hsv_jitter(aug_img, args.hsv_s, args.hsv_v)

            new_stem = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(train_img_dir / f"{new_stem}.jpg"), aug_img)
            write_label(train_lbl_dir / f"{new_stem}.txt", aug_labels)
            added += 1

    after_counts = count_classes(train_lbl_dir)
    print(f"\n========== CLASS COUNTS AFTER OVERSAMPLING (train split) ==========")
    for cid in sorted(after_counts):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        before = before_counts.get(cid, 0)
        print(f"  {name:>3}: {before:4d} -> {after_counts[cid]:4d}")

    print(f"\nAdded {added} augmented images to train split.")
    print("val/ and test/ were not touched -- your evaluation numbers stay honest.")


if __name__ == "__main__":
    main()
