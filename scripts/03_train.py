#!/usr/bin/env python3
"""
03_train.py

Google Colab Compatible YOLO Training Script

Features
--------
✓ Starts a new training automatically.
✓ Resumes automatically if interrupted.
✓ Detects completed training.
✓ Works perfectly with Google Colab.
"""

from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

MODEL = "yolov8n-obb.pt"

DATA = "dataset/data.yaml"

EPOCHS = 100

IMGSZ = 640

BATCH = 8

PROJECT = "training_runs"

NAME = "water_meter_yolov8n_obb"

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

HSV_H = 0.015
HSV_S = 0.50
HSV_V = 0.30

DEGREES = 8.0
TRANSLATE = 0.10
SCALE = 0.40
SHEAR = 2.0
PERSPECTIVE = 0.0002

FLIPUD = 0.0
FLIPLR = 0.0

MOSAIC = 1.0
MIXUP = 0.0
COPY_PASTE = 0.0


def main():

    print("=" * 70)
    print("Water Meter YOLO Training")
    print("=" * 70)

    data_yaml = Path(DATA)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found:\n{data_yaml}"
        )

    run_dir = Path(PROJECT) / NAME
    weights_dir = run_dir / "weights"

    last_checkpoint = weights_dir / "last.pt"
    best_checkpoint = weights_dir / "best.pt"

    # ==========================================================
    # Resume training if interrupted
    # ==========================================================

    if last_checkpoint.exists():

        print("\nCheckpoint detected.")
        print(last_checkpoint)

        print("\nResuming previous training...\n")

        model = YOLO(str(last_checkpoint))

        model.train(resume=True)

        print("\nTraining resumed successfully.")

        return

    # ==========================================================
    # Already trained
    # ==========================================================

    if best_checkpoint.exists():

        print("\nTraining already completed.")
        print("\nBest model:")

        print(best_checkpoint)

        print("\nDelete the training folder if you want to train again.")

        return

    # ==========================================================
    # Start new training
    # ==========================================================

    print("\nNo checkpoint found.")
    print("Starting a NEW training...\n")

    model = YOLO(MODEL)

    model.train(

        data=str(data_yaml),

        epochs=EPOCHS,

        imgsz=IMGSZ,

        batch=BATCH,

        project=PROJECT,

        name=NAME,

        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,

        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,

        flipud=FLIPUD,
        fliplr=FLIPLR,

        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,

        save=True,
        save_period=1,
        exist_ok=True
    )

    print("\n")
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print("\nBest weights:")

    print(best_checkpoint)

    print("\nLast checkpoint:")

    print(last_checkpoint)


if __name__ == "__main__":
    main()
