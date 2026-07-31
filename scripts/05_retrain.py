#!/usr/bin/env python3
"""
05_retrain.py
Retrain / continue training YOLO using the latest best.pt model found in training_runs/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Continue training from latest best.pt")
    parser.add_argument("--training_root", default="training_runs")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="training_runs")
    parser.add_argument("--prefix", default="water_meter_retrain")

    # Data augmentation arguments
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV hue augmentation (fraction)")
    parser.add_argument("--hsv_s", type=float, default=0.5, help="HSV saturation augmentation (fraction)")
    parser.add_argument("--hsv_v", type=float, default=0.3, help="HSV value augmentation (fraction)")
    parser.add_argument("--degrees", type=float, default=8.0, help="Image rotation (degrees)")
    parser.add_argument("--translate", type=float, default=0.1, help="Image translation (fraction)")
    parser.add_argument("--scale", type=float, default=0.4, help="Image scaling (fraction)")
    parser.add_argument("--shear", type=float, default=2.0, help="Image shear (degrees)")
    parser.add_argument("--perspective", type=float, default=0.0002, help="Image perspective (fraction)")
    parser.add_argument("--flipud", type=float, default=0.0, help="Image flip up-down (probability)")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Image flip left-right (probability)")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation (probability)")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp augmentation (probability)")
    parser.add_argument("--copy_paste", type=float, default=0.0, help="Copy-paste augmentation (probability)")
    return parser.parse_args()


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))
    if not best_models:
        raise FileNotFoundError(f"No best.pt model found inside: {training_root}")
    return max(best_models, key=lambda p: p.stat().st_mtime)


def make_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    training_root = Path(args.training_root)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    latest_best = find_latest_best_model(training_root)
    run_name = make_run_name(args.prefix)

    print("========== YOLO RETRAINING ==========")
    print(f"Starting from: {latest_best}")
    print(f"Data YAML: {data_yaml}")
    print(f"New run name: {run_name}")
    print(f"Epochs: {args.epochs}")
    print()

    model = YOLO(str(latest_best))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=run_name,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
    )

    print("\nRetraining completed.")
    print(f"New run saved under: {args.project}/{run_name}")


if __name__ == "__main__":
    main()
