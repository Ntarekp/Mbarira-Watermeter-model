#!/usr/bin/env python3
"""
04_predict.py
Run YOLO inference on test images and save visualized predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO prediction on test images")
    parser.add_argument("--training_root", default="training_runs")
    parser.add_argument("--test_dir", default="dataset/images/test")
    parser.add_argument("--output_project", default="prediction_outputs")
    parser.add_argument("--output_name", default="test_predictions")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))
    if not best_models:
        raise FileNotFoundError(f"No best.pt model found inside: {training_root}")
    return max(best_models, key=lambda p: p.stat().st_mtime)


def iter_test_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def main() -> None:
    args = parse_args()
    training_root = Path(args.training_root)
    test_dir = Path(args.test_dir)
    output_project = Path(args.output_project)

    model_path = find_latest_best_model(training_root)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_dir}")

    test_images = list(iter_test_images(test_dir))
    if not test_images:
        raise RuntimeError(f"No supported test images found in: {test_dir}")

    output_project.mkdir(parents=True, exist_ok=True)

    print("========== YOLO TEST PREDICTION ==========")
    print(f"Model: {model_path}")
    print(f"Test folder: {test_dir}")
    print(f"Images: {len(test_images)}")
    print()

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(test_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=False,
        project=str(output_project),
        name=args.output_name,
        exist_ok=True,
    )

    output_dir = output_project / args.output_name
    label_dir = output_dir / "labels"

    print("\nPrediction completed.")
    print(f"Annotated images saved in: {output_dir}")
    print(f"Prediction txt files saved in: {label_dir}")
    print("\nPer-image summary:")
    for image_path, result in zip(test_images, results):
        num_boxes = 0 if result.boxes is None else len(result.boxes)
        print(f"  - {image_path.name}: {num_boxes} detection(s)")


if __name__ == "__main__":
    main()
