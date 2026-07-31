#!/usr/bin/env python3
"""
10_evaluate_stage2.py
Evaluate the stage-2 (digit-only, cropped) YOLO OBB model on its test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stage-2 digit model performance")
    parser.add_argument("--training_root", default="training_runs_stage2")
    parser.add_argument("--data", default="dataset_stage2/data.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))
    if not best_models:
        raise FileNotFoundError(f"No best.pt model found inside: {training_root}")
    return max(best_models, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()
    training_root = Path(args.training_root)
    data_yaml = Path(args.data)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model_path = find_latest_best_model(training_root)

    print("========== STAGE-2 MODEL EVALUATION ==========")
    print(f"Evaluating model: {model_path}")
    print(f"Data YAML: {data_yaml}")
    print()

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        split="test",
    )

    print("\nEvaluation Results (Test Set):")
    print(f"mAP50-95 (Box): {metrics.box.map:.3f}")
    print(f"mAP50 (Box):    {metrics.box.map50:.3f}")
    print(f"mAP75 (Box):    {metrics.box.map75:.3f}")

    print("\nClass-wise metrics:")
    names = metrics.names
    for cls_idx in sorted(names.keys()):
        cls_name = names[cls_idx]
        p = metrics.box.p[cls_idx]
        r = metrics.box.r[cls_idx]
        ap50 = metrics.box.ap50[cls_idx]
        ap = metrics.box.ap[cls_idx]
        flag = "  <-- below 0.90 mAP50" if ap50 < 0.90 else ""
        print(f"  {cls_name:>3}: P={p:.3f}  R={r:.3f}  mAP50={ap50:.3f}  mAP50-95={ap:.3f}{flag}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
