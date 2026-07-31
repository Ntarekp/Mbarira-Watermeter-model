#!/usr/bin/env python3
"""
09_train_stage2.py

Colab-friendly stage-2 digit trainer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train stage-2 digit-only YOLO OBB model.")
    p.add_argument("--model", default="yolov8n-obb.pt")
    p.add_argument("--data", default="dataset_stage2/data.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--project", default="training_runs_stage2")
    p.add_argument("--name", default="digits_yolov8n_obb")
    p.add_argument("--hsv_h", type=float, default=0.015)
    p.add_argument("--hsv_s", type=float, default=0.5)
    p.add_argument("--hsv_v", type=float, default=0.3)
    p.add_argument("--degrees", type=float, default=10.0)
    p.add_argument("--translate", type=float, default=0.08)
    p.add_argument("--scale", type=float, default=0.3)
    p.add_argument("--shear", type=float, default=2.0)
    p.add_argument("--perspective", type=float, default=0.0002)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--fliplr", type=float, default=0.0)
    p.add_argument("--mosaic", type=float, default=0.5)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy_paste", type=float, default=0.0)
    args, unknown = p.parse_known_args()
    return args, unknown


def main() -> None:
    args, unknown = parse_args()

    running_in_kernel = "ipykernel" in sys.modules or "google.colab" in sys.modules
    got_no_real_flags = not any(a.startswith("--data") or a.startswith("--epochs") for a in sys.argv[1:])
    if running_in_kernel and got_no_real_flags:
        print("=" * 70)
        print("ERROR: This script was executed directly inside the notebook")
        print("kernel instead of via subprocess.")
        print("=" * 70)
        sys.exit(1)

    if unknown:
        print(f"Note: ignoring unrecognized arguments: {unknown}")

    from ultralytics import YOLO

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}\n"
            f"(resolved against current working directory: {Path.cwd()})"
        )

    run_dir = Path(args.project) / args.name
    weights_dir = run_dir / "weights"
    last_ckpt = weights_dir / "last.pt"
    best_ckpt = weights_dir / "best.pt"

    print("=" * 70)
    print("STAGE-2 DIGIT MODEL TRAINING")
    print("=" * 70)
    print(f"Run folder: {run_dir}")

    if last_ckpt.exists():
        print(f"\\nCheckpoint found: {last_ckpt}")
        print("Attempting to resume interrupted training...\\n")
        model = YOLO(str(last_ckpt))
        try:
            model.train(resume=True)
            print("\\nResume completed successfully.")
            print(f"Best weights: {best_ckpt}")
        except AssertionError as e:
            print(f"\\n{e}")
            print("This run already finished all its epochs -- nothing to resume.")
            print(f"Best weights: {best_ckpt}")
        return

    if best_ckpt.exists():
        print(f"\\nA completed run already exists (best.pt present, no last.pt): {best_ckpt}")
        print("Delete this run's folder if you want to start over from scratch.")
        return

    print("\\nNo checkpoint found. Starting a NEW stage-2 training run...\\n")
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
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
        save=True,
        save_period=1,
        exist_ok=True,
    )

    print("\\n" + "=" * 70)
    print("STAGE-2 TRAINING FINISHED")
    print("=" * 70)
    print(f"Best weights: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
