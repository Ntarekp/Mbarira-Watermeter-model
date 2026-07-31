#!/usr/bin/env python3
"""
06_evaluate.py

Evaluate the latest trained YOLO model on the test dataset.
Designed for Google Colab.
"""

from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

try:
    from IPython.display import display, Image
    COLAB = True
except:
    COLAB = False


# =====================================================
# CONFIGURATION
# =====================================================

TRAINING_ROOT = "training_runs"

DATA = "dataset/data.yaml"

IMGSZ = 640

BATCH = 16


# =====================================================
# Helper Functions
# =====================================================

def separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def find_latest_best_model(training_root: Path):

    best_models = list(training_root.glob("*/weights/best.pt"))

    if not best_models:
        raise FileNotFoundError(
            f"No trained model found inside:\n{training_root}"
        )

    return max(best_models, key=lambda p: p.stat().st_mtime)


# =====================================================
# MAIN
# =====================================================

def main():

    separator("YOLO MODEL EVALUATION")

    data_yaml = Path(DATA)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{data_yaml}"
        )

    training_root = Path(TRAINING_ROOT)

    model_path = find_latest_best_model(training_root)

    run_dir = model_path.parent.parent

    print("Model")
    print("-" * 70)
    print("Model Path:")
    print(model_path)

    print("\nModel Size:")
    print(f"{model_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\nDataset")
    print("-" * 70)
    print(data_yaml)

    print("\nImage Size :", IMGSZ)
    print("Batch Size :", BATCH)

    separator("Running Evaluation")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
    )

    separator("OVERALL PERFORMANCE")

    print(f"Precision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")
    print(f"mAP50          : {metrics.box.map50:.4f}")
    print(f"mAP50-95       : {metrics.box.map:.4f}")

    separator("PER CLASS PERFORMANCE")

    header = f"{'Class':<15}{'P':>10}{'R':>10}{'mAP50':>12}{'mAP50-95':>15}"
    print(header)
    print("-" * len(header))

    names = metrics.names

    for i in sorted(names.keys()):

        print(
            f"{names[i]:<15}"
            f"{metrics.box.p[i]:>10.3f}"
            f"{metrics.box.r[i]:>10.3f}"
            f"{metrics.box.ap50[i]:>12.3f}"
            f"{metrics.box.ap[i]:>15.3f}"
        )

    separator("INFERENCE SPEED")

    speed = metrics.speed

    print(f"Preprocess : {speed['preprocess']:.2f} ms/image")
    print(f"Inference : {speed['inference']:.2f} ms/image")
    print(f"Loss       : {speed['loss']:.2f} ms/image")
    print(f"Postprocess: {speed['postprocess']:.2f} ms/image")

    separator("GENERATED FILES")

    expected = [

        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.csv",
    ]

    for file in expected:

        path = run_dir / file

        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")

    separator("VISUAL RESULTS")

    if COLAB:

        images = [

            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "F1_curve.png",
        ]

        for image_name in images:

            image_path = run_dir / image_name

            if image_path.exists():

                print(f"\n{image_name}")

                display(Image(filename=str(image_path)))

    separator("EVALUATION COMPLETE")

    print("Results Folder:")
    print(run_dir)


if __name__ == "__main__":
    main()
