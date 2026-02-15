"""
YOLOv8m augmented training (corruption augmentation).

Identical to baseline training except:
  - Ultralytics Albumentations is patched with corruption augmentations
  - Output directory: experiments/yolo/augmented

Usage:
    python -m scripts.train_yolo_augmented
"""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from scripts.augmentations import patch_ultralytics_augmentations

# Patch BEFORE importing/using YOLO (so the patched class is used during training)
patch_ultralytics_augmentations()

from ultralytics import YOLO


def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="data/processed/visdrone_yolo6/data.yaml",
        epochs=100,
        batch=4,
        imgsz=1024,
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        patience=100,
        amp=True,
        project="experiments/yolo",
        name="augmented",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
