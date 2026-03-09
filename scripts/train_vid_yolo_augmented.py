"""
YOLOv8m augmented training on VisDrone-VID (corruption augmentation).

Applies random corruption (noise / blur / lowres) with 50% probability
during training, identical to the DET augmented experiment.

Usage:
    python -m scripts.train_vid_yolo_augmented
"""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from scripts.augmentations import patch_ultralytics_augmentations

# Patch BEFORE importing/using YOLO
patch_ultralytics_augmentations()

from ultralytics import YOLO


def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="data/processed/visdrone_vid_yolo6/data.yaml",
        epochs=100,
        batch=4,
        imgsz=1024,
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        patience=100,
        amp=True,
        project="experiments/vid_yolo",
        name="augmented",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
