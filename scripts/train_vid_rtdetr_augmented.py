"""
RT-DETR-L augmented training on VisDrone-VID (corruption augmentation).

Applies random corruption (noise / blur / lowres) with 50% probability
during training, identical to the DET augmented experiment.

Usage:
    python -m scripts.train_vid_rtdetr_augmented
"""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from scripts.augmentations import patch_ultralytics_augmentations

# Patch BEFORE importing/using RTDETR
patch_ultralytics_augmentations()

from ultralytics import RTDETR


def main():
    model = RTDETR("rtdetr-l.pt")

    model.train(
        data="data/processed/visdrone_vid_yolo6/data.yaml",
        epochs=100,
        batch=2,
        imgsz=1024,
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        patience=100,
        amp=True,
        project="experiments/vid_rtdetr",
        name="augmented",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
