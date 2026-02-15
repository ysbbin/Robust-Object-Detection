"""
RT-DETR-L augmented training (corruption augmentation).

Identical to baseline training except:
  - Ultralytics Albumentations is patched with corruption augmentations
  - Output directory: experiments/rtdetr/augmented

Usage:
    python -m scripts.train_rtdetr_augmented
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
        data="data/processed/visdrone_yolo6/data.yaml",
        epochs=100,
        batch=2,
        imgsz=1024,
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        patience=100,
        amp=True,
        project="experiments/rtdetr",
        name="augmented",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
