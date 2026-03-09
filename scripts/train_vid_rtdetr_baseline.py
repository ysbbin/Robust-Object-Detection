"""
RT-DETR-L baseline training on VisDrone-VID (clean only).

Usage:
    python -m scripts.train_vid_rtdetr_baseline
"""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

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
        name="baseline",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
