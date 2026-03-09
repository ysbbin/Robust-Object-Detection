"""
YOLOv8m baseline training on VisDrone-VID (clean only).

Usage:
    python -m scripts.train_vid_yolo_baseline
"""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

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
        name="baseline",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
