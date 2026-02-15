"""
Demo inference: side-by-side Baseline vs Augmented detection on corrupted images.

For each model pair, generates comparison images:
  [Clean + GT]  [Blur + Baseline]  [Blur + Augmented]

Usage:
    python -m scripts.demo_inference
"""

import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import gc
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from pycocotools.coco import COCO

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
COCO_TESTSET_ROOT = Path("data/testsets/coco6")
YOLO_TESTSET_ROOT = Path("data/testsets/yolo6")

CLASS_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus", 6: "motor"}
CLASS_COLORS = {
    1: (30, 144, 255),   # pedestrian - dodger blue
    2: (0, 200, 0),      # car - green
    3: (255, 165, 0),    # van - orange
    4: (148, 103, 189),  # truck - purple
    5: (220, 20, 60),    # bus - crimson
    6: (255, 215, 0),    # motor - gold
}

CKPTS = {
    "FasterRCNN":      Path("experiments/frcnn/baseline_clean/best.pth"),
    "FasterRCNN_aug":  Path("experiments/frcnn/augmented/best.pth"),
    "RT-DETR-L":       Path("experiments/rtdetr/baseline_clean/weights/best.pt"),
    "RT-DETR-L_aug":   Path("experiments/rtdetr/augmented/weights/best.pt"),
    "YOLOv8m":         Path("experiments/yolo/baseline_clean/weights/best.pt"),
    "YOLOv8m_aug":     Path("experiments/yolo/augmented/weights/best.pt"),
}

OUT_DIR = Path("experiments/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.35
N_SAMPLES = 5
SEED = 42

# Conditions to visualize
VIS_CONDITIONS = ["Test_Clean", "Test_Blur"]


# ──────────────────────────────────────────────
# Image selection: pick images with many annotations
# ──────────────────────────────────────────────
def select_images(ann_file: str, n: int = N_SAMPLES) -> list:
    coco = COCO(ann_file)
    img_ids = list(coco.imgs.keys())

    # Count annotations per image
    counts = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        counts.append((img_id, len(ann_ids)))

    # Sort by count descending, pick from top candidates with diversity
    counts.sort(key=lambda x: -x[1])
    top = counts[:50]
    random.seed(SEED)
    selected = random.sample(top, min(n, len(top)))
    return [s[0] for s in selected]


# ──────────────────────────────────────────────
# Drawing utilities
# ──────────────────────────────────────────────
def draw_boxes(img: np.ndarray, boxes, labels, scores=None,
               is_gt: bool = False) -> np.ndarray:
    out = img.copy()
    for i, (box, label) in enumerate(zip(boxes, labels)):
        label = int(label)
        x1, y1, x2, y2 = [int(c) for c in box]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        thickness = 2 if is_gt else 2
        line_type = cv2.LINE_AA

        if is_gt:
            # Dashed-style for GT: draw solid with slightly different appearance
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, line_type)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, line_type)

        # Label text
        cls_name = CLASS_NAMES.get(label, f"cls{label}")
        if scores is not None:
            text = f"{cls_name} {scores[i]:.2f}"
        else:
            text = cls_name

        font_scale = 0.5
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_thickness)
        # Background rectangle
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    font_thickness, cv2.LINE_AA)
    return out


def add_title(img: np.ndarray, title: str, height: int = 40) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.zeros((height, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    font_scale = 0.7
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x = (w - tw) // 2
    y = (height + th) // 2
    cv2.putText(bar, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


# ──────────────────────────────────────────────
# Faster R-CNN inference
# ──────────────────────────────────────────────
def build_frcnn(ckpt_path: Path):
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def infer_frcnn(model, img_bgr: np.ndarray, conf_thr: float = CONF_THRESHOLD):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transforms = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    tensor = transforms(pil_img).to(DEVICE)

    outputs = model([tensor])[0]
    mask = outputs["scores"] >= conf_thr
    boxes = outputs["boxes"][mask].cpu().numpy()
    labels = outputs["labels"][mask].cpu().numpy()
    scores = outputs["scores"][mask].cpu().numpy()
    return boxes, labels, scores


# ──────────────────────────────────────────────
# Ultralytics inference
# ──────────────────────────────────────────────
def build_ultralytics(ckpt_path: Path, model_cls: str):
    from ultralytics import YOLO, RTDETR
    if model_cls == "RTDETR":
        model = RTDETR(str(ckpt_path))
        # Workaround: reset shapes to force _generate_anchors on correct device
        for m in model.model.modules():
            if hasattr(m, "valid_mask") and hasattr(m, "shapes"):
                m.shapes = []
    else:
        model = YOLO(str(ckpt_path))
    return model


def infer_ultralytics(model, img_bgr: np.ndarray, conf_thr: float = CONF_THRESHOLD):
    results = model.predict(img_bgr, imgsz=1024, conf=conf_thr,
                            device=0, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    # Ultralytics uses 0-indexed classes, our COCO annotations are 1-indexed
    labels = results.boxes.cls.cpu().numpy().astype(int) + 1
    return boxes, labels, scores


# ──────────────────────────────────────────────
# Generate comparisons
# ──────────────────────────────────────────────
def generate_comparison(img_ids: list, model_pair: tuple,
                        model_name: str, infer_fn, coco_clean, coco_blur):
    base_model, aug_model = model_pair
    clean_img_dir = COCO_TESTSET_ROOT / "Test_Clean" / "images" / "val"
    blur_img_dir = COCO_TESTSET_ROOT / "Test_Blur" / "images" / "val"

    for img_id in img_ids:
        img_info = coco_clean.loadImgs(img_id)[0]
        fname = img_info["file_name"]

        # Load images
        clean_bgr = cv2.imread(str(clean_img_dir / fname))
        blur_bgr = cv2.imread(str(blur_img_dir / fname))
        if clean_bgr is None or blur_bgr is None:
            continue

        # Ground truth from clean annotations
        ann_ids = coco_clean.getAnnIds(imgIds=[img_id])
        anns = coco_clean.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w > 0 and h > 0:
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(a["category_id"])

        # Draw GT on clean
        clean_drawn = draw_boxes(clean_bgr, gt_boxes, gt_labels, is_gt=True)
        clean_drawn = add_title(clean_drawn, "Clean (Ground Truth)")

        # Inference on blur with baseline
        base_boxes, base_labels, base_scores = infer_fn(base_model, blur_bgr)
        blur_base = draw_boxes(blur_bgr, base_boxes, base_labels, base_scores)
        blur_base = add_title(blur_base, f"Blur + {model_name} (Baseline)")

        # Inference on blur with augmented
        aug_boxes, aug_labels, aug_scores = infer_fn(aug_model, blur_bgr)
        blur_aug = draw_boxes(blur_bgr.copy(), aug_boxes, aug_labels, aug_scores)
        blur_aug = add_title(blur_aug, f"Blur + {model_name} (Augmented)")

        # Resize all to same height (before title)
        target_h = 480
        panels = []
        for panel in [clean_drawn, blur_base, blur_aug]:
            h, w = panel.shape[:2]
            scale = target_h / h
            resized = cv2.resize(panel, (int(w * scale), target_h))
            panels.append(resized)

        # Ensure all panels have same height, add separator
        final_h = panels[0].shape[0]
        sep = np.ones((final_h, 3, 3), dtype=np.uint8) * 200
        combined = np.hstack([panels[0], sep, panels[1], sep, panels[2]])

        # Count detections for filename
        n_base = len(base_boxes)
        n_aug = len(aug_boxes)
        n_gt = len(gt_boxes)

        out_path = OUT_DIR / f"{model_name}_img{img_id:04d}_gt{n_gt}_base{n_base}_aug{n_aug}.jpg"
        cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Saved: {out_path.name}  (GT={n_gt}, Base={n_base}, Aug={n_aug})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Generating demo inference comparisons ...\n")

    # Select images with many objects
    ann_clean = str(COCO_TESTSET_ROOT / "Test_Clean" / "annotations" / "instances_val.json")
    ann_blur = str(COCO_TESTSET_ROOT / "Test_Blur" / "annotations" / "instances_val.json")
    coco_clean = COCO(ann_clean)
    coco_blur = COCO(ann_blur)

    img_ids = select_images(ann_clean, N_SAMPLES)
    print(f"Selected {len(img_ids)} images: {img_ids}\n")

    # ── Faster R-CNN ──
    print("=" * 50)
    print("  Faster R-CNN: Baseline vs Augmented")
    print("=" * 50)
    base = build_frcnn(CKPTS["FasterRCNN"])
    aug = build_frcnn(CKPTS["FasterRCNN_aug"])
    generate_comparison(img_ids, (base, aug), "FRCNN",
                        infer_frcnn, coco_clean, coco_blur)
    del base, aug
    torch.cuda.empty_cache()
    gc.collect()

    # ── RT-DETR-L ──
    print("\n" + "=" * 50)
    print("  RT-DETR-L: Baseline vs Augmented")
    print("=" * 50)
    base = build_ultralytics(CKPTS["RT-DETR-L"], "RTDETR")
    aug = build_ultralytics(CKPTS["RT-DETR-L_aug"], "RTDETR")
    generate_comparison(img_ids, (base, aug), "RT-DETR",
                        infer_ultralytics, coco_clean, coco_blur)
    del base, aug
    torch.cuda.empty_cache()
    gc.collect()

    # ── YOLOv8m ──
    print("\n" + "=" * 50)
    print("  YOLOv8m: Baseline vs Augmented")
    print("=" * 50)
    base = build_ultralytics(CKPTS["YOLOv8m"], "YOLO")
    aug = build_ultralytics(CKPTS["YOLOv8m_aug"], "YOLO")
    generate_comparison(img_ids, (base, aug), "YOLOv8m",
                        infer_ultralytics, coco_clean, coco_blur)
    del base, aug
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nAll demo images saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
