"""
Evaluate baseline models on RESTORED test sets.

Runs 3 baseline models x 4 test sets = 12 evaluations on images
that were preprocessed by the restoration U-Net.

Usage:
    python -m scripts.eval_restored
"""

import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import gc
import json
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts.coco_detection_dataset import COCODetectionDataset, collate_fn

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
VARIANTS = ["Test_Clean", "Test_Noise", "Test_Blur", "Test_LowRes"]
SHORT = {"Test_Clean": "Clean", "Test_Noise": "Noise",
         "Test_Blur": "Blur", "Test_LowRes": "LowRes"}

CLASS_NAMES = ["pedestrian", "car", "van", "truck", "bus", "motor"]

COCO_TESTSET_ROOT = Path("data/testsets/coco6_restored")
YOLO_TESTSET_ROOT = Path("data/testsets/yolo6_restored")

CKPTS = {
    "FasterRCNN":  Path("experiments/frcnn/baseline_clean/best.pth"),
    "RT-DETR-L":   Path("experiments/rtdetr/baseline_clean/weights/best.pt"),
    "YOLOv8m":     Path("experiments/yolo/baseline_clean/weights/best.pt"),
}

MODEL_ORDER = ["FasterRCNN", "RT-DETR-L", "YOLOv8m"]

OUT_DIR = Path("experiments")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Faster R-CNN
# ──────────────────────────────────────────────
def _build_frcnn(ckpt_path: Path):
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model


def _frcnn_transforms():
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


@torch.no_grad()
def _eval_frcnn_variant(model, variant: str) -> dict:
    img_dir = str(COCO_TESTSET_ROOT / variant / "images" / "val")
    ann_file = str(COCO_TESTSET_ROOT / variant / "annotations" / "instances_val.json")

    ds = COCODetectionDataset(img_dir, ann_file, transforms=_frcnn_transforms())
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn, pin_memory=True)

    coco_gt = COCO(ann_file)
    results = []

    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": img_id, "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
                })

    if len(results) == 0:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "per_class_ap50": {}}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    per_class_ap50 = _extract_per_class_ap50(coco_eval, coco_gt)
    return {"mAP50_95": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "per_class_ap50": per_class_ap50}


def _extract_per_class_ap50(coco_eval, coco_gt):
    precision = coco_eval.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    per_class = {}
    for k_idx, cat_id in enumerate(cat_ids):
        cat_info = coco_gt.loadCats(cat_id)[0]
        ap50 = precision[0, :, k_idx, 0, 2]
        ap50 = ap50[ap50 > -1]
        per_class[cat_info["name"]] = float(np.mean(ap50)) if len(ap50) > 0 else 0.0
    return per_class


# ──────────────────────────────────────────────
# Ultralytics
# ──────────────────────────────────────────────
def _eval_ultralytics(model_path: str, model_cls: str, variant: str) -> dict:
    from ultralytics import YOLO, RTDETR

    data_yaml = str(YOLO_TESTSET_ROOT / variant / "data.yaml")

    if model_cls == "RTDETR":
        model = RTDETR(model_path)
        for m in model.model.modules():
            if hasattr(m, "valid_mask") and hasattr(m, "shapes"):
                m.shapes = []
    else:
        model = YOLO(model_path)

    results = model.val(data=data_yaml, imgsz=1024, batch=1, device=0, verbose=False)

    ap50_arr = results.box.ap50
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = float(ap50_arr[i]) if i < len(ap50_arr) else 0.0

    metrics = {"mAP50_95": float(results.box.map),
               "mAP50": float(results.box.map50),
               "per_class_ap50": per_class}

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Evaluating 3 BASELINE models on RESTORED test sets\n")

    all_results = {}
    t0 = time.time()

    # Faster R-CNN
    print("=" * 60)
    print("  FasterRCNN (Restored)")
    print("=" * 60)
    model = _build_frcnn(CKPTS["FasterRCNN"])
    all_results["FasterRCNN"] = {}
    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_frcnn_variant(model, v)
        all_results["FasterRCNN"][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # RT-DETR-L
    print("\n" + "=" * 60)
    print("  RT-DETR-L (Restored)")
    print("=" * 60)
    all_results["RT-DETR-L"] = {}
    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_ultralytics(str(CKPTS["RT-DETR-L"]), "RTDETR", v)
        all_results["RT-DETR-L"][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")

    # YOLOv8m
    print("\n" + "=" * 60)
    print("  YOLOv8m (Restored)")
    print("=" * 60)
    all_results["YOLOv8m"] = {}
    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_ultralytics(str(CKPTS["YOLOv8m"]), "YOLO", v)
        all_results["YOLOv8m"][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed/60:.1f} min")

    # Save results
    out_path = OUT_DIR / "eval_restored_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {out_path.resolve()}")

    # Print summary
    _print_summary(all_results)


def _print_summary(all_results: dict):
    header_short = [SHORT[v] for v in VARIANTS]

    print("\n" + "=" * 60)
    print("  Restored Test Set Results (mAP@50)")
    print("=" * 60)
    print(f"{'Model':<14}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (14 + 10 * len(header_short)))
    for m in MODEL_ORDER:
        if m in all_results:
            vals = [all_results[m][v]["mAP50"] for v in VARIANTS]
            print(f"{m:<14}" + "".join(f"{v:>10.4f}" for v in vals))


if __name__ == "__main__":
    main()
