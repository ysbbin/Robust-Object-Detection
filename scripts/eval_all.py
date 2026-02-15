"""
Unified evaluation script: 6 models x 4 test sets = 24 evaluations.

Models (Baseline) : Faster R-CNN, RT-DETR-L, YOLOv8m
Models (Augmented): Faster R-CNN, RT-DETR-L, YOLOv8m
Testsets: Clean, Noise, Blur, LowRes

Usage:
    python -m scripts.eval_all
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

COCO_TESTSET_ROOT = Path("data/testsets/coco6")
YOLO_TESTSET_ROOT = Path("data/testsets/yolo6")

# Checkpoint paths
CKPTS = {
    "FasterRCNN":      Path("experiments/frcnn/baseline_clean/best.pth"),
    "FasterRCNN_aug":  Path("experiments/frcnn/augmented/best.pth"),
    "RT-DETR-L":       Path("experiments/rtdetr/baseline_clean/weights/best.pt"),
    "RT-DETR-L_aug":   Path("experiments/rtdetr/augmented/weights/best.pt"),
    "YOLOv8m":         Path("experiments/yolo/baseline_clean/weights/best.pt"),
    "YOLOv8m_aug":     Path("experiments/yolo/augmented/weights/best.pt"),
}

# Model display order
MODEL_ORDER = [
    "FasterRCNN", "FasterRCNN_aug",
    "RT-DETR-L", "RT-DETR-L_aug",
    "YOLOv8m", "YOLOv8m_aug",
]

# Baseline pairs for comparison
BASELINE_PAIRS = [
    ("FasterRCNN", "FasterRCNN_aug"),
    ("RT-DETR-L", "RT-DETR-L_aug"),
    ("YOLOv8m", "YOLOv8m_aug"),
]

OUT_DIR = Path("experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Faster R-CNN evaluation
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
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])


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
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })

    if len(results) == 0:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "per_class_ap50": {}}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    per_class_ap50 = _extract_per_class_ap50(coco_eval, coco_gt)

    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "per_class_ap50": per_class_ap50,
    }


def _extract_per_class_ap50(coco_eval: COCOeval, coco_gt: COCO) -> dict:
    precision = coco_eval.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    per_class = {}
    for k_idx, cat_id in enumerate(cat_ids):
        cat_info = coco_gt.loadCats(cat_id)[0]
        cat_name = cat_info["name"]
        ap50 = precision[0, :, k_idx, 0, 2]
        ap50 = ap50[ap50 > -1]
        per_class[cat_name] = float(np.mean(ap50)) if len(ap50) > 0 else 0.0
    return per_class


# ──────────────────────────────────────────────
# RT-DETR / YOLOv8 evaluation (Ultralytics)
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

    metrics = {
        "mAP50_95": float(results.box.map),
        "mAP50": float(results.box.map50),
        "per_class_ap50": per_class,
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


# ──────────────────────────────────────────────
# Generic evaluation runners
# ──────────────────────────────────────────────
def _eval_frcnn_model(name: str, ckpt_path: Path, all_results: dict):
    print("=" * 60)
    print(f"  {name}  (pycocotools COCOeval)")
    print("=" * 60)
    model = _build_frcnn(ckpt_path)
    all_results[name] = {}

    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_frcnn_variant(model, v)
        all_results[name][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


def _eval_ultra_model(name: str, ckpt_path: Path, model_cls: str, all_results: dict):
    print("\n" + "=" * 60)
    print(f"  {name}  (Ultralytics)")
    print("=" * 60)
    all_results[name] = {}

    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_ultralytics(str(ckpt_path), model_cls, v)
        all_results[name][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Evaluating 6 models x {len(VARIANTS)} test sets = {6 * len(VARIANTS)} runs\n")

    all_results = {}
    t0 = time.time()

    # Baseline models
    _eval_frcnn_model("FasterRCNN", CKPTS["FasterRCNN"], all_results)
    _eval_ultra_model("RT-DETR-L", CKPTS["RT-DETR-L"], "RTDETR", all_results)
    _eval_ultra_model("YOLOv8m", CKPTS["YOLOv8m"], "YOLO", all_results)

    # Augmented models
    _eval_frcnn_model("FasterRCNN_aug", CKPTS["FasterRCNN_aug"], all_results)
    _eval_ultra_model("RT-DETR-L_aug", CKPTS["RT-DETR-L_aug"], "RTDETR", all_results)
    _eval_ultra_model("YOLOv8m_aug", CKPTS["YOLOv8m_aug"], "YOLO", all_results)

    elapsed = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed/60:.1f} min")

    # Summary
    _print_summary(all_results)
    _print_comparison(all_results)
    _save_json(all_results)
    _save_csv(all_results)


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────
def _print_summary(all_results: dict):
    models = [m for m in MODEL_ORDER if m in all_results]
    header_short = [SHORT[v] for v in VARIANTS]

    # mAP@50 table
    print("\n" + "=" * 60)
    print("  mAP@50 Summary")
    print("=" * 60)
    print(f"{'Model':<18}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (18 + 10 * len(header_short)))
    for m in models:
        vals = [all_results[m][v]["mAP50"] for v in VARIANTS]
        print(f"{m:<18}" + "".join(f"{v:>10.4f}" for v in vals))

    # mAP@50-95 table
    print(f"\n{'Model':<18}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (18 + 10 * len(header_short)))
    for m in models:
        vals = [all_results[m][v]["mAP50_95"] for v in VARIANTS]
        print(f"{m:<18}" + "".join(f"{v:>10.4f}" for v in vals))
    print("  (mAP@50-95)")

    # Degradation table
    print("\n" + "=" * 60)
    print("  Degradation from Clean (%)")
    print("=" * 60)
    deg_variants = VARIANTS[1:]
    deg_short = [SHORT[v] for v in deg_variants]
    print(f"{'Model':<18}" + "".join(f"{h:>10}" for h in deg_short))
    print("-" * (18 + 10 * len(deg_short)))
    for m in models:
        clean = all_results[m]["Test_Clean"]["mAP50"]
        vals = []
        for v in deg_variants:
            cur = all_results[m][v]["mAP50"]
            pct = (cur - clean) / clean * 100 if clean > 0 else 0.0
            vals.append(pct)
        print(f"{m:<18}" + "".join(f"{v:>9.1f}%" for v in vals))


def _print_comparison(all_results: dict):
    """Print Baseline vs Augmented comparison."""
    print("\n" + "=" * 60)
    print("  Baseline vs Augmented (mAP@50 difference)")
    print("=" * 60)
    header_short = [SHORT[v] for v in VARIANTS]
    print(f"{'Model':<14}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (14 + 10 * len(header_short)))

    for base, aug in BASELINE_PAIRS:
        if base not in all_results or aug not in all_results:
            continue
        short_name = base.replace("Faster", "F")
        vals = []
        for v in VARIANTS:
            diff = all_results[aug][v]["mAP50"] - all_results[base][v]["mAP50"]
            vals.append(diff)
        print(f"{short_name:<14}" + "".join(f"{v:>+10.4f}" for v in vals))


def _save_json(all_results: dict):
    out_path = OUT_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {out_path.resolve()}")


def _save_csv(all_results: dict):
    out_path = OUT_DIR / "eval_results.csv"
    models = [m for m in MODEL_ORDER if m in all_results]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["Model", "Metric"]
        for v in VARIANTS:
            header.append(SHORT[v])
        writer.writerow(header)

        # mAP rows
        for m in models:
            row50 = [m, "mAP@50"]
            row95 = [m, "mAP@50-95"]
            for v in VARIANTS:
                row50.append(f"{all_results[m][v]['mAP50']:.4f}")
                row95.append(f"{all_results[m][v]['mAP50_95']:.4f}")
            writer.writerow(row50)
            writer.writerow(row95)

        # Degradation rows
        writer.writerow([])
        writer.writerow(["Model", "Metric"] + [SHORT[v] for v in VARIANTS[1:]])
        for m in models:
            clean = all_results[m]["Test_Clean"]["mAP50"]
            row = [m, "Deg%_mAP50"]
            for v in VARIANTS[1:]:
                cur = all_results[m][v]["mAP50"]
                pct = (cur - clean) / clean * 100 if clean > 0 else 0.0
                row.append(f"{pct:.1f}%")
            writer.writerow(row)

        # Baseline vs Augmented diff
        writer.writerow([])
        writer.writerow(["Model", "Metric"] + [SHORT[v] for v in VARIANTS])
        for base, aug in BASELINE_PAIRS:
            if base not in all_results or aug not in all_results:
                continue
            row = [base, "Aug-Base_mAP50"]
            for v in VARIANTS:
                diff = all_results[aug][v]["mAP50"] - all_results[base][v]["mAP50"]
                row.append(f"{diff:+.4f}")
            writer.writerow(row)

    print(f"CSV  saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
