"""
Unified evaluation script: 3 models x 4 test sets = 12 evaluations.

Models : Faster R-CNN, RT-DETR-L, YOLOv8m
Testsets: Clean, Noise, Blur, LowRes

Usage:
    python -m scripts.eval_all
"""

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

CLASS_NAMES = ["pedestrian", "car", "van", "truck", "bus", "motor"]  # 0-indexed for YOLO, 1-indexed for COCO

COCO_TESTSET_ROOT = Path("data/testsets/coco6")
YOLO_TESTSET_ROOT = Path("data/testsets/yolo6")

FRCNN_CKPT = Path("experiments/frcnn/baseline_clean/best.pth")
RTDETR_CKPT = Path("experiments/rtdetr/baseline_clean/weights/best.pt")
YOLO_CKPT = Path("experiments/yolo/baseline_clean/weights/best.pt")

OUT_DIR = Path("experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Faster R-CNN evaluation
# ──────────────────────────────────────────────
def _build_frcnn():
    """Load Faster R-CNN with the trained VisDrone head."""
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)

    ckpt = torch.load(FRCNN_CKPT, map_location="cpu", weights_only=True)
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
    """Run FRCNN on one COCO-format test set and return metrics."""
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

    # Per-class AP50
    per_class_ap50 = _extract_per_class_ap50(coco_eval, coco_gt)

    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "per_class_ap50": per_class_ap50,
    }


def _extract_per_class_ap50(coco_eval: COCOeval, coco_gt: COCO) -> dict:
    """Extract per-class AP@50 from COCOeval precision array."""
    # precision shape: [T, R, K, A, M]
    # T=IoU thresholds, R=recall thresholds, K=categories, A=areas, M=maxDets
    # IoU=0.5 is index 0, area=all is index 0, maxDets=100 is index 2
    precision = coco_eval.eval["precision"]  # (10, 101, K, 4, 3)
    cat_ids = coco_gt.getCatIds()
    per_class = {}
    for k_idx, cat_id in enumerate(cat_ids):
        cat_info = coco_gt.loadCats(cat_id)[0]
        cat_name = cat_info["name"]
        # AP@50: iou_idx=0, all recall, category k_idx, area=all(0), maxDets=100(2)
        ap50 = precision[0, :, k_idx, 0, 2]
        ap50 = ap50[ap50 > -1]
        per_class[cat_name] = float(np.mean(ap50)) if len(ap50) > 0 else 0.0
    return per_class


# ──────────────────────────────────────────────
# RT-DETR / YOLOv8 evaluation (Ultralytics)
# ──────────────────────────────────────────────
def _eval_ultralytics(model_path: str, model_cls: str, variant: str) -> dict:
    """
    Evaluate an Ultralytics model (YOLO or RTDETR) on a YOLO-format test set.
    """
    from ultralytics import YOLO, RTDETR

    data_yaml = str(YOLO_TESTSET_ROOT / variant / "data.yaml")

    if model_cls == "RTDETR":
        model = RTDETR(model_path)
        # Workaround: Ultralytics pickles the whole model, so cached
        # valid_mask/anchors from training sit on CPU.  Resetting shapes
        # forces _generate_anchors to run on the correct device.
        for m in model.model.modules():
            if hasattr(m, "valid_mask") and hasattr(m, "shapes"):
                m.shapes = []
    else:
        model = YOLO(model_path)

    results = model.val(data=data_yaml, imgsz=1024, batch=1, device=0, verbose=False)

    # Per-class AP50 — results.box.ap50 is a numpy array, one per class
    ap50_arr = results.box.ap50
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = float(ap50_arr[i]) if i < len(ap50_arr) else 0.0

    metrics = {
        "mAP50_95": float(results.box.map),
        "mAP50": float(results.box.map50),
        "per_class_ap50": per_class,
    }

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Evaluating 3 models x {len(VARIANTS)} test sets = {3 * len(VARIANTS)} runs\n")

    all_results = {}  # {model_name: {variant: metrics_dict}}
    t0 = time.time()

    # ── 1. Faster R-CNN ──────────────────────
    print("=" * 60)
    print("  Faster R-CNN  (pycocotools COCOeval)")
    print("=" * 60)
    frcnn = _build_frcnn()
    all_results["FasterRCNN"] = {}

    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_frcnn_variant(frcnn, v)
        all_results["FasterRCNN"][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")

    del frcnn
    torch.cuda.empty_cache()
    gc.collect()

    # ── 2. RT-DETR-L ─────────────────────────
    print("\n" + "=" * 60)
    print("  RT-DETR-L  (Ultralytics)")
    print("=" * 60)
    all_results["RT-DETR-L"] = {}

    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_ultralytics(str(RTDETR_CKPT), "RTDETR", v)
        all_results["RT-DETR-L"][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")

    # ── 3. YOLOv8m ───────────────────────────
    print("\n" + "=" * 60)
    print("  YOLOv8m  (Ultralytics)")
    print("=" * 60)
    all_results["YOLOv8m"] = {}

    for v in VARIANTS:
        print(f"\n  [{SHORT[v]}] evaluating ...", flush=True)
        metrics = _eval_ultralytics(str(YOLO_CKPT), "YOLO", v)
        all_results["YOLOv8m"][v] = metrics
        print(f"  [{SHORT[v]}] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed/60:.1f} min")

    # ── Summary tables ───────────────────────
    _print_summary(all_results)
    _save_json(all_results)
    _save_csv(all_results)


def _print_summary(all_results: dict):
    models = list(all_results.keys())
    header_short = [SHORT[v] for v in VARIANTS]

    # mAP@50 table
    print("\n" + "=" * 60)
    print("  mAP@50 Summary")
    print("=" * 60)
    print(f"{'Model':<14}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (14 + 10 * len(header_short)))
    for m in models:
        vals = [all_results[m][v]["mAP50"] for v in VARIANTS]
        print(f"{m:<14}" + "".join(f"{v:>10.4f}" for v in vals))

    # mAP@50-95 table
    print(f"\n{'Model':<14}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (14 + 10 * len(header_short)))
    for m in models:
        vals = [all_results[m][v]["mAP50_95"] for v in VARIANTS]
        print(f"{m:<14}" + "".join(f"{v:>10.4f}" for v in vals))
    print("  (mAP@50-95)")

    # Degradation table
    print("\n" + "=" * 60)
    print("  Degradation from Clean (%)")
    print("=" * 60)
    deg_variants = VARIANTS[1:]  # skip Clean
    deg_short = [SHORT[v] for v in deg_variants]
    print(f"{'Model':<14}" + "".join(f"{h:>10}" for h in deg_short))
    print("-" * (14 + 10 * len(deg_short)))
    for m in models:
        clean = all_results[m]["Test_Clean"]["mAP50"]
        vals = []
        for v in deg_variants:
            cur = all_results[m][v]["mAP50"]
            if clean > 0:
                pct = (cur - clean) / clean * 100
            else:
                pct = 0.0
            vals.append(pct)
        print(f"{m:<14}" + "".join(f"{v:>9.1f}%" for v in vals))


def _save_json(all_results: dict):
    out_path = OUT_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {out_path.resolve()}")


def _save_csv(all_results: dict):
    out_path = OUT_DIR / "eval_results.csv"
    models = list(all_results.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["Model", "Metric"]
        for v in VARIANTS:
            header.append(SHORT[v])
        writer.writerow(header)

        # Rows
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

    print(f"CSV  saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
