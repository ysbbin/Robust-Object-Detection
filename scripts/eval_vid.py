"""
Video model evaluation: 4 models x 4 test sets = 16 evaluations.

Models:
  vid_yolo   baseline  : YOLOv8m trained on VisDrone-VID (clean)
  vid_yolo   augmented : YOLOv8m trained on VisDrone-VID (corruption aug)
  vid_rtdetr baseline  : RT-DETR-L trained on VisDrone-VID (clean)
  vid_rtdetr augmented : RT-DETR-L trained on VisDrone-VID (corruption aug)

Testsets (same as image eval): Clean, Noise, Blur, LowRes

Usage:
    python -m scripts.eval_vid
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

import torch

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
VARIANTS = ["Test_Clean", "Test_Noise", "Test_Blur", "Test_LowRes"]
SHORT = {
    "Test_Clean": "Clean",
    "Test_Noise": "Noise",
    "Test_Blur":  "Blur",
    "Test_LowRes": "LowRes",
}

CLASS_NAMES = ["pedestrian", "car", "van", "truck", "bus", "motor"]

YOLO_TESTSET_ROOT = Path("data/testsets/yolo6")

CKPTS = {
    "vid_yolo_base": Path("experiments/vid_yolo/baseline/weights/best.pt"),
    "vid_yolo_aug":  Path("experiments/vid_yolo/augmented/weights/best.pt"),
    "vid_rtdetr_base": Path("experiments/vid_rtdetr/baseline/weights/best.pt"),
    "vid_rtdetr_aug":  Path("experiments/vid_rtdetr/augmented/weights/best.pt"),
}

MODEL_ORDER = [
    "vid_yolo_base", "vid_yolo_aug",
    "vid_rtdetr_base", "vid_rtdetr_aug",
]

BASELINE_PAIRS = [
    ("vid_yolo_base",   "vid_yolo_aug"),
    ("vid_rtdetr_base", "vid_rtdetr_aug"),
]

OUT_DIR = Path("experiments")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Evaluation (Ultralytics)
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
        "mAP50":    float(results.box.map50),
        "per_class_ap50": per_class,
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


def _eval_model(name: str, ckpt_path: Path, model_cls: str, all_results: dict):
    print("\n" + "=" * 60)
    print(f"  {name}  (Ultralytics {model_cls})")
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
    print(f"Evaluating 4 video models x {len(VARIANTS)} test sets = {4 * len(VARIANTS)} runs\n")

    all_results = {}
    t0 = time.time()

    _eval_model("vid_yolo_base",   CKPTS["vid_yolo_base"],   "YOLO",   all_results)
    _eval_model("vid_yolo_aug",    CKPTS["vid_yolo_aug"],    "YOLO",   all_results)
    _eval_model("vid_rtdetr_base", CKPTS["vid_rtdetr_base"], "RTDETR", all_results)
    _eval_model("vid_rtdetr_aug",  CKPTS["vid_rtdetr_aug"],  "RTDETR", all_results)

    elapsed = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed/60:.1f} min")

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

    print("\n" + "=" * 60)
    print("  mAP@50 Summary")
    print("=" * 60)
    print(f"{'Model':<20}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (20 + 10 * len(header_short)))
    for m in models:
        vals = [all_results[m][v]["mAP50"] for v in VARIANTS]
        print(f"{m:<20}" + "".join(f"{v:>10.4f}" for v in vals))

    print(f"\n{'Model':<20}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (20 + 10 * len(header_short)))
    for m in models:
        vals = [all_results[m][v]["mAP50_95"] for v in VARIANTS]
        print(f"{m:<20}" + "".join(f"{v:>10.4f}" for v in vals))
    print("  (mAP@50-95)")

    print("\n" + "=" * 60)
    print("  Degradation from Clean (%)")
    print("=" * 60)
    deg_variants = VARIANTS[1:]
    deg_short = [SHORT[v] for v in deg_variants]
    print(f"{'Model':<20}" + "".join(f"{h:>10}" for h in deg_short))
    print("-" * (20 + 10 * len(deg_short)))
    for m in models:
        clean = all_results[m]["Test_Clean"]["mAP50"]
        vals = []
        for v in deg_variants:
            cur = all_results[m][v]["mAP50"]
            pct = (cur - clean) / clean * 100 if clean > 0 else 0.0
            vals.append(pct)
        print(f"{m:<20}" + "".join(f"{v:>9.1f}%" for v in vals))


def _print_comparison(all_results: dict):
    print("\n" + "=" * 60)
    print("  Baseline vs Augmented (mAP@50 difference)")
    print("=" * 60)
    header_short = [SHORT[v] for v in VARIANTS]
    print(f"{'Model':<20}" + "".join(f"{h:>10}" for h in header_short))
    print("-" * (20 + 10 * len(header_short)))

    for base, aug in BASELINE_PAIRS:
        if base not in all_results or aug not in all_results:
            continue
        vals = []
        for v in VARIANTS:
            diff = all_results[aug][v]["mAP50"] - all_results[base][v]["mAP50"]
            vals.append(diff)
        print(f"{base:<20}" + "".join(f"{v:>+10.4f}" for v in vals))


def _save_json(all_results: dict):
    out_path = OUT_DIR / "vid_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {out_path.resolve()}")


def _save_csv(all_results: dict):
    out_path = OUT_DIR / "vid_eval_results.csv"
    models = [m for m in MODEL_ORDER if m in all_results]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["Model", "Metric"] + [SHORT[v] for v in VARIANTS]
        writer.writerow(header)

        for m in models:
            row50 = [m, "mAP@50"]
            row95 = [m, "mAP@50-95"]
            for v in VARIANTS:
                row50.append(f"{all_results[m][v]['mAP50']:.4f}")
                row95.append(f"{all_results[m][v]['mAP50_95']:.4f}")
            writer.writerow(row50)
            writer.writerow(row95)

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
