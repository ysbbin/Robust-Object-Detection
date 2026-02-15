"""
Faster R-CNN augmented training (corruption augmentation).

Identical to train_frcnn_baseline.py except:
  - Training transforms include RandomCorruption (noise/blur/lowres, p=0.5)
  - Output directory: experiments/frcnn/augmented

Usage:
    python -m scripts.train_frcnn_augmented
"""

import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import time
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from scripts.coco_detection_dataset import COCODetectionDataset, collate_fn
from scripts.augmentations import RandomCorruption

# ====== Hyperparameters (same as baseline) ======
SEED = 42
EPOCHS = 24
BATCH_SIZE = 2
LR = 0.005
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

DATA_ROOT = Path("data/processed/visdrone_coco6")
TRAIN_IMG = DATA_ROOT / "images/train"
VAL_IMG = DATA_ROOT / "images/val"
TRAIN_ANN = DATA_ROOT / "annotations/instances_train.json"
VAL_ANN = DATA_ROOT / "annotations/instances_val.json"

OUT_DIR = Path("experiments/frcnn/augmented")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(train: bool):
    if train:
        return T.Compose([
            RandomCorruption(p=0.5),  # <-- corruption augmentation
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
    else:
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])


@torch.no_grad()
def evaluate_coco(model, data_loader, ann_file: str, device: torch.device):
    model.eval()
    coco_gt = COCO(ann_file)
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
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
        return {"mAP50_95": 0.0, "mAP50": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
    }


def save_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    print("Mode: AUGMENTED training (corruption p=0.5)\n", flush=True)

    # Dataset
    train_ds = COCODetectionDataset(str(TRAIN_IMG), str(TRAIN_ANN), transforms=build_transforms(train=True))
    val_ds = COCODetectionDataset(str(VAL_IMG), str(VAL_ANN), transforms=build_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    # Model
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # Logging
    history_path = OUT_DIR / "history.jsonl"
    best_ckpt = OUT_DIR / "best.pth"
    last_ckpt = OUT_DIR / "last.pth"

    start_time = time.time()

    n_batches = len(train_loader)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            epoch_loss += float(losses.item())

            if (i + 1) % 100 == 0 or (i + 1) == n_batches:
                print(f"  [Epoch {epoch:03d}] batch {i+1}/{n_batches}", flush=True)

        lr_scheduler.step()

        log = {
            "epoch": epoch,
            "train_loss_sum": epoch_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "mAP50": None,
            "mAP50_95": None,
            "elapsed_sec": int(time.time() - start_time),
        }
        save_jsonl(history_path, log)
        print(f"[Epoch {epoch:03d}/{EPOCHS}] loss_sum={epoch_loss:.4f}", flush=True)

        torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt)

    # Final evaluation
    print("\nEvaluating on clean val set (final)...", flush=True)
    final_metrics = evaluate_coco(model, val_loader, str(VAL_ANN), device)
    print(
        f"Final Augmented | mAP50={final_metrics['mAP50']:.4f} "
        f"mAP50-95={final_metrics['mAP50_95']:.4f}",
        flush=True,
    )

    torch.save({"model": model.state_dict(), "epoch": "final", "metrics": final_metrics}, best_ckpt)

    final_log = {
        "epoch": "final",
        "train_loss_sum": None,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "mAP50": final_metrics["mAP50"],
        "mAP50_95": final_metrics["mAP50_95"],
        "elapsed_sec": int(time.time() - start_time),
    }
    save_jsonl(history_path, final_log)

    print("\nTraining done.", flush=True)
    print("Best checkpoint:", best_ckpt.resolve(), flush=True)


if __name__ == "__main__":
    main()
