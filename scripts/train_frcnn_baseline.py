import os
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


# ====== 동일 실험 조건(핵심) ======
SEED = 42
IMG_MAX = 1024  # YOLO imgsz=1024와 최대한 맞추는 목표치
EPOCHS = 24     # Faster R-CNN strong baseline 권장(2x). YOLO처럼 100으로 맞추고 싶으면 여기만 변경
BATCH_SIZE = 2  # 3070Ti 8GB 기준 시작값. OOM 없으면 3~4까지 올려볼 수 있음.
LR = 0.005
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

DATA_ROOT = Path("data/processed/visdrone_coco6")
TRAIN_IMG = DATA_ROOT / "images/train"
VAL_IMG = DATA_ROOT / "images/val"
TRAIN_ANN = DATA_ROOT / "annotations/instances_train.json"
VAL_ANN = DATA_ROOT / "annotations/instances_val.json"

OUT_DIR = Path("experiments/frcnn/baseline_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # (옵션) 재현성 더 강하게
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(train: bool):
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),  # 0~1 float로만 변환
    ])



@torch.no_grad()
def evaluate_coco(model, data_loader, ann_file: str, device: torch.device):
    """
    COCOeval 기반 mAP 측정 (Clean val baseline 및 이후 corrupted val 평가에 사용 가능)
    """
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

            # COCO expects bbox in [x, y, w, h]
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })

    # 예측이 0개면 mAP 0
    if len(results) == 0:
        return {"mAP50_95": 0.0, "mAP50": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats: [0]=mAP@[.5:.95], [1]=mAP@.5
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

    # Dataset
    train_ds = COCODetectionDataset(str(TRAIN_IMG), str(TRAIN_ANN), transforms=build_transforms(train=True))
    val_ds = COCODetectionDataset(str(VAL_IMG), str(VAL_ANN), transforms=build_transforms(train=False))

    # ✅ Windows 안정화: num_workers=0
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    ## Model (COCO pretrained backbone) + VisDrone 6 classes head
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")  # COCO pretrained (backbone/FPN)

    # ✅ head 교체: 배경 1 + 클래스 6 = 7
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)

    model.to(device)


    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # Logging / checkpoints
    history_path = OUT_DIR / "history.jsonl"
    best_ckpt = OUT_DIR / "best.pth"
    last_ckpt = OUT_DIR / "last.pth"

    start_time = time.time()

    # ====== Train loop (epoch 중에는 COCOeval 수행 X) ======
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            epoch_loss += float(losses.item())

        lr_scheduler.step()

        # ✅ epoch 기록: loss만 남김 (평가 지연/종료 이슈 방지)
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

        # ✅ last checkpoint 저장 (metrics는 None으로 저장해도 됨)
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt)

    # ====== Final baseline evaluation (딱 1번만 수행) ======
    print("\nEvaluating on clean val set (final baseline)...", flush=True)
    final_metrics = evaluate_coco(model, val_loader, str(VAL_ANN), device)
    print(
        f"Final Clean Baseline | mAP50={final_metrics['mAP50']:.4f} "
        f"mAP50-95={final_metrics['mAP50_95']:.4f}",
        flush=True
    )

    # best checkpoint는 "final baseline" 기준으로 저장
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
    print("Last checkpoint:", last_ckpt.resolve(), flush=True)
    print("History:", history_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
