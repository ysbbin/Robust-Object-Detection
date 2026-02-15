# Corruption Augmentation 학습

## 1. 목적

Baseline 평가에서 확인된 열화 조건(Noise, Blur, LowRes)에서의 성능 하락을 개선하기 위해, **학습 단계에서 동일한 유형의 corruption을 augmentation으로 주입**하여 강건성을 높인다.

핵심 가설: 학습 시 열화된 이미지를 함께 보여주면, 모델이 열화에 강건한 feature를 학습하여 테스트 시 성능 하락이 줄어들 것이다.

---

## 2. Augmentation 설계

### 2-1. Corruption 파라미터

테스트셋 생성(`build_corrupted_testsets.py`)과 **동일한 파라미터**를 사용하여 학습-평가 간 일관성을 유지한다.

| Corruption | 파라미터 | 값 |
|---|---|---|
| Gaussian Noise | sigma | 15 |
| Motion Blur | kernel size, angle | 9, 0 deg |
| Low Resolution | downscale factor | 0.5x |

### 2-2. 적용 방식

- 확률: 각 이미지에 대해 **50% 확률**로 corruption 적용
- 유형 선택: noise, blur, lowres 중 **랜덤 1개** 선택
- 적용 위치: **픽셀 레벨만** (bounding box 좌표는 변경 없음)

### 2-3. 모델별 구현 방식

**Faster R-CNN** (`scripts/train_frcnn_augmented.py`):
- `RandomCorruption` 클래스 (PIL Image transform)
- torchvision transforms 파이프라인에 삽입: `RandomCorruption(p=0.5)` -> `ToImage()` -> `ToDtype()`
- PIL -> numpy BGR -> corruption -> PIL RGB 변환

**RT-DETR / YOLOv8** (`scripts/train_rtdetr_augmented.py`, `scripts/train_yolo_augmented.py`):
- Ultralytics `Albumentations.__call__` 메서드를 **monkey-patching**
- `patch_ultralytics_augmentations()` 함수를 모델 import 전에 호출
- `labels["img"]` (numpy BGR 배열)에 직접 corruption 적용
- 기존 Ultralytics augmentation 파이프라인(mosaic, mixup 등)은 그대로 유지

구현: `scripts/augmentations.py` (공용 모듈)

---

## 3. 학습 설정

### 3-1. Faster R-CNN (Augmented)

| 항목 | 값 |
|---|---|
| Backbone | ResNet-50 FPN v2 (ImageNet pretrained) |
| Epochs | 24 |
| Batch size | 2 |
| Optimizer | SGD (lr=0.005, momentum=0.9, weight_decay=0.0005) |
| LR Scheduler | StepLR (step=8, gamma=0.1) |
| Augmentation | RandomCorruption(p=0.5) + 기본 전처리 |
| 학습 시간 | ~26.3시간 (94,637초) |
| 출력 | `experiments/frcnn/augmented/best.pth` |

### 3-2. RT-DETR-L (Augmented)

| 항목 | 값 |
|---|---|
| Backbone | RT-DETR-L (COCO pretrained) |
| Epochs | 100 |
| Batch size | 2 |
| Image size | 1024 |
| Optimizer | auto (AdamW) |
| Augmentation | Corruption patch + mosaic, fliplr, hsv, erasing 등 |
| 출력 | `experiments/rtdetr/augmented/weights/best.pt` |

### 3-3. YOLOv8m (Augmented)

| 항목 | 값 |
|---|---|
| Backbone | YOLOv8m (COCO pretrained) |
| Epochs | 100 |
| Batch size | 4 |
| Image size | 1024 |
| Optimizer | auto (AdamW) |
| Augmentation | Corruption patch + mosaic, fliplr, hsv, erasing 등 |
| 출력 | `experiments/yolo/augmented/weights/best.pt` |

**공통 사항**: Baseline과 동일한 하이퍼파라미터를 사용하여, 성능 차이가 순수하게 corruption augmentation에 의한 것임을 보장한다.

---

## 4. 학습 곡선 (Faster R-CNN)

```
Epoch   Loss Sum     LR
  1     3198.15     0.005
  4     2552.99     0.005
  7     2426.61     0.005
  8     2410.48     0.0005   <- StepLR 감소
 12     1845.30     0.0005
 16     1687.94     5e-05    <- StepLR 감소
 20     1588.32     5e-05
 24     1566.13     5e-06    <- StepLR 감소 (최종)
```

- Baseline 대비 loss가 전반적으로 높음 (corruption된 이미지가 더 어려운 학습 샘플이기 때문)
- Baseline final loss: 1323.86 vs Augmented final loss: 1566.13
- 이는 예상된 현상이며, loss가 높다고 성능이 나쁜 것이 아님

### Clean Validation 성능

| 학습 조건 | mAP@50 | mAP@50-95 |
|---|---|---|
| Baseline | 0.5308 | 0.3363 |
| Augmented | 0.5408 | 0.3458 |

Augmented 모델이 Clean에서도 오히려 +1.0%p 향상 (과적합 방지 효과).

---

## 5. 관련 파일

| 파일 | 설명 |
|---|---|
| `scripts/augmentations.py` | 공용 corruption augmentation 모듈 |
| `scripts/train_frcnn_augmented.py` | Faster R-CNN 증강 학습 스크립트 |
| `scripts/train_rtdetr_augmented.py` | RT-DETR 증강 학습 스크립트 |
| `scripts/train_yolo_augmented.py` | YOLOv8 증강 학습 스크립트 |
| `experiments/frcnn/augmented/` | FRCNN 증강 학습 결과 |
| `experiments/rtdetr/augmented/` | RT-DETR 증강 학습 결과 |
| `experiments/yolo/augmented/` | YOLOv8 증강 학습 결과 |

---

*학습 수행일: 2026-02-12*
*학습 환경: NVIDIA GeForce RTX 3070 Ti (8GB), Python 3.11, PyTorch 2.5.1, Ultralytics 8.3.209*
