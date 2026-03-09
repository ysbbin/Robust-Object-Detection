# VisDrone-VID 비디오 모델 실험

## 1. 실험 개요

### 목적

이미지 탐지(DET) 실험에서 검증된 Corruption Augmentation 전략이 **비디오 데이터**에서도 동일하게 효과적인지 확인한다. VisDrone-VID(비디오 시퀀스) 데이터로 YOLOv8m, RT-DETR-L을 학습하고, 동일한 열화 테스트셋으로 평가하여 DET 결과와 비교한다.

### 실험 조건

| 항목 | 내용 |
|------|------|
| 데이터셋 | VisDrone-VID (드론 시점 비디오 시퀀스) |
| 클래스 수 | 6개 (pedestrian, car, van, truck, bus, motor) |
| 포맷 | YOLO (프레임 단위 이미지로 변환) |
| 모델 | YOLOv8m, RT-DETR-L |
| 전략 | Baseline (clean 학습), Augmented (corruption augmentation 학습) |
| 테스트셋 | DET 실험과 동일 (yolo6/Test_Clean, Noise, Blur, LowRes) |
| 평가 지표 | mAP@50, mAP@50-95, 클래스별 AP@50 |
| 총 평가 횟수 | 4 모델 x 4 테스트셋 = 16회 |
| 학습 스크립트 | `scripts/train_vid_yolo_baseline.py`, `train_vid_yolo_augmented.py`, `train_vid_rtdetr_baseline.py`, `train_vid_rtdetr_augmented.py` |
| 평가 스크립트 | `scripts/eval_vid.py` |
| 시각화 스크립트 | `scripts/plot_vid_results.py` |
| 결과 파일 | `experiments/vid_eval_results.json`, `experiments/vid_eval_results.csv` |

---

## 2. 데이터셋 준비

### 2-1. VisDrone-VID 소개

| 항목 | 내용 |
|------|------|
| 데이터 | 드론 촬영 비디오 시퀀스 |
| 원본 어노테이션 형식 | `frame_index, track_id, x, y, w, h, score, category, truncation, occlusion` |
| 사용 분할 | train / val |
| 클래스 | DET 실험과 동일한 6개 클래스 사용 (category ID: 1, 4, 5, 6, 9, 10) |

### 2-2. YOLO 포맷 변환

변환 스크립트: `scripts/convert_visdrone_vid_to_yolo.py`

- 비디오 시퀀스의 각 프레임을 독립적인 이미지 파일로 추출
- 파일명 형식: `{시퀀스명}_{프레임ID:07d}.jpg`
- 어노테이션을 YOLO 형식(정규화 좌표)으로 변환
- 출력: `data/processed/visdrone_vid_yolo6/`

```
data/processed/visdrone_vid_yolo6/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

---

## 3. 학습 설정

### 3-1. YOLOv8m Baseline

| 항목 | 값 |
|------|-----|
| 사전학습 가중치 | YOLOv8m (COCO pretrained) |
| 데이터 | VisDrone-VID YOLO6 (clean) |
| Epochs | 100 |
| Batch size | 4 |
| Image size | 1024 |
| Optimizer | auto (AdamW) |
| patience | 100 (early stopping 없음) |
| AMP | True |
| 출력 | `experiments/vid_yolo/baseline/weights/best.pt` |
| 최종 val mAP@50 | 0.357 (마지막 epoch) |

### 3-2. YOLOv8m Augmented

| 항목 | 값 |
|------|-----|
| 사전학습 가중치 | YOLOv8m (COCO pretrained) |
| 데이터 | VisDrone-VID YOLO6 (clean + corruption augmentation) |
| Epochs | 100 |
| Batch size | 4 |
| Image size | 1024 |
| Augmentation | `patch_ultralytics_augmentations()` — 50% 확률로 Noise/Blur/LowRes 중 랜덤 1개 적용 |
| 출력 | `experiments/vid_yolo/augmented/weights/best.pt` |
| 최종 val mAP@50 | 0.342 (마지막 epoch) |

### 3-3. RT-DETR-L Baseline

| 항목 | 값 |
|------|-----|
| 사전학습 가중치 | RT-DETR-L (COCO pretrained) |
| 데이터 | VisDrone-VID YOLO6 (clean) |
| Epochs | 100 |
| Batch size | 2 |
| Image size | 1024 |
| Optimizer | auto (AdamW) |
| patience | 100 (early stopping 없음) |
| AMP | True |
| 출력 | `experiments/vid_rtdetr/baseline/weights/best.pt` |
| 최종 val mAP@50 | 0.346 (마지막 epoch) |

### 3-4. RT-DETR-L Augmented

| 항목 | 값 |
|------|-----|
| 사전학습 가중치 | RT-DETR-L (COCO pretrained) |
| 데이터 | VisDrone-VID YOLO6 (clean + corruption augmentation) |
| Epochs | 100 |
| Batch size | 2 |
| Image size | 1024 |
| Augmentation | `patch_ultralytics_augmentations()` — 50% 확률로 Noise/Blur/LowRes 중 랜덤 1개 적용 |
| 출력 | `experiments/vid_rtdetr/augmented/weights/best.pt` |
| 최종 val mAP@50 | 0.335 (마지막 epoch) |

**공통 사항**: Baseline과 Augmented 간 비교의 순수성을 위해, Augmentation 방식 외 모든 하이퍼파라미터를 동일하게 유지한다.

---

## 4. 평가 방법

### 4-1. 테스트셋

DET 실험에서 사용한 동일한 테스트셋을 재사용한다.

| 테스트셋 | 경로 | 설명 |
|----------|------|------|
| Clean | `data/testsets/yolo6/Test_Clean` | 원본 이미지 |
| Noise | `data/testsets/yolo6/Test_Noise` | 가우시안 노이즈 (sigma=15) |
| Blur | `data/testsets/yolo6/Test_Blur` | 모션 블러 (kernel=9) |
| LowRes | `data/testsets/yolo6/Test_LowRes` | 저해상도 (0.5x 축소 후 복원) |

### 4-2. 평가 방식

Ultralytics `model.val()` API 사용 (imgsz=1024, batch=1, device=0)

---

## 5. 정량 결과

### 5-1. mAP@50

| Model | Clean | Noise | Blur | LowRes |
|-------|------:|------:|-----:|-------:|
| vid_yolo_base | 0.387 | 0.330 | 0.239 | 0.309 |
| vid_yolo_aug | **0.409** | **0.391** | **0.372** | **0.385** |
| vid_rtdetr_base | 0.316 | 0.287 | 0.218 | 0.265 |
| vid_rtdetr_aug | **0.335** | **0.319** | **0.320** | **0.310** |

### 5-2. mAP@50-95

| Model | Clean | Noise | Blur | LowRes |
|-------|------:|------:|-----:|-------:|
| vid_yolo_base | 0.220 | 0.191 | 0.125 | 0.172 |
| vid_yolo_aug | **0.230** | **0.219** | **0.212** | **0.216** |
| vid_rtdetr_base | 0.162 | 0.149 | 0.106 | 0.138 |
| vid_rtdetr_aug | **0.166** | **0.159** | **0.161** | **0.156** |

### 5-3. 성능 하락률 (Clean 대비 mAP@50 기준)

| Model | Noise | Blur | LowRes |
|-------|------:|-----:|-------:|
| vid_yolo_base | -14.7% | **-38.3%** | -20.3% |
| vid_yolo_aug | -4.3% | -9.1% | -5.8% |
| vid_rtdetr_base | -8.9% | **-30.9%** | -15.9% |
| vid_rtdetr_aug | -4.7% | **-4.5%** | -7.4% |

### 5-4. Augmentation 개선 효과 (Aug - Base, mAP@50)

| Model | Clean | Noise | Blur | LowRes |
|-------|------:|------:|-----:|-------:|
| YOLOv8m | +0.022 | +0.061 | **+0.133** | +0.077 |
| RT-DETR-L | +0.019 | +0.032 | **+0.102** | +0.045 |

### 5-5. 클래스별 AP@50 (Clean)

| Class | yolo_base | yolo_aug | rtdetr_base | rtdetr_aug |
|-------|----------:|---------:|------------:|-----------:|
| pedestrian | 0.344 | 0.370 | 0.240 | 0.244 |
| car | 0.720 | 0.735 | 0.589 | 0.603 |
| van | 0.353 | 0.365 | 0.293 | 0.323 |
| truck | 0.246 | 0.281 | 0.234 | 0.238 |
| bus | 0.403 | 0.423 | 0.334 | 0.391 |
| motor | 0.256 | 0.281 | 0.204 | 0.211 |

### 5-6. 클래스별 AP@50 (Blur — 가장 극적인 조건)

| Class | yolo_base | yolo_aug | rtdetr_base | rtdetr_aug |
|-------|----------:|---------:|------------:|-----------:|
| pedestrian | 0.139 | 0.302 | 0.119 | 0.208 |
| car | 0.528 | 0.718 | 0.495 | 0.599 |
| van | 0.195 | 0.328 | 0.206 | 0.320 |
| truck | 0.147 | 0.244 | 0.116 | 0.216 |
| bus | 0.322 | 0.396 | 0.267 | 0.385 |
| motor | 0.103 | 0.240 | 0.106 | 0.191 |

---

## 6. 핵심 해석

### 6-1. Blur가 비디오 모델에서도 가장 치명적

이미지 탐지 실험과 동일하게, Blur 조건이 가장 큰 성능 하락을 유발한다.

- vid_yolo_base: Blur 시 -38.3% (DET YOLOv8m baseline: -35.1%)
- vid_rtdetr_base: Blur 시 -30.9% (DET RT-DETR baseline: -26.0%)

비디오 프레임은 빠른 움직임으로 인한 모션 블러가 실환경에서 더 빈번하므로, 이 결과는 실용적으로 중요하다.

### 6-2. Corruption Augmentation 효과가 비디오에서도 일관되게 유효

| 조건 | YOLOv8m (DET) | YOLOv8m (VID) |
|------|-------------:|-------------:|
| Blur 하락률 (baseline) | -35.1% | -38.3% |
| Blur 하락률 (augmented) | -7.9% | -9.1% |
| Blur 개선 효과 | +0.175 mAP50 | +0.133 mAP50 |

학습 도메인(이미지 vs 비디오)이 달라져도 Corruption Augmentation은 유사한 수준의 강건성 향상을 제공한다.

### 6-3. RT-DETR-L_aug의 Blur 강건성이 특히 인상적

vid_rtdetr_aug는 Blur 조건에서 단 **-4.5%** 하락에 그친다. 이는 4개 모델 중 가장 낮은 하락률이다.

- Clean → Blur: 0.335 → 0.320 (고작 0.015 차이)
- Transformer의 전역 어텐션(global attention)이 부분적인 열화에도 전체 맥락으로 보완 가능함을 시사

### 6-4. YOLOv8m의 절대 성능 우위 유지

모든 조건에서 YOLOv8m이 RT-DETR-L보다 절대 mAP가 높다. 이는 DET 실험 결과와 동일한 패턴이다.

- 단, RT-DETR_aug의 Blur 강건성(하락률 -4.5%)은 YOLOv8m_aug(-9.1%)보다 우수
- 배포 시나리오에 따라 선택 기준이 달라짐: 절대 성능 → YOLO, Blur 강건성 → RT-DETR

### 6-5. LowRes에서 YOLOv8m_aug의 높은 개선 효과

vid_yolo_aug의 LowRes 개선폭이 +0.077 mAP50로, RT-DETR(+0.045)보다 크다.

- YOLOv8m baseline의 LowRes 하락률(-20.3%)이 RT-DETR(-15.9%)보다 크기 때문
- Augmentation 적용 후 격차가 크게 줄어들어, 두 모델 모두 LowRes에서 -5~-7% 수준으로 수렴

### 6-6. 클래스별 패턴

- **Car**: 모든 조건에서 가장 높은 AP (Clean 기준 0.589~0.735). 크기가 크고 형태가 일정하여 열화에도 상대적으로 강건
- **Pedestrian, Motor**: Clean에서도 낮고 Blur 시 급락. 소형 객체이며 비디오 시퀀스에서 움직임이 많아 블러 취약성이 증가
- **Truck**: 비디오 데이터에서도 탐지 난이도가 높음 (Clean AP 0.234~0.281)

---

## 7. DET vs VID 비교

### YOLOv8m Baseline 비교

| 조건 | DET mAP50 | VID mAP50 | 차이 |
|------|----------:|----------:|-----:|
| Clean | 0.666 | 0.387 | -0.279 |
| Noise | 0.577 | 0.330 | -0.247 |
| Blur | 0.432 | 0.239 | -0.193 |
| LowRes | 0.628 | 0.309 | -0.319 |

VID 모델의 절대 성능이 DET보다 낮은 이유:
1. VisDrone-VID 데이터는 DET보다 밀도가 높고 객체가 더 작음
2. 비디오 시퀀스 특성상 빠른 움직임, 흔들림 등 추가 도전 요소 존재
3. 동일 테스트셋(DET 기반)으로 평가하므로, 학습 도메인 불일치(train-test domain gap)도 영향

### Augmentation 효과 비교

| 모델 | DET Blur 개선 | VID Blur 개선 |
|------|-------------:|-------------:|
| YOLOv8m | +0.175 mAP50 | +0.133 mAP50 |
| RT-DETR-L | +0.127 mAP50 | +0.102 mAP50 |

절대 개선폭은 DET가 크지만, **상대적 개선 비율**은 VID에서도 유사하다.

---

## 8. 종합 순위 (VID 실험 기준)

| 관점 | 1위 | 2위 |
|------|-----|-----|
| 절대 성능 (mAP@50) | YOLOv8m_aug | YOLOv8m_base |
| Blur 강건성 (하락률) | RT-DETR_aug (-4.5%) | YOLOv8m_aug (-9.1%) |
| Noise 강건성 (하락률) | YOLOv8m_aug (-4.3%) | RT-DETR_aug (-4.7%) |

---

## 9. 관련 파일

| 파일 | 설명 |
|------|------|
| `scripts/convert_visdrone_vid_to_yolo.py` | VisDrone-VID → YOLO 포맷 변환 |
| `scripts/train_vid_yolo_baseline.py` | YOLOv8m VID baseline 학습 |
| `scripts/train_vid_yolo_augmented.py` | YOLOv8m VID augmented 학습 |
| `scripts/train_vid_rtdetr_baseline.py` | RT-DETR-L VID baseline 학습 |
| `scripts/train_vid_rtdetr_augmented.py` | RT-DETR-L VID augmented 학습 |
| `scripts/eval_vid.py` | 비디오 모델 평가 (16회) |
| `scripts/plot_vid_results.py` | 비디오 결과 시각화 |
| `experiments/vid_yolo/` | YOLOv8m VID 학습 결과 |
| `experiments/vid_rtdetr/` | RT-DETR-L VID 학습 결과 |
| `experiments/vid_eval_results.json` | 평가 결과 (JSON) |
| `experiments/vid_eval_results.csv` | 평가 결과 (CSV) |
| `experiments/figures/vid_*.png` | 시각화 결과 (5개 차트) |

---

*학습 수행일: 2026-03-08*
*평가 수행일: 2026-03-08*
*학습 환경: NVIDIA GeForce RTX 3070 Ti (8GB), Python 3.11, PyTorch 2.5.1, Ultralytics 8.x*
