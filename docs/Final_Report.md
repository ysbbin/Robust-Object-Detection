# Robust Object Detection under Image Degradation
## VisDrone-DET 기반 객체 탐지 모델 강건성 분석 및 개선

---

## 1. 프로젝트 개요

### 1.1 연구 배경 및 동기

드론, 항공기, 감시 카메라 등의 실환경에서는 카메라 진동(blur), 센서 노이즈(noise), 저해상도(low resolution) 등 다양한 이미지 열화(corruption)가 불가피하게 발생한다. 이러한 열화 조건에서 객체 탐지 모델의 성능이 얼마나 저하되는지, 그리고 이를 개선하기 위한 전략이 무엇인지를 실험적으로 검증하는 것이 본 프로젝트의 핵심이다.

### 1.2 연구 질문 (Research Question)

> **RQ1.** Clean 이미지에서 학습된 객체 탐지 모델은 실환경의 열화 조건(노이즈, 블러, 저해상도)에서도 성능을 유지하는가?
>
> **RQ2.** Corruption Augmentation 또는 Image Restoration 전처리를 적용하면 강건성이 얼마나 개선되는가?
>
> **RQ3.** 각 강건성 전략의 장단점과 최적 적용 시나리오는 무엇인가?

### 1.3 프로젝트 구성

| Phase | 내용 | 평가 횟수 |
|---|---|---|
| Phase 1 | Baseline 모델 학습 및 평가 | 3 x 4 = 12 |
| Phase 2 | Corruption Augmentation 학습 및 평가 | 3 x 4 = 12 |
| Phase 3 | Image Restoration 전처리 및 평가 | 3 x 4 = 12 |
| **합계** | **3가지 전략 x 3개 모델 x 4개 테스트셋** | **36회 평가** |

---

## 2. 실험 설계

### 2.1 데이터셋: VisDrone-DET

| 항목 | 내용 |
|---|---|
| 데이터셋 | VisDrone-DET (드론 시점 객체 탐지) |
| 클래스 수 | 6개 (pedestrian, car, van, truck, bus, motor) |
| 포맷 | COCO (Faster R-CNN용) + YOLO (RT-DETR, YOLOv8용) |
| 특성 | 소형 객체 다수, 밀집 장면, 다양한 해상도 |

VisDrone 데이터셋은 드론에서 촬영된 항공 시점 영상으로, 방산/항공 분야의 실제 운용 환경과 유사한 특성을 지닌다.

### 2.2 객체 탐지 모델 (3종)

| Model | Type | Framework | 특성 |
|---|---|---|---|
| **Faster R-CNN** (ResNet-50 FPN v2) | 2-Stage CNN | torchvision | 높은 정확도, 상대적으로 느림 |
| **RT-DETR-L** | Transformer | Ultralytics | 전역 어텐션 기반, 강건성 우수 |
| **YOLOv8m** | 1-Stage CNN | Ultralytics | 최고 성능, 실시간 처리 가능 |

3개 모델은 **서로 다른 아키텍처 패러다임**(2-stage CNN, Transformer, 1-stage CNN)을 대표하여, 아키텍처가 강건성에 미치는 영향을 비교할 수 있다.

### 2.3 이미지 열화 조건 (4종)

| 조건 | 설명 | 파라미터 | 실환경 대응 |
|---|---|---|---|
| **Clean** | 원본 이미지 | - | 이상적 환경 |
| **Noise** | 가우시안 노이즈 | sigma=15 | 저조도 센서 노이즈 |
| **Blur** | 모션 블러 | kernel=9, angle=0 | 카메라/드론 진동 |
| **LowRes** | 저해상도 (축소 후 확대) | factor=0.5x | 원거리 촬영, 대역폭 제한 |

### 2.4 강건성 개선 전략 (3종)

```
Strategy A (Baseline)
  Clean 학습 → 열화 이미지 직접 추론

Strategy B (Augmented)
  Corruption Augmentation 학습 → 열화 이미지 직접 추론

Strategy C (Restored)
  Clean 학습 → U-Net 복원 → 복원 이미지 추론
```

---

## 3. Phase 1: Baseline 평가

### 3.1 학습 설정

모든 모델은 Clean 데이터로만 학습하여, 열화에 대한 사전 지식 없이 **제로샷 강건성(zero-shot robustness)**을 측정한다.

| 모델 | Epochs | Batch | Optimizer | Pretrained |
|---|---|---|---|---|
| Faster R-CNN | 24 | 2 | SGD (lr=0.005) | ImageNet |
| RT-DETR-L | 100 | 2 | AdamW (auto) | COCO |
| YOLOv8m | 100 | 4 | AdamW (auto) | COCO |

### 3.2 Baseline 결과 (mAP@50)

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN | 0.532 | 0.472 | 0.287 | 0.454 |
| RT-DETR-L | 0.536 | 0.475 | 0.397 | 0.500 |
| YOLOv8m | **0.666** | **0.577** | **0.432** | **0.628** |

### 3.3 성능 하락률 (Clean 대비)

| Model | Noise | Blur | LowRes |
|---|---:|---:|---:|
| FasterRCNN | -11.3% | **-46.1%** | -14.7% |
| RT-DETR-L | -11.4% | -26.0% | -6.6% |
| YOLOv8m | -13.4% | -35.1% | -5.7% |

### 3.4 Baseline 핵심 발견

**1. Blur가 가장 치명적인 열화**
- Faster R-CNN: -46.1% 하락 (사실상 탐지 불가 수준)
- 블러는 객체의 경계(edge)와 텍스처를 직접 파괴하여, 모든 모델에 가장 큰 영향

**2. 아키텍처별 강건성 차이**
- **Faster R-CNN (2-Stage)**: 가장 취약. RPN의 영역 제안 실패가 2단계로 누적되는 cascade failure
- **RT-DETR-L (Transformer)**: 상대적으로 강건. 전역 어텐션(self-attention)이 지역 정보 손실을 보완
- **YOLOv8m (1-Stage)**: 절대 성능 최고. 하락 후에도 mAP 1위 유지

**3. Clean 성능 ≠ 실환경 강건성**
- YOLOv8m: Clean 0.666이지만 Blur 0.432 (-35.1%)
- "Clean에서 잘 되는 모델이 실환경에서도 잘 된다"는 가정은 위험

---

## 4. Phase 2: Corruption Augmentation

### 4.1 전략 설명

학습 시 50% 확률로 corruption(Noise/Blur/LowRes 중 랜덤 1개)을 적용하여, 모델이 열화에 강건한 특징을 학습하도록 유도한다.

- **적용 확률**: 50% (나머지 50%는 Clean 유지)
- **파라미터**: 테스트셋 생성과 동일 (sigma=15, kernel=9, factor=0.5)
- **구현**: `scripts/augmentations.py` 공용 모듈

### 4.2 모델별 구현

| 모델 | 구현 방식 |
|---|---|
| Faster R-CNN | `RandomCorruption` 클래스 (torchvision transforms 파이프라인) |
| RT-DETR / YOLOv8 | Ultralytics `Albumentations.__call__` monkey-patching |

하이퍼파라미터는 Baseline과 동일하게 유지하여, 성능 차이가 **순수하게 augmentation에 의한 것**임을 보장.

### 4.3 Augmented 결과 (mAP@50)

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN_aug | 0.540 | 0.514 | 0.442 | 0.487 |
| RT-DETR-L_aug | **0.578** | **0.547** | **0.524** | **0.543** |
| YOLOv8m_aug | 0.660 | **0.640** | **0.608** | **0.639** |

### 4.4 Baseline 대비 개선 효과

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN | +0.009 | +0.043 | **+0.156** | +0.033 |
| RT-DETR-L | +0.042 | +0.072 | **+0.127** | +0.042 |
| YOLOv8m | -0.006 | +0.063 | **+0.175** | +0.012 |

### 4.5 하락률 변화 (Augmented)

| Model | Noise | Blur | LowRes |
|---|---:|---:|---:|
| FasterRCNN_aug | -4.8% | -18.1% | -10.0% |
| RT-DETR-L_aug | -5.3% | -9.4% | -6.1% |
| **YOLOv8m_aug** | **-3.0%** | **-7.9%** | **-3.1%** |

![Baseline vs Augmented Comparison](../experiments/figures/map50_comparison.png)
*Figure 1. 6개 모델(Baseline 3 + Augmented 3) x 4개 테스트셋 mAP@50 비교*

![Degradation Comparison](../experiments/figures/degradation_comparison.png)
*Figure 2. Clean 대비 성능 하락률 비교*

### 4.6 Augmentation 핵심 발견

**1. Blur에서 가장 극적인 개선**
- FRCNN: 0.287 → 0.442 (+0.156), 하락률 -46.1% → -18.1%
- YOLOv8m: 0.432 → 0.608 (+0.175), 하락률 -35.1% → -7.9%

**2. Clean 성능 유지 또는 향상**
- RT-DETR-L: Clean +4.2%p 향상 (정규화 효과)
- 나머지: 동등 수준 유지

**3. YOLOv8m_aug: 종합 최강**
- 모든 corruption 조건에서 절대 성능 1위
- 최소 하락률: -3.0% ~ -7.9%

### 4.7 Demo: Baseline vs Augmented (Blur)

실제 이미지에서 Baseline과 Augmented 모델의 탐지 결과를 비교:

![FRCNN Demo](../experiments/demo/FRCNN_img0161_gt102_base19_aug50.jpg)
*Figure 3. Faster R-CNN — GT: 102개 | Baseline: 19개 (19%) | Augmented: 50개 (49%)*

![YOLOv8m Demo](../experiments/demo/YOLOv8m_img0366_gt170_base38_aug75.jpg)
*Figure 4. YOLOv8m — GT: 170개 | Baseline: 38개 (22%) | Augmented: 75개 (44%)*

![RT-DETR Demo](../experiments/demo/RT-DETR_img0366_gt170_base94_aug162.jpg)
*Figure 5. RT-DETR-L — GT: 170개 | Baseline: 94개 (55%) | Augmented: 162개 (95%)*

---

## 5. Phase 3: Image Restoration

### 5.1 전략 설명

열화된 이미지를 **경량 U-Net으로 복원(전처리)**한 후, 기존 Baseline 모델로 탐지를 수행한다. 모델 자체를 재학습하지 않고, **전처리 파이프라인만 추가**하는 방식.

```
[열화 이미지] → [U-Net 복원] → [Baseline 모델 추론] → [탐지 결과]
```

### 5.2 Restoration U-Net 아키텍처

| 항목 | 상세 |
|---|---|
| 아키텍처 | Lightweight U-Net + Residual Learning |
| Encoder | 4 downsample blocks [32, 64, 128, 256] |
| Bottleneck | 256 channels |
| Decoder | 4 upsample blocks + skip connections |
| 출력 | `output = input + learned_residual` |
| 파라미터 수 | **3.70M** (경량) |

### 5.3 학습 설정

| 항목 | 값 |
|---|---|
| Epochs | 60 |
| Batch Size | 8 |
| Patch Size | 256 x 256 (랜덤 크롭) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=60, eta_min=1e-6) |
| Loss | L1 Loss + (1 - SSIM), alpha=0.5 |
| 학습 데이터 | Clean 이미지에 on-the-fly corruption 적용 → Clean 복원 타겟 |

### 5.4 복원 모델 성능

| Metric | Best Value | Epoch |
|---|---|---|
| **PSNR** | **34.03 dB** | 55 |
| **SSIM** | **0.947** | 55 |

### 5.5 Restored 탐지 결과 (mAP@50)

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FRCNN (Restored) | 0.532 | 0.177 | **0.502** | 0.483 |
| RT-DETR (Restored) | 0.536 | 0.233 | **0.514** | 0.509 |
| YOLOv8m (Restored) | **0.666** | 0.201 | **0.640** | **0.642** |

### 5.6 Restoration 핵심 발견

**1. Blur 복원에서 최강 성능**
- U-Net이 motion blur 제거에 매우 효과적
- FRCNN: 0.287 → 0.502 (+0.216) — Augmented(0.442)보다도 +0.060 높음
- YOLOv8m: 0.432 → 0.640 (+0.208) — Augmented(0.608)보다 +0.032 높음

**2. Noise 조건에서 치명적 실패**
- FRCNN: 0.472 → 0.177 (-62.5%)
- YOLOv8m: 0.577 → 0.201 (-65.2%)
- **원인**: U-Net이 노이즈 제거 시 텍스처/엣지 정보까지 과도하게 제거 (over-smoothing)

**3. LowRes: 소폭 개선**
- Baseline 대비 개선되나, Augmented와 비슷한 수준

---

## 6. 3-Strategy 종합 비교

### 6.1 전체 비교 차트

![3-Strategy Comparison](../experiments/figures/three_strategy_comparison.png)
*Figure 6. 3가지 전략(Baseline / Augmented / Restored) x 3개 모델 x 4개 조건 비교*

### 6.2 전략별 Baseline 대비 개선 효과

![Strategy Improvement](../experiments/figures/strategy_improvement.png)
*Figure 7. Augmented와 Restored의 Baseline 대비 mAP@50 개선/하락*

| Model | Strategy | Noise | Blur | LowRes |
|---|---|---:|---:|---:|
| FRCNN | Augmented | +0.043 | +0.156 | +0.033 |
| | Restored | **-0.294** | **+0.216** | +0.029 |
| RT-DETR | Augmented | **+0.072** | +0.127 | **+0.042** |
| | Restored | -0.242 | **+0.118** | +0.008 |
| YOLOv8m | Augmented | **+0.063** | +0.175 | +0.011 |
| | Restored | -0.376 | **+0.208** | +0.015 |

### 6.3 조건별 최적 전략

![Best Strategy Heatmap](../experiments/figures/best_strategy_heatmap.png)
*Figure 8. 각 모델-조건 조합에서의 최적 전략 (A=Augmented, R=Restored)*

| 조건 | 최적 전략 | 근거 |
|---|---|---|
| **Noise** | **Augmented** | Restored는 over-smoothing으로 성능 악화 |
| **Blur** | **Restored** | U-Net 복원이 가장 효과적 (+0.208~0.216) |
| **LowRes** | Augmented/Restored 유사 | 두 전략 모두 소폭 개선 |
| **범용** | **Augmented** | 모든 조건에서 안정적으로 개선 |

### 6.4 강건성 프로파일 (Radar Chart)

![Robustness Radar](../experiments/figures/three_strategy_radar.png)
*Figure 9. 3가지 전략의 강건성 프로파일 — Restored(주황)가 Noise에서 급격히 축소되는 것이 뚜렷*

---

## 7. 클래스별 심층 분석

### 7.1 Blur 조건에서의 클래스별 AP@50

![Class Heatmap](../experiments/figures/class_ap50_blur_heatmap.png)
*Figure 10. Blur 조건에서 모델/클래스별 AP@50 히트맵*

| Class | 특성 | Augmentation 개선 패턴 |
|---|---|---|
| **pedestrian** | 소형, 다양한 포즈 | 가장 큰 개선 (+0.19~0.25) |
| **car** | 대형, 일정한 형태 | 이미 높은 수준에서 추가 향상 |
| **van** | 중형, truck과 혼동 | 일관된 개선 |
| **truck** | Clean에서도 상대적으로 낮음 | 개선되나 여전히 취약 |
| **bus** | 대형, 뚜렷한 특징 | 안정적 개선 |
| **motor** | 소형, 매우 취약 | pedestrian과 함께 가장 큰 개선 |

**핵심**: 소형 객체(pedestrian, motor)가 열화에 가장 취약하며, augmentation으로 가장 큰 개선을 보인다.

---

## 8. 아키텍처별 분석

### 8.1 Faster R-CNN (2-Stage CNN)

| 특성 | 분석 |
|---|---|
| **약점** | Blur에서 -46.1% 하락 (cascade failure: RPN 실패 → 분류 실패) |
| **Augmented 효과** | Blur 하락률 -46.1% → -18.1% (가장 극적 개선) |
| **Restored 효과** | Blur에서 0.502 달성 (Augmented 0.442보다 높음) |
| **결론** | 두 전략 모두 효과적이며, 특히 Restoration이 Blur에서 우위 |

### 8.2 RT-DETR-L (Transformer)

| 특성 | 분석 |
|---|---|
| **강점** | 전역 어텐션으로 기본 강건성이 높음 (Blur -26.0%) |
| **Augmented 효과** | Clean까지 +4.2%p 향상 (정규화 효과), 모든 조건 개선 |
| **Restored 효과** | Blur/LowRes에서 Augmented와 유사한 수준 |
| **결론** | Augmented가 가장 균형 잡힌 전략 |

### 8.3 YOLOv8m (1-Stage CNN)

| 특성 | 분석 |
|---|---|
| **강점** | 절대 성능 최고 (Clean 0.666) |
| **Augmented 효과** | 하락률 -3.0%~-7.9%로 최고 강건성 달성 |
| **Restored 효과** | Blur 0.640 (Augmented 0.608보다 높음), LowRes도 소폭 우위 |
| **결론** | 실시간 배포에 최적, Blur 특화 시 Restoration 조합 고려 |

---

## 9. 종합 순위

### 9.1 절대 성능 기준 (mAP@50)

| 순위 | Clean | Noise | Blur | LowRes |
|---|---|---|---|---|
| 1 | YOLOv8m (0.666) | YOLOv8m_aug (0.640) | YOLOv8m_rest (0.640) | YOLOv8m_rest (0.642) |
| 2 | YOLOv8m_aug (0.660) | YOLOv8m (0.577) | YOLOv8m_aug (0.608) | YOLOv8m_aug (0.639) |
| 3 | RT-DETR_aug (0.578) | RT-DETR_aug (0.547) | RT-DETR_aug (0.524) | RT-DETR_aug (0.543) |

### 9.2 강건성 기준 (평균 하락률, Augmented 기준)

| 순위 | Model | Noise | Blur | LowRes | 평균 |
|---|---|---:|---:|---:|---:|
| **1** | **YOLOv8m_aug** | -3.0% | -7.9% | -3.1% | **-4.7%** |
| 2 | RT-DETR-L_aug | -5.3% | -9.4% | -6.1% | -6.9% |
| 3 | FasterRCNN_aug | -4.8% | -18.1% | -10.0% | -11.0% |

---

## 10. 결론

### 10.1 연구 질문에 대한 답변

**RQ1. Clean 학습 모델의 실환경 강건성은?**
- **아니오**, 모든 모델이 열화 조건에서 유의미한 성능 하락을 보인다
- 특히 Blur에서 최대 -46.1% 하락 (Faster R-CNN)
- Clean 성능이 높다고 실환경에서의 강건성이 보장되지 않음

**RQ2. 강건성 개선 전략의 효과는?**
- **Corruption Augmentation**: 모든 조건에서 일관되게 개선 (평균 -4.7% ~ -11.0% 하락률)
- **Image Restoration**: Blur에서 최고 성능이지만, Noise에서 치명적 실패

**RQ3. 최적 적용 시나리오는?**

| 시나리오 | 권장 전략 | 근거 |
|---|---|---|
| 범용 강건성 (추천) | **Augmented** | 모든 조건에서 안정적, 추가 연산 없음 |
| Blur/진동 특화 | **Restored** | U-Net 복원이 deblurring에 최적 |
| 실시간 강건 탐지 | **YOLOv8m_aug** | 절대 성능 + 강건성 + 속도 |
| 열화 유형 알려진 경우 | **조건별 최적 선택** | Noise→Augmented, Blur→Restored |

### 10.2 핵심 기여

1. **3가지 강건성 전략의 정량적 비교**: 36회 평가를 통한 체계적 실험
2. **아키텍처별 강건성 차이 규명**: CNN vs Transformer의 열화 대응 메커니즘 분석
3. **전략별 trade-off 분석**: Augmentation의 범용성 vs Restoration의 특화성
4. **실용적 권장사항 도출**: 운용 환경에 따른 최적 전략 가이드

### 10.3 방산/항공 분야 시사점

| 관점 | 시사점 |
|---|---|
| **드론 운용** | 카메라 진동(blur)이 주 열화 → Restoration 전처리 효과적 |
| **감시 시스템** | 다양한 열화 혼합 → Augmented 학습이 범용적으로 안전 |
| **실시간 처리** | YOLOv8m_aug가 성능+속도+강건성 최적 |
| **배포 유연성** | Restoration은 모델 재학습 없이 전처리만 추가 → 기존 시스템에 적용 용이 |

---

## 11. 기술 상세

### 11.1 개발 환경

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti (8GB VRAM) |
| OS | Windows 11 |
| Python | 3.11 |
| PyTorch | 2.5.1 (CUDA 12.1) |
| Ultralytics | 8.3.209 |

### 11.2 프로젝트 구조

```
Robust-Object-Detection/
├── scripts/
│   ├── convert_visdrone_to_coco.py      # VisDrone → COCO 변환
│   ├── convert_visdrone_to_yolo.py      # VisDrone → YOLO 변환
│   ├── build_corrupted_testsets.py      # 열화 테스트셋 생성
│   ├── coco_detection_dataset.py        # COCO 포맷 Dataset
│   ├── augmentations.py                 # 공용 corruption augmentation
│   ├── train_frcnn_baseline.py          # FRCNN 베이스라인 학습
│   ├── train_frcnn_augmented.py         # FRCNN 증강 학습
│   ├── train_rtdetr_augmented.py        # RT-DETR 증강 학습
│   ├── train_yolo_augmented.py          # YOLOv8 증강 학습
│   ├── restoration_net.py               # U-Net 모델 정의
│   ├── train_restoration.py             # 복원 모델 학습
│   ├── restore_testsets.py              # 테스트셋 복원
│   ├── eval_all.py                      # Baseline/Augmented 평가 (24회)
│   ├── eval_restored.py                 # Restored 평가 (12회)
│   ├── plot_results.py                  # Baseline vs Augmented 시각화
│   ├── plot_three_strategies.py         # 3-전략 비교 시각화
│   └── demo_inference.py               # 데모 비교 이미지 생성
├── experiments/
│   ├── frcnn/                           # Faster R-CNN 체크포인트
│   ├── rtdetr/                          # RT-DETR 체크포인트
│   ├── yolo/                            # YOLOv8 체크포인트
│   ├── restoration/                     # 복원 모델 체크포인트
│   ├── figures/                         # 시각화 차트 (9개)
│   ├── demo/                            # 데모 비교 이미지 (15개)
│   ├── eval_results.json                # Baseline/Augmented 결과
│   └── eval_restored_results.json       # Restored 결과
├── docs/
│   ├── 01_baseline_eval_results.md
│   ├── 02_augmented_training.md
│   ├── 03_final_comparison.md
│   ├── 04_visualization.md
│   ├── 05_demo_inference.md
│   ├── 06_restoration_experiment.md
│   └── Final_Report.md                  # 종합 보고서 (본 문서)
└── data/                                # 데이터셋 (gitignored)
```

### 11.3 실행 가이드

```bash
# 1. 데이터 전처리
python -m scripts.convert_visdrone_to_coco
python -m scripts.convert_visdrone_to_yolo
python -m scripts.build_corrupted_testsets

# 2. Baseline 학습
python -m scripts.train_frcnn_baseline

# 3. Augmented 학습
python -m scripts.train_frcnn_augmented
python -m scripts.train_rtdetr_augmented
python -m scripts.train_yolo_augmented

# 4. Restoration 모델 학습 및 적용
python -m scripts.train_restoration
python -m scripts.restore_testsets

# 5. 평가
python -m scripts.eval_all          # 24회 (Baseline + Augmented)
python -m scripts.eval_restored     # 12회 (Restored)

# 6. 시각화
python -m scripts.plot_results
python -m scripts.plot_three_strategies
python -m scripts.demo_inference
```

---

*Author: Seungbin Yang*
*Date: 2026-02*
*Environment: NVIDIA RTX 3070 Ti, Windows 11, Python 3.11, PyTorch 2.5.1*
