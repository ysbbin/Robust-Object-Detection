# Image Restoration + Detection: 3-Strategy 비교 분석

## 개요

Baseline, Corruption Augmentation에 이어 세 번째 강건성 전략으로 **Image Restoration 전처리**를 구현하고, 3가지 전략을 종합 비교하였다.

**핵심 아이디어**: 열화된 이미지를 U-Net으로 복원한 후, 기존 Baseline 모델로 탐지 수행

```
Strategy A: Baseline         — Clean으로만 학습 → 열화 이미지 직접 추론
Strategy B: Augmented        — Corruption augmentation으로 학습 → 열화 이미지 직접 추론
Strategy C: Restored (신규)  — Clean으로만 학습 → U-Net 복원 → 복원 이미지로 추론
```

## 1. Restoration U-Net 아키텍처

### 모델 구조

| 구성 요소 | 상세 |
|---|---|
| 아키텍처 | Lightweight U-Net + Residual Learning |
| Encoder | 4 downsample blocks [32, 64, 128, 256] |
| Bottleneck | 256 channels |
| Decoder | 4 upsample blocks + skip connections |
| 출력 방식 | `output = input + learned_residual` |
| 파라미터 수 | **3.70M** (경량) |
| 입력 제약 | 없음 (Fully Convolutional) |

### 핵심 설계 결정

- **Residual Learning**: 입력 이미지에 잔차(residual)를 더하는 방식으로, 학습 안정성과 수렴 속도 향상
- **Padding**: 추론 시 16의 배수로 패딩하여 U-Net 호환성 보장
- **경량 설계**: 탐지 파이프라인 앞단에 위치하므로 연산 오버헤드를 최소화

## 2. 학습 설정

| 하이퍼파라미터 | 값 |
|---|---|
| Epochs | 60 |
| Batch Size | 8 |
| Patch Size | 256 x 256 (랜덤 크롭) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=60, eta_min=1e-6) |
| Loss | L1 Loss + (1 - SSIM), alpha=0.5 |

### 학습 데이터

- **입력**: 학습 이미지에 on-the-fly corruption 적용 (Noise/Blur/LowRes 랜덤)
- **타겟**: 원본 Clean 이미지
- 기존 `augmentations.py`의 corruption 함수를 재활용

### 학습 결과

| Metric | Best Value | Epoch |
|---|---|---|
| PSNR | **34.03 dB** | 55 |
| SSIM | **0.947** | 55 |
| Final Train Loss | 0.032 | 60 |

## 3. 복원 성능 (이미지 품질)

U-Net은 열화 이미지를 시각적으로 상당히 복원하며, 특히 **Blur와 LowRes에서 효과적**이다. 그러나 Noise 조건에서는 텍스처 정보까지 제거되는 over-smoothing 경향이 관찰됨.

## 4. 탐지 성능 비교 (mAP@50)

### 3-Strategy 비교 차트

![3-Strategy Comparison](../experiments/figures/three_strategy_comparison.png)

### 전체 결과표

| Model | Strategy | Clean | Noise | Blur | LowRes |
|---|---|---:|---:|---:|---:|
| **FasterRCNN** | Baseline | 0.532 | 0.472 | 0.287 | 0.454 |
| | Augmented | 0.540 | **0.514** | 0.442 | 0.487 |
| | Restored | 0.532 | 0.177 | **0.502** | **0.483** |
| **RT-DETR-L** | Baseline | 0.536 | 0.475 | 0.397 | 0.500 |
| | Augmented | **0.578** | **0.547** | **0.524** | **0.543** |
| | Restored | 0.536 | 0.233 | 0.514 | 0.509 |
| **YOLOv8m** | Baseline | **0.666** | 0.577 | 0.432 | 0.628 |
| | Augmented | 0.660 | **0.640** | 0.608 | 0.639 |
| | Restored | **0.666** | 0.201 | **0.640** | **0.642** |

### Baseline 대비 개선 효과

![Strategy Improvement](../experiments/figures/strategy_improvement.png)

| Model | Strategy | Noise | Blur | LowRes |
|---|---|---:|---:|---:|
| **FRCNN** | Augmented | +0.043 | +0.156 | +0.033 |
| | Restored | **-0.294** | **+0.216** | +0.029 |
| **RT-DETR** | Augmented | **+0.072** | **+0.127** | **+0.042** |
| | Restored | -0.242 | +0.118 | +0.008 |
| **YOLOv8m** | Augmented | **+0.063** | +0.175 | +0.011 |
| | Restored | -0.376 | **+0.208** | +0.015 |

### Best Strategy 히트맵

![Best Strategy](../experiments/figures/best_strategy_heatmap.png)

### Radar Chart

![Radar](../experiments/figures/three_strategy_radar.png)

## 5. 분석 및 인사이트

### Restoration 전략의 강점

1. **Blur 조건 최강**: 모든 모델에서 Augmented보다 높은 성능
   - FRCNN: +0.216 (Baseline 대비), Augmented보다 +0.060 더 높음
   - YOLOv8m: +0.208 (Baseline 대비), Augmented보다 +0.032 더 높음
   - U-Net이 motion blur 제거에 매우 효과적

2. **LowRes 조건 양호**: Baseline 대비 개선, Augmented와 동등 수준
   - YOLOv8m에서 Restored(0.642)가 Augmented(0.639)를 근소하게 상회

3. **Clean 성능 보존**: Baseline과 동일 (전처리 파이프라인이므로 Clean은 변화 없음)

### Restoration 전략의 약점

1. **Noise 조건 치명적 실패**: 모든 모델에서 Baseline보다도 크게 하락
   - FRCNN: 0.472 → 0.177 (-62.5%)
   - YOLOv8m: 0.577 → 0.201 (-65.2%)
   - **원인 분석**: U-Net이 노이즈 제거 시 텍스처/엣지 정보까지 과도하게 제거 (over-smoothing)
   - 특히 소형 객체(pedestrian, motor)에서 치명적 — 미세한 특징이 사라짐

2. **추가 연산 비용**: 추론 시 U-Net 전처리 단계 추가 필요

### 전략별 적합 시나리오

| 전략 | 최적 시나리오 | 비적합 시나리오 |
|---|---|---|
| **Baseline** | 깨끗한 입력이 보장되는 환경 | 실환경 열화 조건 |
| **Augmented** | **범용적 강건성** 필요 시 (추천) | Clean 성능 극대화 필요 시 |
| **Restored** | **Blur/LowRes** 열화가 지배적인 환경 | 노이즈가 심한 환경 |

## 6. 결론

- **범용 강건성**: Corruption Augmentation이 모든 조건에서 안정적으로 개선 → **가장 실용적인 전략**
- **특화 강건성**: Image Restoration은 Blur 조건에서 최고 성능을 달성하지만, Noise에 취약한 trade-off 존재
- **최적 조합 가능성**: 입력 열화 유형을 분류한 후, 조건별 최적 전략을 선택하는 adaptive pipeline이 잠재적 최선
- **방산/항공 관점**: 카메라 진동에 의한 blur가 주된 열화인 드론/항공 촬영 환경에서는 Restoration 전략이 특히 효과적

## 7. 관련 파일

| 파일 | 설명 |
|---|---|
| `scripts/restoration_net.py` | U-Net 모델 정의 |
| `scripts/train_restoration.py` | 복원 모델 학습 |
| `scripts/restore_testsets.py` | 테스트셋 복원 |
| `scripts/eval_restored.py` | 복원 테스트셋 평가 |
| `scripts/plot_three_strategies.py` | 3-전략 비교 시각화 |
| `experiments/eval_restored_results.json` | 복원 평가 결과 |
| `experiments/restoration/best.pth` | 학습된 U-Net 체크포인트 |
| `experiments/restoration/history.jsonl` | 학습 이력 |
