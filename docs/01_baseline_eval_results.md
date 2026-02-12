# Baseline 평가 결과 및 해석

## 1. 실험 개요

### 목적

Clean 데이터로만 학습한 3개 모델(Faster R-CNN, RT-DETR-L, YOLOv8m)을 4종의 테스트셋에서 평가하여, 각 모델의 **절대 성능**과 **열화 조건에서의 강건성(Robustness)**을 정량 비교한다.

### 실험 조건

| 항목 | 내용 |
|------|------|
| 데이터셋 | VisDrone 6클래스 (pedestrian, car, van, truck, bus, motor) |
| 모델 | Faster R-CNN (2-stage CNN), RT-DETR-L (Transformer), YOLOv8m (1-stage CNN) |
| 테스트셋 | Clean(원본), Noise(가우시안 노이즈), Blur(모션 블러), LowRes(저해상도) |
| 평가 지표 | mAP@50, mAP@50-95, 클래스별 AP@50 |
| 총 평가 횟수 | 3 모델 x 4 테스트셋 = 12회 |
| 학습 조건 | 모든 모델은 **Clean 데이터로만 학습** (제로샷 강건성 측정) |
| 평가 스크립트 | `scripts/eval_all.py` |
| 결과 파일 | `experiments/eval_results.json`, `experiments/eval_results.csv` |

---

## 2. 정량 결과

### 2-1. mAP@50

| Model | Clean | Noise | Blur | LowRes |
|-------|------:|------:|-----:|-------:|
| FasterRCNN | 0.532 | 0.472 | 0.287 | 0.454 |
| RT-DETR-L | 0.536 | 0.475 | 0.397 | 0.500 |
| YOLOv8m | **0.666** | **0.577** | **0.432** | **0.628** |

### 2-2. mAP@50-95

| Model | Clean | Noise | Blur | LowRes |
|-------|------:|------:|-----:|-------:|
| FasterRCNN | 0.336 | 0.296 | 0.170 | 0.283 |
| RT-DETR-L | 0.338 | 0.294 | 0.231 | 0.312 |
| YOLOv8m | **0.446** | **0.377** | **0.265** | **0.417** |

### 2-3. 성능 하락률 (Clean 대비 mAP@50 기준)

| Model | Noise | Blur | LowRes |
|-------|------:|-----:|-------:|
| FasterRCNN | -11.3% | **-46.1%** | -14.7% |
| RT-DETR-L | -11.4% | -26.0% | **-6.6%** |
| YOLOv8m | -13.4% | -35.1% | **-5.7%** |

### 2-4. 클래스별 AP@50

**Clean**

| Class | FRCNN | RT-DETR | YOLOv8m |
|-------|------:|--------:|--------:|
| pedestrian | 0.488 | 0.459 | 0.638 |
| car | 0.789 | 0.807 | 0.881 |
| van | 0.466 | 0.472 | 0.572 |
| truck | 0.396 | 0.460 | 0.530 |
| bus | 0.595 | 0.552 | 0.733 |
| motor | 0.457 | 0.465 | 0.641 |

**Blur (가장 큰 성능 하락 조건)**

| Class | FRCNN | RT-DETR | YOLOv8m |
|-------|------:|--------:|--------:|
| pedestrian | 0.167 | 0.254 | 0.289 |
| car | 0.628 | 0.734 | 0.772 |
| van | 0.237 | 0.369 | 0.394 |
| truck | 0.180 | 0.297 | 0.305 |
| bus | 0.348 | 0.438 | 0.552 |
| motor | 0.160 | 0.288 | 0.281 |

---

## 3. 핵심 해석

### 3-1. Blur가 가장 치명적인 열화

세 모델 모두 블러에서 가장 큰 성능 하락을 보인다. 특히 Faster R-CNN은 -46.1%로 거의 절반이 하락한다.

- 블러는 객체의 **경계(edge)와 텍스처 정보를 직접 파괴**한다
- Faster R-CNN의 RPN(Region Proposal Network)은 선명한 edge feature에 크게 의존한다
- 블러 상태에서 작은 객체(pedestrian, motor)는 배경과 거의 구분이 되지 않는다

### 3-2. Faster R-CNN이 가장 취약

Faster R-CNN은 2-stage 구조로 (1) 영역 제안 (2) 분류/회귀를 순차적으로 수행한다. 1단계에서 영역 제안이 실패하면 2단계로 넘어갈 기회 자체가 사라지므로 **열화에 대한 오류가 누적(cascade failure)**된다.

- Blur 시 pedestrian AP50: 0.488 -> 0.167 (-65.7%)
- Blur 시 motor AP50: 0.457 -> 0.160 (-65.0%)
- 사실상 탐지 불가 수준까지 하락

### 3-3. RT-DETR-L이 상대적으로 가장 강건

하락률 기준으로 RT-DETR이 가장 안정적이다:
- Blur: -26.0% (FRCNN -46.1%, YOLO -35.1%)
- LowRes: -6.6% (FRCNN -14.7%, YOLO -5.7%)

Transformer의 **self-attention 메커니즘**이 이 강건성의 원인으로 분석된다:
- CNN은 작은 커널(3x3)로 **지역적 패턴(local texture)**에 의존하여 블러/노이즈에 민감
- Transformer는 **전역적 관계(global context)**를 학습하여, 일부 지역 정보가 훼손되어도 전체 맥락으로 보완 가능
- 예: 블러된 bus를 CNN은 텍스처로 인식하지 못하지만, Transformer는 "도로 위에 크고 직사각형인 물체"라는 전역적 단서로 추론 가능

### 3-4. YOLOv8m: 절대 성능 vs 강건성 트레이드오프

YOLOv8m은 절대 성능이 높아서 **하락 후에도 여전히 mAP 1위**이다. 다만 하락률(-35.1% Blur)은 RT-DETR보다 크다.

- YOLOv8의 높은 Clean 성능이 데이터에 과적합(overfitting)된 측면이 있을 수 있음
- 1-stage 구조로 빠르지만, feature 추출 단계의 열화가 곧바로 최종 출력에 반영됨

### 3-5. 클래스별 패턴

**Car는 모든 조건에서 가장 잘 탐지된다** (Clean AP50: 0.79~0.88):
- 크기가 크고, 형태가 일정하고, 학습 데이터에 가장 많이 등장하기 때문

**Pedestrian과 Motor가 가장 취약하다:**
- 작은 크기, 다양한 포즈/형태
- 열화 시 배경과의 구분이 급격히 어려워짐

**Truck이 Clean에서도 상대적으로 낮다** (0.40~0.53):
- VisDrone 데이터에서 truck 샘플이 상대적으로 적음
- Van과 시각적으로 유사하여 혼동 발생

---

## 4. 종합 순위

| 관점 | 1위 | 2위 | 3위 |
|------|-----|-----|-----|
| 절대 성능 (mAP@50) | YOLOv8m | RT-DETR-L | FasterRCNN |
| 강건성 (하락률 기준) | RT-DETR-L | YOLOv8m | FasterRCNN |

---

## 5. 시사점 및 다음 단계

| 관점 | 결론 |
|------|------|
| 실용적 배포 | YOLOv8m -- 열화 후에도 절대 mAP가 가장 높아 실전 성능 최고 |
| 강건성 연구 | RT-DETR-L -- Transformer의 전역 어텐션이 열화에 대한 내성을 제공 |
| 개선 필요 | FasterRCNN -- Blur에서 -46% 하락은 실용 불가 수준 |
| 핵심 발견 | Clean에서의 높은 정확도가 실환경 강건성을 보장하지 않음 |

### 후속 실험 방향

1. **Augmentation 기반 강건성 개선**: 학습 시 열화 데이터를 augmentation으로 섞어 재학습
2. **Adversarial Training**: 의도적으로 모델을 공격하는 샘플로 학습하여 내성 강화
3. **모델 앙상블**: 서로 다른 약점을 가진 모델을 결합하여 전체 강건성 향상
4. **도메인 적응(Domain Adaptation)**: 열화 도메인에 대한 fine-tuning 또는 스타일 전이

---

*평가 수행일: 2026-02-12*
*평가 환경: NVIDIA GeForce RTX 3070 Ti (8GB), Python 3.11, PyTorch 2.5.1, Ultralytics 8.3.209*
