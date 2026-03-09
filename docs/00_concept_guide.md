# 프로젝트 개념 정리 가이드

> 이 프로젝트를 이해하기 위해 필요한 모든 개념을 정리한 문서.
> 모델 구조, 학습 방식, 평가 지표, 실험 결과, 인사이트까지 한 곳에서 확인할 수 있다.

---

## 목차

1. [프로젝트 배경 및 핵심 질문](#1-프로젝트-배경-및-핵심-질문)
2. [객체 탐지 기초 개념](#2-객체-탐지-기초-개념)
3. [사용 모델: 3가지 아키텍처](#3-사용-모델-3가지-아키텍처)
4. [이미지 열화(Corruption) 개념](#4-이미지-열화corruption-개념)
5. [강건성 개선 전략 3가지](#5-강건성-개선-전략-3가지)
6. [데이터셋: VisDrone](#6-데이터셋-visdrone)
7. [실험 결과 및 인사이트 (이미지 탐지)](#7-실험-결과-및-인사이트-이미지-탐지)
8. [실험 결과 및 인사이트 (비디오 탐지)](#8-실험-결과-및-인사이트-비디오-탐지)
9. [전체 종합 결론](#9-전체-종합-결론)

---

## 1. 프로젝트 배경 및 핵심 질문

### 왜 이 프로젝트인가?

드론, 항공기, 감시 카메라 등 **실환경 객체 탐지 시스템**은 이상적인 조건에서 동작하지 않는다.

- **카메라 진동** → 모션 블러 발생
- **저조도 센서** → 가우시안 노이즈 발생
- **원거리 촬영 / 대역폭 제한** → 저해상도 이미지

문제는 대부분의 객체 탐지 연구가 **Clean(깨끗한) 이미지** 기준 성능만 보고한다는 것이다.

> "Clean에서 좋은 성능 = 실환경에서도 좋은 성능" 이 가정이 맞는가?

이 프로젝트는 이 가정을 실험적으로 **부정**하고, 개선 방법을 비교 검증한다.

### 핵심 연구 질문

| 질문 | 답 (요약) |
|---|---|
| Clean 모델이 열화 조건에서도 잘 되는가? | **아니다.** Faster R-CNN은 Blur에서 -46.1% 폭락 |
| Corruption Augmentation이 효과적인가? | **매우 효과적.** Blur 하락률 -35% → -8% |
| Image Restoration이 더 나은가? | **상황에 따라 다름.** Blur에서는 최강, Noise에서는 역효과 |
| 비디오 도메인에서도 동일한 패턴인가? | **그렇다.** 동일한 패턴이 VID에서도 재현됨 |

---

## 2. 객체 탐지 기초 개념

### 2-1. 객체 탐지란?

이미지에서 **"무엇이 어디에 있는가"** 를 동시에 답하는 태스크.

- **분류(Classification)**: 이미지 전체에서 클래스 예측
- **탐지(Detection)**: 각 객체의 **위치(Bounding Box) + 클래스** 동시 예측

출력 형태: `[x1, y1, x2, y2, class, confidence]`

### 2-2. IoU (Intersection over Union)

예측 박스와 정답 박스가 **얼마나 겹치는지** 측정하는 지표.

```
IoU = 교집합 넓이 / 합집합 넓이
```

- IoU = 1.0: 완벽히 일치
- IoU = 0.0: 전혀 겹치지 않음
- **임계값(threshold)**: 보통 0.5 이상이면 올바른 탐지로 간주

### 2-3. mAP (mean Average Precision)

객체 탐지 모델의 표준 성능 지표.

**계산 과정:**
1. Precision-Recall 곡선 작성
2. 각 클래스별 AP(Area under PR curve) 계산
3. 모든 클래스의 AP 평균 = **mAP**

**mAP@50**: IoU 임계값 0.5에서의 mAP. 가장 많이 사용되는 지표.

**mAP@50-95**: IoU 임계값 0.5~0.95 (0.05 간격)에서의 mAP 평균. 더 엄격한 지표.

```
mAP@50   → "박스가 50% 이상 겹치면 맞다고 인정"
mAP@50-95 → "다양한 겹침 기준으로 종합 평가"
```

### 2-4. Precision vs Recall

| 지표 | 의미 | 계산 |
|---|---|---|
| Precision | 탐지한 것 중 맞은 비율 | TP / (TP + FP) |
| Recall | 실제 객체 중 탐지한 비율 | TP / (TP + FN) |

- **Precision 우선**: False Alarm을 줄이고 싶을 때
- **Recall 우선**: 놓치는 객체를 줄이고 싶을 때 (보안, 군사 분야에서 중요)

---

## 3. 사용 모델: 3가지 아키텍처

### 3-1. Faster R-CNN (2-Stage CNN)

**핵심 아이디어**: 탐지를 2단계로 나눔

```
Stage 1: Region Proposal Network (RPN)
  → "여기에 객체가 있을 것 같다" (후보 영역 제안)

Stage 2: RoI Head
  → "이게 정확히 뭔가?" (분류 + 박스 정밀화)
```

**아키텍처 구성**

| 구성 요소 | 역할 |
|---|---|
| Backbone (ResNet-50) | 이미지에서 특징(feature) 추출 |
| FPN (Feature Pyramid Network) | 다양한 크기의 객체 탐지를 위한 다중 스케일 특징 맵 |
| RPN | Anchor 기반 객체 후보 영역 생성 |
| RoI Align | 후보 영역을 고정 크기로 변환 |
| Box Head | 최종 클래스 분류 + 박스 좌표 회귀 |

**학습 방식**

- ImageNet으로 사전학습된 ResNet-50 백본 사용
- VisDrone 데이터로 Fine-tuning
- Optimizer: SGD (lr=0.005, momentum=0.9)
- Epochs: 24
- Loss = RPN loss + Classification loss + Regression loss

**열화에 취약한 이유**

2-stage 구조의 **Cascade Failure** 문제:
- Stage 1에서 RPN이 블러/노이즈로 인해 객체 후보를 못 잡으면 → Stage 2에 넘어갈 기회 자체가 없음
- CNN의 지역적(local) 텍스처 패턴에 의존 → 열화로 텍스처 파괴 시 치명적

---

### 3-2. RT-DETR-L (Transformer 기반)

**핵심 아이디어**: Transformer의 Self-Attention으로 전역적(global) 관계 학습

```
전통 CNN: 3x3 커널로 주변 픽셀만 봄 (지역적)
Transformer: 이미지 전체의 모든 위치 간 관계를 봄 (전역적)
```

**아키텍처 구성**

| 구성 요소 | 역할 |
|---|---|
| Backbone (ResNet) | 초기 특징 추출 |
| Encoder (Transformer) | Self-Attention으로 전역 문맥 학습 |
| Decoder (Cross-Attention) | 쿼리(query)가 이미지 특징에서 객체 정보 추출 |
| 출력 | N개의 (class, box) 직접 예측 (NMS 불필요) |

**학습 방식**

- COCO pretrained 가중치 (rtdetr-l.pt) 사용
- VisDrone 데이터로 Fine-tuning
- Optimizer: AdamW (자동 설정)
- Epochs: 100, Image size: 1024

**열화에 강건한 이유**

- Blur된 pedestrian을 CNN은 텍스처로 인식 못하지만, Transformer는 **"도로 위에 직립한 형태"** 라는 전역 단서로 추론 가능
- Self-Attention이 일부 영역 정보가 훼손되어도 다른 영역 정보로 보완

---

### 3-3. YOLOv8m (1-Stage CNN)

**핵심 아이디어**: 이미지를 그리드로 나눠 한 번에 탐지 (빠른 추론)

```
2-Stage: Region Proposal → Classification (느리지만 정확)
1-Stage: 그리드별 직접 예측 → 빠르고 실용적
```

**아키텍처 구성**

| 구성 요소 | 역할 |
|---|---|
| Backbone (CSPDarknet) | 효율적인 특징 추출 |
| Neck (C2f + PAN) | 다중 스케일 특징 융합 |
| Head | 각 그리드에서 박스 + 클래스 직접 예측 |

**학습 방식**

- COCO pretrained 가중치 (yolov8m.pt) 사용
- VisDrone 데이터로 Fine-tuning
- Optimizer: AdamW (자동 설정)
- Epochs: 100, Batch: 4, Image size: 1024

**특징**

- 3개 모델 중 **절대 성능 최고** (Clean mAP50: 0.666)
- 실시간 추론 가능
- Clean 성능이 높지만, 열화 시 하락폭도 상대적으로 큼 (과적합 측면 존재)

---

### 3-4. 3가지 아키텍처 비교

| 관점 | Faster R-CNN | RT-DETR-L | YOLOv8m |
|---|---|---|---|
| 패러다임 | 2-Stage CNN | Transformer | 1-Stage CNN |
| Clean 성능 | 보통 (0.532) | 보통 (0.536) | 최고 (0.666) |
| 기본 강건성 | 최취약 | 가장 강건 | 중간 |
| 추론 속도 | 느림 | 중간 | 빠름 |
| 열화 취약 원인 | Cascade Failure | 상대적으로 내성 | 지역 특징 의존 |

---

## 4. 이미지 열화(Corruption) 개념

### 4-1. 왜 이 3가지 열화인가?

드론/항공 촬영 실환경에서 가장 빈번하게 발생하는 열화 유형:

| 열화 | 실환경 원인 | 구현 방식 |
|---|---|---|
| **Gaussian Noise** | 저조도 촬영 시 센서 노이즈 | 정규분포 노이즈 추가 (sigma=15) |
| **Motion Blur** | 드론 진동, 빠른 이동 | 선형 커널 컨볼루션 (kernel=9) |
| **Low Resolution** | 원거리 촬영, 통신 대역폭 제한 | 0.5x 축소 후 원본 크기로 복원 |

### 4-2. 각 열화의 특성

**Gaussian Noise**
```
원본 픽셀값 + N(0, sigma²) 랜덤 노이즈
→ 텍스처를 흐릿하게 만들지는 않지만 랜덤 픽셀 교란
→ 고주파 특징(엣지, 텍스처)은 일부 보존
```

**Motion Blur**
```
선형 방향으로 픽셀을 "번지게" 만듦
→ 엣지와 경계 정보가 완전히 파괴
→ 소형 객체는 배경과 구분 불가 수준까지 열화
→ 이 프로젝트에서 가장 치명적인 열화
```

**Low Resolution**
```
이미지를 0.5배 축소 → 다시 원본 크기로 확대
→ 픽셀 정보 손실 + 부드러운 흐림 효과
→ Blur보다 덜 치명적 (전반적 구조는 유지됨)
```

### 4-3. 열화 심각도 순위

```
Blur (-38% 이상) > LowRes (-15~20%) > Noise (-9~15%)
```

**Blur가 가장 치명적인 이유:**
- 객체의 경계(edge)와 텍스처를 동시에 파괴
- 소형 객체(pedestrian, motor)는 배경에 완전히 흡수됨
- CNN의 지역적 특징 추출 자체가 불가능해짐

---

## 5. 강건성 개선 전략 3가지

### 5-1. Strategy A: Baseline

```
[Clean 이미지] → 학습 → [모델] → [열화 이미지] → 추론 → 결과
```

- 기준점(reference) 역할
- 학습 시 열화 이미지를 전혀 보지 않음
- 예측: 열화에 취약할 것

### 5-2. Strategy B: Corruption Augmentation

**핵심 아이디어**: 학습할 때 열화를 경험시켜서 강건성을 학습하게 한다.

```
학습 중 각 이미지에 대해:
  50% 확률로 → {Noise, Blur, LowRes} 중 랜덤 1개 적용
  50% 확률로 → Clean 이미지 그대로 사용
```

**구현 방식**

- Faster R-CNN: `RandomCorruption` 커스텀 transform으로 PIL 파이프라인에 삽입
- RT-DETR / YOLOv8: Ultralytics 내부 augmentation 파이프라인을 **monkey-patching**
  - `patch_ultralytics_augmentations()` 호출 → 기존 augmentation 파이프라인에 corruption 주입
  - 기존 augmentation(mosaic, flip 등)은 그대로 유지

**왜 50% 확률인가?**
- 100%면 Clean 성능이 하락할 수 있음
- 너무 낮으면 강건성 향상 효과 미미
- 50%는 Clean 성능 유지 + 강건성 향상의 균형점

**장점**
- 구현이 간단 (기존 학습 파이프라인에 몇 줄 추가)
- 추론 시 추가 연산 없음
- 모든 열화 조건에서 일관된 개선

### 5-3. Strategy C: Image Restoration (U-Net 전처리)

**핵심 아이디어**: 탐지 전에 열화 이미지를 먼저 복원한다.

```
[열화 이미지] → U-Net 복원 → [복원 이미지] → Baseline 모델 → 결과
```

**U-Net 아키텍처**

```
입력 이미지 (열화)
    ↓
Encoder (4단계 다운샘플링: 32→64→128→256 채널)
    ↓
Bottleneck (256 채널)
    ↓
Decoder (4단계 업샘플링 + Skip Connection)
    ↓
출력: 입력 + Residual (잔차 학습)
```

- **Residual Learning**: 전체 이미지를 재생성하는 것이 아니라 **열화된 부분만 수정**하는 잔차를 학습 → 안정적이고 빠른 수렴
- **Skip Connection**: Encoder의 특징을 Decoder에 직접 전달 → 세밀한 구조 복원
- 파라미터: 3.70M (경량)

**학습 설정**

| 항목 | 설정 |
|---|---|
| 입력 | 열화 이미지 (Noise/Blur/LowRes 중 랜덤 적용) |
| 타겟 | 원본 Clean 이미지 |
| Loss | L1 Loss + (1 - SSIM) 복합 손실 |
| 성능 | PSNR 34.03dB, SSIM 0.947 |

**L1 + SSIM 복합 손실의 이유**
- L1 Loss만 쓰면: 픽셀 단위 오차 최소화 → 흐릿한 복원
- SSIM만 쓰면: 구조적 유사도 → 세밀한 텍스처 부족
- 두 손실 조합: 픽셀 정확도 + 구조적 품질 동시 달성

---

## 6. 데이터셋: VisDrone

### 6-1. VisDrone-DET (이미지 탐지)

| 항목 | 내용 |
|---|---|
| 수집 방식 | 드론에서 촬영한 항공 시점 이미지 |
| 특성 | 소형 객체 다수, 밀집 장면, 다양한 고도/각도 |
| 사용 클래스 | 6개: pedestrian, car, van, truck, bus, motor |
| 포맷 | COCO (Faster R-CNN용) + YOLO (RT-DETR, YOLOv8용) |
| 테스트셋 크기 | 각 조건 548장 |

**왜 6개 클래스인가?**

VisDrone 원본에는 10개 클래스가 있지만, 학습 샘플 수가 적거나 탐지 신뢰도가 낮은 클래스(bicycle, awning-tricycle 등)를 제외하고 **실용적으로 의미 있는 6개**만 선택.

**드론 데이터의 특성**

```
- 객체가 매우 작음 (pedestrian이 수 픽셀 수준)
- 밀집된 장면 (수백 개의 차량/보행자)
- 지면에서 내려다보는 항공 시점 (지상 데이터와 다른 분포)
```

### 6-2. VisDrone-VID (비디오 탐지)

| 항목 | 내용 |
|---|---|
| 수집 방식 | 드론 촬영 비디오 시퀀스 |
| 처리 방식 | 프레임 단위 이미지로 추출 → YOLO 포맷 저장 |
| 어노테이션 | `frame_index, track_id, x, y, w, h, category, ...` |
| 파일명 | `{시퀀스명}_{프레임ID:07d}.jpg` |

**DET vs VID의 차이**

| 항목 | DET | VID |
|---|---|---|
| 데이터 특성 | 정지 이미지 | 비디오 프레임 (시간적 연속성) |
| 도전 요소 | 소형/밀집 객체 | + 움직임 블러, 자세 변화 |
| 학습 방식 | 이미지 단위 | 프레임 단위 (같은 방식) |

### 6-3. 열화 테스트셋 생성

```python
# 3가지 열화를 각각 적용하여 별도 테스트셋 구성
Test_Clean  : 원본 이미지 548장
Test_Noise  : Gaussian noise (sigma=15) 적용
Test_Blur   : Motion blur (kernel=9, angle=0도) 적용
Test_LowRes : 0.5x 축소 후 원본 크기로 업스케일
```

---

## 7. 실험 결과 및 인사이트 (이미지 탐지)

### 7-1. Baseline 결과

**mAP@50**

| 모델 | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| Faster R-CNN | 0.532 | 0.472 | 0.287 | 0.454 |
| RT-DETR-L | 0.536 | 0.475 | 0.397 | 0.500 |
| YOLOv8m | **0.666** | **0.577** | **0.432** | **0.628** |

**성능 하락률**

| 모델 | Noise | Blur | LowRes |
|---|---:|---:|---:|
| Faster R-CNN | -11.3% | **-46.1%** | -14.7% |
| RT-DETR-L | -11.4% | -26.0% | -6.6% |
| YOLOv8m | -13.4% | -35.1% | -5.7% |

**인사이트: 왜 RT-DETR이 Blur에서 가장 강건한가?**

CNN은 3×3 커널로 **지역적 패턴(텍스처, 엣지)**에 의존한다.
Blur가 텍스처를 파괴하면 CNN은 특징을 잡지 못한다.

Transformer는 **전역적 관계(global context)**를 학습한다.
"도로 위에 직립한 형태" "차선 옆에 직사각형" 같은 전체 맥락으로 추론하기 때문에
일부 지역 정보가 훼손되어도 보완이 가능하다.

**인사이트: YOLOv8m의 절대 성능 vs 강건성 트레이드오프**

YOLOv8m은 Clean에서 압도적 1위(0.666)지만 Blur에서 -35.1% 하락.
RT-DETR은 Clean에서 상대적으로 낮지만 Blur 하락률 -26.0%로 더 강건.

→ **"Clean 성능 = 실환경 강건성"이 아님을 증명**

---

### 7-2. Corruption Augmentation 결과

**mAP@50 (Augmented)**

| 모델 | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN_aug | 0.540 | 0.514 | 0.442 | 0.487 |
| RT-DETR-L_aug | **0.578** | **0.547** | **0.524** | **0.543** |
| YOLOv8m_aug | 0.660 | **0.640** | **0.608** | **0.639** |

**Baseline 대비 개선폭 (mAP@50)**

| 모델 | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN | +0.009 | +0.043 | **+0.156** | +0.033 |
| RT-DETR-L | +0.042 | +0.072 | **+0.127** | +0.042 |
| YOLOv8m | -0.006 | +0.063 | **+0.175** | +0.012 |

**핵심 인사이트**

1. **Blur 개선이 가장 극적**: 3개 모델 모두 Blur에서 가장 큰 개선
   - Baseline에서 가장 취약했던 조건 → Augmented에서 가장 많이 회복

2. **Clean 성능 유지**: RT-DETR은 오히려 +4.2%p 향상 (정규화 효과)
   - Corruption augmentation이 다양한 이미지를 학습시켜 더 일반화된 특징 학습

3. **YOLOv8m_aug가 종합 최강**:
   - 모든 열화 조건에서 절대 성능 1위
   - 하락률 -3.0% ~ -7.9% (가장 낮음)

4. **구현 비용 대비 효과 극적**:
   - 학습 코드에 "50% 확률로 corruption 적용" 몇 줄만 추가
   - Blur 하락률 YOLOv8m: -35.1% → -7.9% (4.4배 강건해짐)

---

### 7-3. Image Restoration 결과

**3-Strategy 비교 (mAP@50)**

| 모델 | 전략 | Clean | Noise | Blur | LowRes |
|---|---|---:|---:|---:|---:|
| FasterRCNN | Baseline | 0.532 | 0.472 | 0.287 | 0.454 |
| | Augmented | 0.540 | **0.514** | 0.442 | 0.487 |
| | **Restored** | 0.532 | 0.177 | **0.502** | 0.483 |
| RT-DETR-L | Baseline | 0.536 | 0.475 | 0.397 | 0.500 |
| | **Augmented** | **0.578** | **0.547** | **0.524** | **0.543** |
| | Restored | 0.536 | 0.233 | 0.514 | 0.509 |
| YOLOv8m | Baseline | **0.666** | 0.577 | 0.432 | 0.628 |
| | Augmented | 0.660 | **0.640** | 0.608 | 0.639 |
| | **Restored** | **0.666** | 0.201 | **0.640** | **0.642** |

**Restoration의 강점: Blur**

U-Net이 motion blur 제거에 탁월하다.
- FasterRCNN: Blur 0.287 → 0.502 (+0.216), Augmented(0.442)보다 +0.060 높음
- YOLOv8m: Blur 0.432 → 0.640 (+0.208), Augmented(0.608)보다 +0.032 높음

**Restoration의 치명적 약점: Noise**

U-Net이 Noise 제거 시 **텍스처/엣지 정보까지 과도하게 제거(over-smoothing)**한다.
- FasterRCNN: Noise 0.472 → 0.177 (-62.5%)
- YOLOv8m: Noise 0.577 → 0.201 (-65.2%)

**원인**: 노이즈는 픽셀 단위 랜덤 교란이라 U-Net이 이를 제거하려다 배경과 객체 경계까지 뭉갬. 특히 소형 객체(pedestrian, motor)의 미세한 특징이 사라짐.

**전략별 최적 시나리오**

| 전략 | 최적 상황 | 비적합 상황 |
|---|---|---|
| **Baseline** | 깨끗한 입력 보장 환경 | 실환경 열화 조건 |
| **Augmented** | 범용 강건성 필요 (기본 추천) | Clean 성능 극대화 필요 시 |
| **Restored** | Blur/LowRes가 지배적인 환경 | 노이즈가 심한 환경 |

---

### 7-4. 클래스별 분석

**왜 Car는 잘 탐지되고 Pedestrian은 어려운가?**

| 클래스 | Clean AP50 (YOLOv8m) | Blur AP50 (YOLOv8m) | 이유 |
|---|---:|---:|---|
| car | 0.881 | 0.772 | 크기 크고, 형태 일정, 훈련 샘플 다수 |
| pedestrian | 0.638 | 0.289 | 소형, 다양한 자세, 열화에 특히 취약 |
| motor | 0.641 | 0.281 | 소형, 유사 클래스(bicycle)와 혼동 |
| truck | 0.530 | 0.305 | van과 시각적 유사, 샘플 수 부족 |

**Corruption Augmentation 후 가장 크게 개선된 클래스**

소형 객체(pedestrian, motor)가 가장 극적으로 개선된다.
- 원래 열화 시 배경과 구분 불가 수준 → augmentation 후 실용 가능 수준으로 회복
- pedestrian Blur AP50 (YOLOv8m): 0.289 → 0.535 (거의 2배)

---

## 8. 실험 결과 및 인사이트 (비디오 탐지)

### 8-1. 비디오 모델 Baseline 결과

**mAP@50**

| 모델 | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| YOLOv8m-VID | 0.387 | 0.330 | 0.239 | 0.309 |
| RT-DETR-VID | 0.316 | 0.287 | 0.218 | 0.265 |

**성능 하락률**

| 모델 | Noise | Blur | LowRes |
|---|---:|---:|---:|
| YOLOv8m-VID | -14.7% | **-38.3%** | -20.3% |
| RT-DETR-VID | -8.9% | **-30.9%** | -15.9% |

**DET vs VID 절대 성능 차이 원인**

VID 모델의 절대 성능이 DET보다 낮다 (YOLOv8m 기준 0.666 → 0.387).

1. **더 어려운 데이터**: VisDrone-VID는 비디오 시퀀스 특성상 객체가 더 작고 밀집되어 있음
2. **추가 도전 요소**: 빠른 움직임, 가려짐(occlusion), 자세 급변
3. **학습-평가 도메인 불일치**: VID로 학습하고 DET 테스트셋으로 평가 (domain gap)

---

### 8-2. 비디오 모델 Augmentation 결과

**mAP@50**

| 모델 | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| YOLOv8m-VID_aug | 0.409 | 0.391 | 0.372 | 0.385 |
| RT-DETR-VID_aug | 0.335 | 0.319 | **0.320** | 0.310 |

**성능 하락률 (Augmented)**

| 모델 | Noise | Blur | LowRes |
|---|---:|---:|---:|
| YOLOv8m-VID_aug | -4.3% | -9.1% | -5.8% |
| RT-DETR-VID_aug | -4.7% | **-4.5%** | -7.4% |

**핵심 인사이트**

1. **DET와 동일한 패턴 재현**: Blur 최취약, Augmentation 일관 효과적
   - 학습 도메인(이미지 vs 비디오)이 달라도 강건성 전략의 효과는 동일

2. **RT-DETR-VID_aug의 Blur 강건성이 특히 탁월**:
   - Blur 하락률 단 -4.5% (4개 비디오 모델 중 최저)
   - Clean → Blur 0.335 → 0.320 (고작 0.015 차이)
   - Transformer + Augmentation 시너지 효과가 비디오에서도 유효

3. **비디오 도메인에서도 Augmentation 효과 검증됨**:
   - YOLOv8m Blur 하락률: -38.3% → -9.1% (4.2배 강건해짐)
   - DET에서의 4.4배 개선과 거의 동일

---

### 8-3. DET vs VID Augmentation 효과 비교

| 모델 | DET Blur 개선 | VID Blur 개선 |
|---|---:|---:|
| YOLOv8m | +0.175 mAP50 | +0.133 mAP50 |
| RT-DETR-L | +0.127 mAP50 | +0.102 mAP50 |

절대 개선폭은 DET가 크지만, **상대적 강건성 향상 비율은 VID에서도 동일 수준**.

---

## 9. 전체 종합 결론

### 9-1. 핵심 발견 요약

**발견 1: "Clean 성능 ≠ 실환경 강건성"**

```
YOLOv8m Baseline:
  Clean 0.666 (1위) → Blur 0.432 (-35.1%)

RT-DETR Baseline:
  Clean 0.536 (3위) → Blur 0.397 (-26.0%)

→ Clean에서 1위인 모델이 실환경에서도 1위가 아님
```

**발견 2: Corruption Augmentation은 비용 대비 효과가 극적**

```
구현 비용: 학습 코드 수 줄 추가
효과:
  YOLOv8m Blur 하락률: -35.1% → -7.9% (4.4배 강건)
  RT-DETR  Blur 하락률: -26.0% → -9.4% (2.8배 강건)
  Clean 성능: 유지 또는 오히려 향상
```

**발견 3: 열화 유형별 최적 전략이 다름**

| 열화 조건 | 최적 전략 | 근거 |
|---|---|---|
| Noise | Augmented | Restoration이 over-smoothing으로 역효과 |
| Blur | Restored (or Augmented) | U-Net이 blur 제거에 탁월 |
| LowRes | 둘 다 유효 | 비슷한 수준의 개선 |
| 혼합/미지 | **Augmented** | 범용적으로 안정적 |

**발견 4: 결과가 비디오 도메인에서도 일반화됨**

이미지 탐지에서 발견한 패턴이 비디오 탐지에서도 동일하게 재현됨:
- Blur 최취약, Augmentation 일관 효과적
- 도메인이 달라도 강건성 전략의 유효성은 유지됨

---

### 9-2. 실용적 권장사항

**일반 배포 시나리오 (다양한 열화 가능)**
→ **YOLOv8m + Corruption Augmentation**
- 절대 성능 + 강건성 모두 1위
- 빠른 추론 속도로 실시간 적용 가능

**Blur가 주된 환경 (드론 진동이 심한 경우)**
→ **U-Net Restoration + 탐지 파이프라인**
- Blur에서 Augmented보다 더 높은 성능
- 단, Noise가 없는 환경에서만 권장

**Transformer 모델의 강건성 우선**
→ **RT-DETR-L + Corruption Augmentation**
- 절대 성능은 낮지만 Blur 강건성 지표 최고 수준
- 비디오 환경(RT-DETR-VID_aug Blur -4.5%)에서 특히 탁월

---

### 9-3. 연구적 시사점

1. **단순한 augmentation으로도 큰 효과**: 복잡한 adversarial training 없이 50% 확률 corruption만으로 충분
2. **아키텍처에 따라 augmentation 효과가 다름**:
   - Transformer: Clean 성능도 함께 향상 (정규화 효과)
   - CNN: Clean은 유지하면서 열화 조건에서만 크게 향상
3. **최적 전략은 배포 환경의 열화 유형에 따라 선택**해야 함
4. **Adaptive Pipeline의 가능성**: 입력 열화 유형을 분류(classifier)한 후 조건별 최적 전략 선택 → 잠재적 최선
5. **드론/항공 분야 적용 가치**: 방산, 항공 감시, 재난 탐지 시스템에서 실환경 강건성은 핵심 요구사항

---

### 9-4. 전체 실험 규모 요약

| 구분 | 학습 | 평가 |
|---|---|---|
| DET Baseline | 3개 모델 | 3 x 4 = 12회 |
| DET Augmented | 3개 모델 | 3 x 4 = 12회 |
| DET Restored | U-Net 1개 | 3 x 4 = 12회 |
| VID Baseline | 2개 모델 | 2 x 4 = 8회 |
| VID Augmented | 2개 모델 | 2 x 4 = 8회 |
| **합계** | **총 11개 모델 학습** | **총 52회 평가** |

---

*이 문서는 프로젝트의 모든 개념적 내용을 포괄적으로 정리한 참조 문서입니다.*
*각 실험의 상세 수치는 `experiments/eval_results.csv`, `experiments/vid_eval_results.csv`를 참조하세요.*
