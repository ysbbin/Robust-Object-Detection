# Robust Object Detection on VisDrone

객체 탐지 모델의 이미지 열화(corruption) 조건에서의 강건성(Robustness)을 실험적으로 분석하고, corruption augmentation을 통한 개선 효과를 검증하는 프로젝트.

**이미지 탐지(VisDrone-DET)**와 **비디오 탐지(VisDrone-VID)** 두 도메인에서 실험을 진행하였다.

## 연구 질문

> Clean 이미지에서 학습된 객체 탐지 모델은 실환경의 열화 조건(노이즈, 블러, 저해상도)에서도 성능을 유지하는가? 그리고 corruption augmentation 또는 이미지 복원 전처리를 적용하면 강건성이 얼마나 개선되는가? 해당 전략은 비디오 데이터에서도 동일하게 유효한가?

## 데모: Baseline vs Augmented (Blur 조건)

Blur 이미지에서 Baseline 모델은 대부분의 객체를 탐지하지 못하지만, Augmented 모델은 약 **2배 더 많은 객체를 탐지**합니다.

### Faster R-CNN
![FRCNN Demo](experiments/demo/FRCNN_img0161_gt102_base19_aug50.jpg)
> 정답(GT): 102개 | Baseline: **19개** (19%) | Augmented: **50개** (49%) — 2.6배 개선

### YOLOv8m
![YOLOv8m Demo](experiments/demo/YOLOv8m_img0366_gt170_base38_aug75.jpg)
> 정답(GT): 170개 | Baseline: **38개** (22%) | Augmented: **75개** (44%) — 2.0배 개선

### RT-DETR-L
![RT-DETR Demo](experiments/demo/RT-DETR_img0366_gt170_base94_aug162.jpg)
> 정답(GT): 170개 | Baseline: **94개** (55%) | Augmented: **162개** (95%) — 1.7배 개선

## 주요 결과

### 3가지 전략 비교 (Baseline vs Augmented vs Restored)

![3-Strategy Comparison](experiments/figures/three_strategy_comparison.png)

3가지 강건성 전략 비교:
- **Baseline**: Clean 이미지로만 학습
- **Augmented**: Corruption augmentation으로 학습
- **Restored**: U-Net으로 이미지 복원 후 Baseline 모델로 추론

| 모델 | 전략 | Clean | Noise | Blur | LowRes |
|---|---|---:|---:|---:|---:|
| FasterRCNN | Baseline | 0.532 | 0.472 | 0.287 | 0.454 |
| | Augmented | 0.540 | **0.514** | 0.442 | 0.487 |
| | Restored | 0.532 | 0.177 | **0.502** | 0.483 |
| RT-DETR-L | Baseline | 0.536 | 0.475 | 0.397 | 0.500 |
| | Augmented | **0.578** | **0.547** | **0.524** | **0.543** |
| | Restored | 0.536 | 0.233 | 0.514 | 0.509 |
| YOLOv8m | Baseline | **0.666** | 0.577 | 0.432 | 0.628 |
| | Augmented | 0.660 | **0.640** | 0.608 | 0.639 |
| | Restored | **0.666** | 0.201 | **0.640** | **0.642** |

### 전략별 개선 효과

![Strategy Improvement](experiments/figures/strategy_improvement.png)

### 조건별 최적 전략

![Best Strategy](experiments/figures/best_strategy_heatmap.png)

### 강건성 프로파일 (3가지 전략)

![Radar](experiments/figures/three_strategy_radar.png)

### Baseline vs Augmented 비교 (6개 모델)

![mAP@50 Comparison](experiments/figures/map50_comparison.png)

### Clean 대비 성능 하락률

![Degradation](experiments/figures/degradation_comparison.png)

### 클래스별 AP@50 (Blur 조건)

![Heatmap](experiments/figures/class_ap50_blur_heatmap.png)

## 핵심 발견

### 이미지 탐지 (VisDrone-DET)

1. **Blur가 가장 치명적인 열화**: Faster R-CNN baseline에서 최대 -46.1% mAP 하락
2. **Corruption Augmentation이 전반적으로 가장 강건한 전략**: 모든 열화 조건에서 일관되게 개선
3. **이미지 복원(Restored)은 Blur에 특히 효과적**: FRCNN +0.216, YOLOv8 +0.208 mAP 향상
4. **이미지 복원은 Noise에서 오히려 역효과**: U-Net 과도한 스무딩으로 텍스처 정보 파괴 → 성능 -62~65% 폭락
5. **YOLOv8m_aug가 전체적으로 최고 강건성**: 모든 조건에서 -3.0% ~ -7.9% 이내 하락
6. **열화 유형별 최적 전략이 다름**: Noise → Augmented, Blur → Restored, LowRes → 둘 다 유효

### 비디오 탐지 (VisDrone-VID)

7. **Corruption Augmentation이 비디오 도메인에서도 동일하게 효과적**: Blur 하락률 -38.3% → -9.1% (YOLOv8m), -30.9% → -4.5% (RT-DETR-L)
8. **RT-DETR-L_aug의 Blur 강건성이 특히 탁월**: Blur 조건 하락률 -4.5% — 비디오 모델 중 최저
9. **결과가 도메인을 초월해 일반화됨**: DET와 VID 모두 동일한 패턴 (Blur 최취약, Augmentation 일관 효과적)

## 실험 설계

### 이미지 탐지 (VisDrone-DET)

#### 사용 모델 (3종)

| 모델 | 유형 | 프레임워크 |
|---|---|---|
| Faster R-CNN (ResNet-50 FPN v2) | 2-Stage CNN | torchvision |
| RT-DETR-L | Transformer | Ultralytics |
| YOLOv8m | 1-Stage CNN | Ultralytics |

#### 데이터셋

- **VisDrone-DET** (드론 시점 객체 탐지)
- 6개 클래스: pedestrian, car, van, truck, bus, motor
- COCO 포맷 (Faster R-CNN) + YOLO 포맷 (RT-DETR, YOLOv8)

#### 강건성 전략 (3종)

- **Baseline**: Clean 데이터로만 학습 → 열화 이미지 직접 추론
- **Augmented**: 학습 시 50% 확률로 corruption(noise/blur/lowres 중 랜덤 1개) 적용
- **Restored**: 경량 U-Net (3.70M 파라미터)으로 이미지 복원 후 Baseline 모델로 추론
  - 복원 모델 성능: PSNR 34.03dB, SSIM 0.947

**총 평가 횟수: 9개 설정 x 4개 테스트셋 = 36회**

---

### 비디오 탐지 (VisDrone-VID)

#### 사용 모델 (2종)

| 모델 | 유형 | 프레임워크 |
|---|---|---|
| RT-DETR-L | Transformer | Ultralytics |
| YOLOv8m | 1-Stage CNN | Ultralytics |

#### 데이터셋

- **VisDrone-VID** (드론 시점 비디오 시퀀스, 프레임 단위 탐지)
- 6개 클래스: pedestrian, car, van, truck, bus, motor
- YOLO 포맷 (비디오 프레임 단위 추출)

#### 강건성 전략 (2종)

- **Baseline**: Clean 데이터로만 학습
- **Augmented**: 학습 시 50% 확률로 corruption 적용

#### 비디오 모델 결과 (mAP@50)

| 모델 | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| YOLOv8m-VID Baseline | 0.387 | 0.330 | 0.239 | 0.309 |
| YOLOv8m-VID Augmented | **0.409** | **0.391** | **0.372** | **0.385** |
| RT-DETR-VID Baseline | 0.316 | 0.287 | 0.218 | 0.265 |
| RT-DETR-VID Augmented | **0.335** | **0.319** | **0.320** | **0.310** |

![VID mAP@50 비교](experiments/figures/vid_map50_comparison.png)

![VID 성능 하락률](experiments/figures/vid_degradation_comparison.png)

**총 평가 횟수: 4개 설정 x 4개 테스트셋 = 16회**

---

### 테스트 조건 (공통, 4종)

| 조건 | 설명 | 파라미터 |
|---|---|---|
| Clean | 원본 이미지 | - |
| Noise | 가우시안 노이즈 | sigma=15 |
| Blur | 모션 블러 | kernel=9, angle=0도 |
| LowRes | 저해상도 (축소 후 복원) | 0.5x 축소 |

## 프로젝트 구조

```
Robust-Object-Detection/
├── scripts/
│   ├── augmentations.py                 # 공용 corruption augmentation 모듈
│   ├── convert_visdrone_to_coco.py      # VisDrone-DET → COCO 포맷 변환
│   ├── convert_visdrone_to_yolo.py      # VisDrone-DET → YOLO 포맷 변환
│   ├── convert_visdrone_vid_to_yolo.py  # VisDrone-VID → YOLO 포맷 변환 (프레임 추출)
│   ├── build_corrupted_testsets.py      # 열화 테스트셋 생성
│   ├── coco_detection_dataset.py        # COCO 포맷용 PyTorch Dataset
│   ├── train_frcnn_baseline.py          # Faster R-CNN baseline 학습
│   ├── train_frcnn_augmented.py         # Faster R-CNN augmented 학습
│   ├── train_rtdetr_augmented.py        # RT-DETR-L augmented 학습 (DET)
│   ├── train_yolo_augmented.py          # YOLOv8m augmented 학습 (DET)
│   ├── train_vid_yolo_baseline.py       # YOLOv8m baseline 학습 (VID)
│   ├── train_vid_yolo_augmented.py      # YOLOv8m augmented 학습 (VID)
│   ├── train_vid_rtdetr_baseline.py     # RT-DETR-L baseline 학습 (VID)
│   ├── train_vid_rtdetr_augmented.py    # RT-DETR-L augmented 학습 (VID)
│   ├── eval_all.py                      # DET 평가 (6x4=24회)
│   ├── eval_vid.py                      # VID 평가 (4x4=16회)
│   ├── restoration_net.py               # 복원 U-Net 모델 정의
│   ├── train_restoration.py             # 복원 모델 학습
│   ├── restore_testsets.py              # 테스트셋 복원 적용
│   ├── eval_restored.py                 # 복원 이미지 평가
│   ├── plot_results.py                  # DET 시각화: Baseline vs Augmented
│   ├── plot_three_strategies.py         # DET 시각화: 3가지 전략 비교
│   ├── plot_vid_results.py              # VID 시각화: Baseline vs Augmented
│   └── demo_inference.py               # 데모 비교 이미지 생성
├── experiments/
│   ├── frcnn/                           # Faster R-CNN 학습 결과 (DET)
│   ├── rtdetr/                          # RT-DETR-L 학습 결과 (DET)
│   ├── yolo/                            # YOLOv8m 학습 결과 (DET)
│   ├── vid_rtdetr/                      # RT-DETR-L 학습 결과 (VID)
│   ├── vid_yolo/                        # YOLOv8m 학습 결과 (VID)
│   ├── restoration/                     # 복원 모델 체크포인트
│   ├── figures/                         # 시각화 차트
│   ├── demo/                            # 데모 비교 이미지
│   ├── eval_results.json                # DET 평가 결과 (Baseline/Augmented)
│   ├── eval_restored_results.json       # DET 평가 결과 (Restored)
│   └── vid_eval_results.json            # VID 평가 결과
├── docs/
│   ├── 01_baseline_eval_results.md      # DET: Baseline 평가 분석
│   ├── 02_augmented_training.md         # DET: Augmented 학습 상세
│   ├── 03_final_comparison.md           # DET: 최종 비교 분석
│   ├── 04_visualization.md              # DET: 시각화 가이드
│   ├── 05_demo_inference.md             # DET: 데모 추론 분석
│   ├── 06_restoration_experiment.md     # DET: 이미지 복원 & 3전략 비교
│   └── 07_vid_experiment.md             # VID: 비디오 모델 실험 보고서
└── data/                                # (gitignored)
    ├── processed/                        # 전처리된 데이터셋
    └── testsets/                         # 열화 테스트셋
```

## 실행 방법

### 환경 설정

- Python 3.11+
- PyTorch 2.5+
- CUDA 지원 GPU (RTX 3070 Ti 8GB에서 테스트)

```bash
pip install torch torchvision ultralytics pycocotools opencv-python numpy matplotlib seaborn
```

### 데이터 준비

```bash
# 1. VisDrone-DET/VID 데이터셋 다운로드 후 data/ 하위에 배치

# 2. DET: COCO/YOLO 포맷 변환
python -m scripts.convert_visdrone_to_coco
python -m scripts.convert_visdrone_to_yolo

# 3. VID: YOLO 포맷 변환 (프레임 추출)
python -m scripts.convert_visdrone_vid_to_yolo

# 4. 열화 테스트셋 생성
python -m scripts.build_corrupted_testsets
```

### 이미지 탐지 학습 (VisDrone-DET)

```bash
# Baseline (clean 데이터로만 학습)
python -m scripts.train_frcnn_baseline

# Augmented (corruption augmentation 적용)
python -m scripts.train_frcnn_augmented
python -m scripts.train_rtdetr_augmented
python -m scripts.train_yolo_augmented
```

### 이미지 복원 모델 학습

```bash
# 복원 U-Net 학습
python -m scripts.train_restoration

# 테스트셋에 복원 적용
python -m scripts.restore_testsets
```

### 비디오 탐지 학습 (VisDrone-VID)

```bash
# Baseline
python -m scripts.train_vid_yolo_baseline
python -m scripts.train_vid_rtdetr_baseline

# Augmented
python -m scripts.train_vid_yolo_augmented
python -m scripts.train_vid_rtdetr_augmented
```

### 평가 및 시각화

```bash
# DET 평가: Baseline/Augmented (24회)
python -m scripts.eval_all

# DET 평가: Restored (12회)
python -m scripts.eval_restored

# VID 평가 (16회)
python -m scripts.eval_vid

# 시각화 차트 생성
python -m scripts.plot_results
python -m scripts.plot_three_strategies
python -m scripts.plot_vid_results

# 데모 비교 이미지 생성
python -m scripts.demo_inference
```

## 실험 환경

- GPU: NVIDIA GeForce RTX 3070 Ti (8GB)
- OS: Windows 11
- Python 3.11, PyTorch 2.5.1, Ultralytics 8.3.209
