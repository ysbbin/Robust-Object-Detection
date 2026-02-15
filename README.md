# Robust Object Detection on VisDrone-DET

객체 탐지 모델의 **이미지 열화(corruption) 조건에서의 강건성(Robustness)**을 실험적으로 분석하고, corruption augmentation을 통한 개선 효과를 검증하는 프로젝트.

## Research Question

> Clean 이미지에서 학습된 객체 탐지 모델은 실환경의 열화 조건(노이즈, 블러, 저해상도)에서도 성능을 유지하는가? 그리고 학습 시 corruption augmentation을 적용하면 강건성이 얼마나 개선되는가?

## Experiment Design

### Models (3 architectures)

| Model | Type | Framework |
|---|---|---|
| Faster R-CNN (ResNet-50 FPN v2) | 2-Stage CNN | torchvision |
| RT-DETR-L | Transformer | Ultralytics |
| YOLOv8m | 1-Stage CNN | Ultralytics |

### Dataset

- **VisDrone-DET** (drone-view object detection)
- 6 classes: pedestrian, car, van, truck, bus, motor
- COCO format (Faster R-CNN) + YOLO format (RT-DETR, YOLOv8)

### Test Conditions (4 types)

| Condition | Description | Parameter |
|---|---|---|
| Clean | Original images | - |
| Noise | Gaussian noise | sigma=15 |
| Blur | Motion blur | kernel=9, angle=0 deg |
| LowRes | Downscale + upscale | factor=0.5x |

### Training Variants (2 per model)

- **Baseline**: Clean 데이터로만 학습
- **Augmented**: 학습 시 50% 확률로 corruption(noise/blur/lowres 중 랜덤 1개) 적용

### Total Evaluations: 6 models x 4 test sets = 24

## Key Results

### mAP@50

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN | 0.532 | 0.472 | 0.287 | 0.454 |
| FasterRCNN_aug | 0.540 | 0.514 | **0.442** | 0.487 |
| RT-DETR-L | 0.536 | 0.475 | 0.397 | 0.500 |
| RT-DETR-L_aug | 0.578 | 0.547 | **0.524** | 0.543 |
| YOLOv8m | **0.666** | 0.577 | 0.432 | 0.628 |
| YOLOv8m_aug | 0.660 | **0.640** | **0.608** | **0.639** |

### Degradation from Clean (mAP@50)

| Model | Noise | Blur | LowRes |
|---|---:|---:|---:|
| FasterRCNN | -11.3% | -46.1% | -14.7% |
| FasterRCNN_aug | -4.8% | -18.1% | -10.0% |
| RT-DETR-L | -11.4% | -26.0% | -6.6% |
| RT-DETR-L_aug | -5.3% | -9.4% | -6.1% |
| YOLOv8m | -13.4% | -35.1% | -5.7% |
| **YOLOv8m_aug** | **-3.0%** | **-7.9%** | **-3.1%** |

### Key Findings

1. **Blur is the most critical corruption**: up to -46.1% mAP drop for Faster R-CNN baseline
2. **Corruption augmentation dramatically improves robustness**: Blur degradation reduced from -46.1% to -18.1% (FRCNN), -35.1% to -7.9% (YOLOv8)
3. **Clean performance is preserved**: augmented models maintain or improve Clean mAP
4. **YOLOv8m_aug achieves best overall robustness**: only -3.0% ~ -7.9% degradation across all conditions
5. **RT-DETR-L_aug shows best balanced improvement**: Clean +4.2%p while corruption robustness also improved

## Project Structure

```
Robust-Object-Detection/
├── scripts/
│   ├── convert_visdrone_to_coco.py      # VisDrone -> COCO format
│   ├── convert_visdrone_to_yolo.py      # VisDrone -> YOLO format
│   ├── build_corrupted_testsets.py      # Generate corrupted test sets
│   ├── coco_detection_dataset.py        # PyTorch Dataset for COCO format
│   ├── augmentations.py                 # Shared corruption augmentation module
│   ├── train_frcnn_baseline.py          # Faster R-CNN baseline training
│   ├── train_frcnn_augmented.py         # Faster R-CNN augmented training
│   ├── train_rtdetr_augmented.py        # RT-DETR augmented training
│   ├── train_yolo_augmented.py          # YOLOv8 augmented training
│   └── eval_all.py                      # Unified evaluation (6x4=24 runs)
├── experiments/
│   ├── frcnn/                           # Faster R-CNN results
│   ├── rtdetr/                          # RT-DETR results
│   ├── yolo/                            # YOLOv8 results
│   ├── eval_results.json                # Full evaluation results
│   └── eval_results.csv                 # Summary CSV
├── docs/
│   ├── 01_baseline_eval_results.md      # Baseline evaluation analysis
│   ├── 02_augmented_training.md         # Augmented training details
│   └── 03_final_comparison.md           # Final comparison analysis
└── data/                                # (gitignored)
    ├── processed/                       # Processed datasets
    └── testsets/                         # Corrupted test sets
```

## Setup

### Requirements

- Python 3.11+
- PyTorch 2.5+
- CUDA-compatible GPU (tested on RTX 3070 Ti 8GB)

```bash
pip install torch torchvision ultralytics pycocotools opencv-python numpy
```

### Data Preparation

```bash
# 1. Download VisDrone-DET dataset and place in data/raw/

# 2. Convert to COCO/YOLO format
python -m scripts.convert_visdrone_to_coco
python -m scripts.convert_visdrone_to_yolo

# 3. Generate corrupted test sets
python -m scripts.build_corrupted_testsets
```

### Training

```bash
# Baseline (clean only)
python -m scripts.train_frcnn_baseline

# Augmented (with corruption augmentation)
python -m scripts.train_frcnn_augmented
python -m scripts.train_rtdetr_augmented
python -m scripts.train_yolo_augmented
```

### Evaluation

```bash
python -m scripts.eval_all
```

## Environment

- GPU: NVIDIA GeForce RTX 3070 Ti (8GB)
- OS: Windows 11
- Python 3.11, PyTorch 2.5.1, Ultralytics 8.3.209
