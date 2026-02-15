# Baseline vs Augmented 최종 비교 분석

## 1. 실험 개요

### 목적

Corruption augmentation이 객체 탐지 모델의 강건성을 얼마나 개선하는지 정량적으로 검증한다.

### 실험 구조

| 항목 | 내용 |
|---|---|
| 모델 | Faster R-CNN, RT-DETR-L, YOLOv8m (각각 Baseline + Augmented = 6개) |
| 테스트셋 | Clean, Noise, Blur, LowRes (4종) |
| 총 평가 | 6 모델 x 4 테스트셋 = **24회** |
| 평가 스크립트 | `scripts/eval_all.py` |
| 결과 파일 | `experiments/eval_results.json`, `experiments/eval_results.csv` |

---

## 2. 전체 결과

### 2-1. mAP@50

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN | 0.5318 | 0.4716 | 0.2868 | 0.4535 |
| **FasterRCNN_aug** | **0.5403** | **0.5143** | **0.4424** | **0.4865** |
| RT-DETR-L | 0.5359 | 0.4748 | 0.3967 | 0.5004 |
| **RT-DETR-L_aug** | **0.5779** | **0.5471** | **0.5238** | **0.5426** |
| YOLOv8m | **0.6657** | 0.5766 | 0.4322 | 0.6279 |
| **YOLOv8m_aug** | 0.6596 | **0.6398** | **0.6075** | **0.6393** |

### 2-2. mAP@50-95

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN | 0.3362 | 0.2961 | 0.1696 | 0.2828 |
| **FasterRCNN_aug** | **0.3460** | **0.3241** | **0.2728** | **0.3062** |
| RT-DETR-L | 0.3376 | 0.2941 | 0.2312 | 0.3121 |
| **RT-DETR-L_aug** | **0.3765** | **0.3557** | **0.3352** | **0.3507** |
| YOLOv8m | **0.4462** | 0.3766 | 0.2650 | 0.4168 |
| **YOLOv8m_aug** | 0.4443 | **0.4269** | **0.3993** | **0.4276** |

---

## 3. 강건성 분석

### 3-1. Clean 대비 성능 하락률 (mAP@50)

| Model | Noise | Blur | LowRes |
|---|---:|---:|---:|
| FasterRCNN | -11.3% | **-46.1%** | -14.7% |
| FasterRCNN_aug | -4.8% | -18.1% | -10.0% |
| RT-DETR-L | -11.4% | -26.0% | -6.6% |
| RT-DETR-L_aug | -5.3% | -9.4% | -6.1% |
| YOLOv8m | -13.4% | -35.1% | -5.7% |
| **YOLOv8m_aug** | **-3.0%** | **-7.9%** | **-3.1%** |

### 3-2. Augmented - Baseline 차이 (mAP@50)

| Model | Clean | Noise | Blur | LowRes |
|---|---:|---:|---:|---:|
| FasterRCNN | +0.0085 | +0.0427 | **+0.1556** | +0.0330 |
| RT-DETR-L | +0.0420 | +0.0723 | **+0.1271** | +0.0422 |
| YOLOv8m | -0.0061 | +0.0632 | **+0.1753** | +0.0115 |

---

## 4. 핵심 발견

### 4-1. Blur 강건성 대폭 개선 (핵심 성과)

3개 모델 모두 Blur에서 가장 극적인 개선을 달성했다.

| Model | Baseline Blur | Augmented Blur | 개선폭 | 하락률 변화 |
|---|---:|---:|---:|---|
| FasterRCNN | 0.2868 | 0.4424 | **+0.1556** | -46.1% -> -18.1% |
| RT-DETR-L | 0.3967 | 0.5238 | **+0.1271** | -26.0% -> -9.4% |
| YOLOv8m | 0.4322 | 0.6075 | **+0.1753** | -35.1% -> -7.9% |

- Faster R-CNN: 사실상 탐지 불가(0.287) -> 실용 가능 수준(0.442)으로 회복
- YOLOv8m: 하락률 -35.1%에서 -7.9%로, **약 4.4배 강건해짐**
- RT-DETR-L: 하락률 -26.0%에서 -9.4%로, **약 2.8배 강건해짐**

### 4-2. Clean 성능 유지 (과적합 방지 효과)

| Model | Baseline Clean | Augmented Clean | 차이 |
|---|---:|---:|---:|
| FasterRCNN | 0.5318 | 0.5403 | **+0.0085** |
| RT-DETR-L | 0.5359 | 0.5779 | **+0.0420** |
| YOLOv8m | 0.6657 | 0.6596 | -0.0061 |

- RT-DETR-L: Clean에서도 +4.2%p 향상 (corruption augmentation의 정규화 효과)
- FasterRCNN: Clean에서 +0.9%p 미세 향상
- YOLOv8m: -0.6%p 미세 하락 (통계적으로 유의미하지 않은 수준)
- **결론: Clean 성능을 희생하지 않으면서 강건성을 대폭 개선**

### 4-3. Noise/LowRes도 일관되게 개선

**Noise:**
- FasterRCNN: +0.0427 (하락률 -11.3% -> -4.8%)
- RT-DETR-L: +0.0723 (하락률 -11.4% -> -5.3%)
- YOLOv8m: +0.0632 (하락률 -13.4% -> -3.0%)

**LowRes:**
- FasterRCNN: +0.0330 (하락률 -14.7% -> -10.0%)
- RT-DETR-L: +0.0422 (하락률 -6.6% -> -6.1%)
- YOLOv8m: +0.0115 (하락률 -5.7% -> -3.1%)

### 4-4. YOLOv8m_aug가 종합 최강

| 관점 | 최고 모델 | 수치 |
|---|---|---|
| Clean mAP@50 | YOLOv8m (baseline) | 0.666 |
| Noise mAP@50 | **YOLOv8m_aug** | 0.640 |
| Blur mAP@50 | **YOLOv8m_aug** | 0.608 |
| LowRes mAP@50 | **YOLOv8m_aug** | 0.639 |
| 최소 하락률 | **YOLOv8m_aug** | -3.0% ~ -7.9% |

YOLOv8m_aug는 모든 corruption 조건에서 절대 성능 1위이면서, 하락률도 가장 낮다.

---

## 5. 클래스별 분석 (Blur 기준)

### 5-1. Blur에서 클래스별 AP@50 비교

| Class | FRCNN | FRCNN_aug | RT-DETR | RT-DETR_aug | YOLOv8 | YOLOv8_aug |
|---|---:|---:|---:|---:|---:|---:|
| pedestrian | 0.167 | **0.354** | 0.254 | **0.472** | 0.289 | **0.535** |
| car | 0.628 | **0.755** | 0.734 | **0.833** | 0.772 | **0.861** |
| van | 0.237 | **0.411** | 0.369 | **0.455** | 0.394 | **0.537** |
| truck | 0.180 | **0.295** | 0.297 | **0.398** | 0.305 | **0.461** |
| bus | 0.348 | **0.487** | 0.438 | **0.505** | 0.552 | **0.686** |
| motor | 0.160 | **0.353** | 0.288 | **0.479** | 0.281 | **0.565** |

### 5-2. 클래스별 개선 패턴

**가장 큰 개선: 소형 객체 (pedestrian, motor)**
- pedestrian: FRCNN +0.187, RT-DETR +0.218, YOLOv8 +0.246
- motor: FRCNN +0.193, RT-DETR +0.191, YOLOv8 +0.284

소형 객체는 Blur에서 거의 탐지 불가 수준이었으나, augmentation 후 실용 가능 수준으로 회복.

**일관된 개선: 대형 객체 (car, bus)**
- car: 이미 높은 수준에서 추가 향상 (0.63~0.77 -> 0.76~0.86)
- bus: 전 모델에서 +0.07~0.14 향상

---

## 6. 아키텍처별 분석

### 6-1. Faster R-CNN (2-Stage CNN)

- **가장 큰 약점이 가장 크게 개선됨**: Blur 하락률 -46.1% -> -18.1%
- Cascade failure 문제 완화: RPN이 blur된 이미지에서 제안 생성 능력 학습
- 그러나 여전히 3개 모델 중 가장 취약 (Blur에서 mAP 0.442)
- Clean 성능도 소폭 향상 (+0.9%p): 정규화 효과

### 6-2. RT-DETR-L (Transformer)

- **Clean 성능까지 크게 향상**: 0.536 -> 0.578 (+4.2%p)
- Transformer의 전역 어텐션이 corruption augmentation과 시너지
- 다양한 조건의 이미지를 학습하면서 더 일반화된 표현 학습
- 모든 조건에서 균형 잡힌 개선

### 6-3. YOLOv8m (1-Stage CNN)

- **절대 성능과 강건성 모두 최고**: 실용 배포에 가장 적합
- Clean에서 유일하게 미세 하락 (-0.6%p): 이미 높은 Clean 성능에 의한 trade-off
- Blur 하락률 -35.1% -> -7.9%: 가장 극적인 강건성 개선
- 1-stage 구조의 효율성 + corruption 학습의 효과가 결합

---

## 7. 종합 순위 (최종)

### mAP@50 기준

| 순위 | Clean | Noise | Blur | LowRes |
|---|---|---|---|---|
| 1위 | YOLOv8m (0.666) | YOLOv8m_aug (0.640) | YOLOv8m_aug (0.608) | YOLOv8m_aug (0.639) |
| 2위 | YOLOv8m_aug (0.660) | YOLOv8m (0.577) | RT-DETR-L_aug (0.524) | YOLOv8m (0.628) |
| 3위 | RT-DETR-L_aug (0.578) | RT-DETR-L_aug (0.547) | FasterRCNN_aug (0.442) | RT-DETR-L_aug (0.543) |

### 강건성 기준 (평균 하락률)

| 순위 | Model | Noise | Blur | LowRes | 평균 하락률 |
|---|---|---:|---:|---:|---:|
| **1위** | **YOLOv8m_aug** | -3.0% | -7.9% | -3.1% | **-4.7%** |
| 2위 | RT-DETR-L_aug | -5.3% | -9.4% | -6.1% | -6.9% |
| 3위 | FasterRCNN_aug | -4.8% | -18.1% | -10.0% | -11.0% |
| 4위 | RT-DETR-L | -11.4% | -26.0% | -6.6% | -14.7% |
| 5위 | YOLOv8m | -13.4% | -35.1% | -5.7% | -18.1% |
| 6위 | FasterRCNN | -11.3% | -46.1% | -14.7% | -24.0% |

---

## 8. 결론

### 8-1. 핵심 결론

1. **Corruption augmentation은 효과적이다**: 3개 모델 모두 모든 corruption 조건에서 성능 향상
2. **Clean 성능을 희생하지 않는다**: RT-DETR은 오히려 +4.2%p 향상, 나머지도 동등
3. **Blur가 가장 극적으로 개선된다**: 평균 +0.153 mAP@50 향상 (3개 모델 평균)
4. **YOLOv8m_aug가 실용 배포에 최적**: 절대 성능 + 강건성 모두 1위
5. **구현이 간단하다**: 학습 파이프라인에 50% 확률 corruption만 추가하면 됨

### 8-2. 실용적 권장사항

| 시나리오 | 권장 모델 | 근거 |
|---|---|---|
| 실시간 강건 탐지 | **YOLOv8m_aug** | 절대 성능 최고 + 강건성 최고 + 빠른 추론 |
| 강건성 우선 (리소스 여유) | RT-DETR-L_aug | 가장 균형 잡힌 개선, Clean도 +4.2%p |
| 레거시 시스템 개선 | FasterRCNN_aug | Blur -46% -> -18%로 실용 가능 수준 회복 |

### 8-3. 연구적 시사점

- **"Clean에서 잘 되는 모델이 실환경에서도 잘 된다"는 가정은 위험하다**
  - Baseline YOLOv8m: Clean 0.666 but Blur 0.432 (-35.1%)
  - Augmented YOLOv8m: Clean 0.660 but Blur 0.608 (-7.9%)
- **단순한 augmentation으로도 강건성을 크게 높일 수 있다**
  - 복잡한 adversarial training 없이 50% corruption만으로 효과적
- **아키텍처에 따라 augmentation 효과가 다르다**
  - Transformer(RT-DETR): Clean까지 향상 (정규화 효과)
  - CNN(YOLOv8): Clean 유지하면서 corruption에서만 크게 향상

---

*평가 수행일: 2026-02-12*
*평가 환경: NVIDIA GeForce RTX 3070 Ti (8GB), Python 3.11, PyTorch 2.5.1, Ultralytics 8.3.209*
