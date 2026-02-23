# 포트폴리오 정리: Robust Object Detection

---

## 1. 프로젝트 한 줄 요약

**드론 항공 영상에서 노이즈, 블러, 저해상도 등 실환경 열화 조건이 객체 탐지 성능에 미치는 영향을 정량 분석하고, 3가지 강건성 개선 전략(Corruption Augmentation, Image Restoration)을 비교 검증한 프로젝트.**

---

## 2. 프로젝트를 하게 된 이유

방산/항공 분야에서 드론이나 감시 카메라의 객체 탐지 시스템은 이상적인 환경이 아닌 **실전 환경**에서 동작해야 한다. 카메라 진동으로 인한 블러, 센서 노이즈, 저해상도 등이 불가피한데, 대부분의 객체 탐지 연구는 깨끗한(Clean) 이미지 기준 성능만 보고한다.

"Clean에서 잘 되는 모델이 실제로도 잘 될까?"라는 질문에서 출발하여, 이를 **실험적으로 검증**하고 **개선 방법을 비교**하는 것이 목표였다.

---

## 3. 내가 한 것 (전체 흐름)

### Phase 1: 실험 환경 구축 및 Baseline 평가

- VisDrone-DET 데이터셋을 COCO/YOLO 포맷으로 변환 (6개 클래스 선별)
- 3개 모델(Faster R-CNN, RT-DETR-L, YOLOv8m) Baseline 학습
- 노이즈/블러/저해상도 테스트셋 생성 (각 548장)
- 3 모델 x 4 조건 = 12회 평가 → 성능 하락 정량화

### Phase 2: Corruption Augmentation

- 학습 시 50% 확률로 corruption을 적용하여 강건성 향상
- 3개 모델 모두 augmented 버전 재학습
- 6 모델 x 4 조건 = 24회 평가 → Baseline 대비 개선 효과 검증

### Phase 3: Image Restoration (U-Net 전처리)

- 경량 U-Net (3.70M 파라미터) 설계 및 학습
- 열화 이미지를 복원한 후 기존 Baseline 모델로 추론하는 파이프라인 구축
- 3 모델 x 4 조건 = 12회 추가 평가

### Phase 4: 종합 비교 및 시각화

- 총 36회 평가 결과를 9개 차트로 시각화
- 데모 비교 이미지 15장 생성
- 종합 보고서 및 문서화

---

## 4. 핵심 결과 (면접에서 말할 수치)

### "Clean에서 잘 되는 모델이 실환경에서도 잘 되나요?"

**아니요.** Faster R-CNN은 Blur 조건에서 mAP가 **0.532 → 0.287로 46.1% 하락**했습니다. Clean 성능이 높다고 실환경 강건성이 보장되지 않습니다.

### "Corruption Augmentation의 효과는?"

- YOLOv8m: Blur 하락률 **-35.1% → -7.9%** (4.4배 강건해짐)
- Clean 성능은 유지하면서 모든 열화 조건에서 일관되게 개선
- 구현이 간단 (학습 시 50% 확률로 corruption만 추가)

### "Image Restoration의 효과는?"

- Blur에서 **최고 성능** (YOLOv8m: 0.432 → 0.640, +0.208)
- 하지만 Noise에서는 오히려 **0.577 → 0.201로 65% 악화** (over-smoothing)
- 전략마다 강점/약점이 다르다는 것이 핵심 인사이트

### "최종 권장 전략은?"

- **범용 강건성**: Corruption Augmentation (모든 조건에서 안정적)
- **Blur 특화**: Image Restoration (deblurring에 최적)
- **실전 배포**: YOLOv8m + Augmentation (성능 + 속도 + 강건성)

---

## 5. 기술적으로 어려웠던 점과 해결 과정

### 5-1. RT-DETR 추론 시 GPU 디바이스 에러

**문제**: RT-DETR 모델을 GPU에서 추론할 때 `RuntimeError: Expected all tensors to be on the same device` 에러 발생.

**원인 분석**: Ultralytics의 `RTDETRDecoder` 클래스에서 `valid_mask`와 `shapes`가 **클래스 레벨 속성**으로 정의되어 있어, 모델을 `.cuda()`로 옮겨도 CPU에 남아 있었다. 일반적인 `nn.Parameter`나 `nn.Buffer`와 달리 `.to(device)` 호출에 자동으로 이동하지 않는 파이썬 속성이었다.

**해결**: 모델 로딩 후 해당 모듈의 `m.shapes = []`로 초기화하여 CPU 텐서 참조를 제거. 추론 시 자동으로 GPU에서 재생성되도록 함.

**배운 점**: 프레임워크 내부 구현을 파악하는 디버깅 능력의 중요성. 에러 메시지만으로는 원인을 알 수 없었고, Ultralytics 소스코드의 `RTDETRDecoder` 클래스를 직접 분석해야 했다.

### 5-2. Ultralytics 학습 파이프라인에 Custom Augmentation 주입

**문제**: RT-DETR과 YOLOv8은 Ultralytics 프레임워크의 자체 학습 루프를 사용하는데, 이 파이프라인에 커스텀 corruption augmentation을 삽입할 공식적인 인터페이스가 없었다.

**해결**: `ultralytics.data.augment.Albumentations.__call__` 메서드를 **monkey-patching**하여, 기존 augmentation 파이프라인(mosaic, mixup 등)은 유지하면서 corruption을 추가 적용. `labels["img"]` (numpy BGR 배열)에 직접 접근하여 corruption 적용.

**배운 점**: 프레임워크를 포크하지 않고도 핵심 동작을 수정할 수 있는 기법. 다만 프레임워크 업데이트 시 호환성 문제가 생길 수 있으므로, 버전을 고정하고 테스트를 철저히 해야 한다.

### 5-3. U-Net 채널 불일치 에러

**문제**: Restoration U-Net 구현 시 `RuntimeError: expected input to have 256 channels, but got 384 channels` 에러 발생.

**원인 분석**: Decoder의 UpBlock에서 `ConvTranspose2d`로 업샘플링한 텐서와 skip connection 텐서를 concat할 때, 채널 수 계산이 잘못되었다. 기존 코드는 `ConvBlock(out_ch * 2, out_ch)`로 skip과 upsampled가 같은 채널 수라고 가정했으나, 실제로는 다른 경우가 있었다.

**해결**: `UpBlock`의 생성자를 `(in_ch, skip_ch, out_ch)`로 분리하여, 업샘플링 채널과 skip 채널을 명시적으로 지정. `ConvBlock(in_ch + skip_ch, out_ch)`로 정확한 채널 연산.

**배운 점**: U-Net 같은 encoder-decoder 구조에서 채널 수 추적이 중요하다. 특히 skip connection이 있는 구조에서는 각 레이어의 입출력 채널을 명확하게 문서화하면서 구현해야 한다.

### 5-4. Restoration이 Noise에서 오히려 성능을 악화시킨 문제

**문제**: U-Net으로 복원한 이미지에서 탐지 성능이 Noise 조건에서 Baseline보다도 **크게 하락** (0.577 → 0.201).

**원인 분석**: U-Net이 3가지 corruption(Noise/Blur/LowRes)을 동시에 학습했는데, 노이즈 제거 과정에서 **객체의 텍스처와 엣지 정보까지 과도하게 제거**(over-smoothing). 특히 소형 객체(pedestrian, motor)의 미세한 특징이 사라져서 탐지가 불가능해짐.

**해결/인사이트**: 이것은 "해결"이라기보다 **중요한 연구 발견**이었다. 단일 복원 모델이 모든 열화를 효과적으로 처리하기는 어렵다는 것을 실험적으로 입증. 이 결과를 통해 **"각 전략에는 고유한 trade-off가 있으며, 운용 환경에 맞는 전략 선택이 중요하다"**는 결론을 도출할 수 있었다.

**배운 점**: "모든 상황에서 잘 되는 만능 솔루션"은 없다. 실제 시스템 설계에서는 예상되는 열화 유형에 맞는 전략을 선택하거나, 열화 유형을 분류한 후 조건별 최적 전략을 적용하는 adaptive pipeline이 필요하다.

### 5-5. Faster R-CNN의 극단적인 Blur 취약성 분석

**문제**: Faster R-CNN이 Blur에서 -46.1% 하락 (다른 모델 대비 압도적으로 취약).

**원인 분석**: 2-Stage 구조의 **cascade failure** 특성. 1단계 RPN(Region Proposal Network)이 블러된 이미지에서 영역 제안에 실패하면, 2단계 분류/회귀로 넘어갈 기회 자체가 사라진다. 즉, 에러가 단계별로 **누적**되는 구조적 취약점.

반면 YOLOv8(1-Stage)은 한 번의 forward pass에서 직접 검출하므로 이러한 누적 에러가 없고, RT-DETR(Transformer)은 전역 어텐션으로 지역 정보 손실을 보완할 수 있었다.

**배운 점**: 아키텍처 설계 자체가 강건성에 구조적 영향을 미친다. 단순히 "정확도가 높은 모델"이 아니라, 운용 환경의 특성(열화 유형, 실시간 요구 등)에 맞는 아키텍처를 선택해야 한다.

---

## 6. 이 프로젝트에서 배운 것

### 기술적 역량

| 역량 | 상세 |
|---|---|
| **객체 탐지 모델 학습/평가** | Faster R-CNN(torchvision), RT-DETR, YOLOv8(Ultralytics) 3가지 프레임워크 경험 |
| **COCO 평가 체계** | pycocotools의 COCOeval을 활용한 mAP 계산, 클래스별/IoU별 분석 |
| **이미지 전처리/복원** | U-Net 기반 image restoration, PSNR/SSIM 품질 평가 |
| **실험 설계** | 변인 통제 (동일 하이퍼파라미터, 동일 데이터, 전략만 변경) |
| **데이터 파이프라인** | VisDrone → COCO/YOLO 변환, corruption 적용, 데이터 로더 구현 |
| **시각화** | matplotlib/seaborn으로 비교 차트, 히트맵, 레이더 차트 등 9종 |

### 연구적 인사이트

1. **Clean 성능 ≠ 실환경 강건성**: 벤치마크 성능만으로 실전 배포를 결정하면 위험하다
2. **단순한 augmentation이 효과적**: 복잡한 기법 없이 50% corruption만으로도 강건성을 4배 이상 개선 가능
3. **전략마다 trade-off 존재**: 만능 솔루션은 없고, 운용 환경에 맞는 선택이 중요
4. **아키텍처가 강건성에 구조적 영향**: CNN의 지역적 특징 vs Transformer의 전역 어텐션
5. **소형 객체가 가장 취약**: 열화에 의한 정보 손실이 소형 객체에 치명적

### 엔지니어링 역량

- 프레임워크 내부 코드 분석 및 디버깅 (Ultralytics RTDETRDecoder 버그 대응)
- Monkey-patching을 통한 프레임워크 확장
- GPU 메모리 관리 (모델 순차 로딩, `torch.cuda.empty_cache()`, `gc.collect()`)
- 체계적 실험 관리 (JSON 결과 저장, 자동화된 평가 스크립트)

---

## 7. 면접 예상 질문 및 답변

### Q1. "이 프로젝트의 핵심 기여가 무엇인가요?"

3가지 서로 다른 강건성 전략(Baseline, Corruption Augmentation, Image Restoration)을 동일 조건에서 체계적으로 비교한 것입니다. 총 36회 평가를 통해 각 전략의 강점과 한계를 정량적으로 규명했고, 운용 환경에 따른 최적 전략 선택 가이드를 도출했습니다.

### Q2. "왜 이 3개 모델을 선택했나요?"

객체 탐지의 3가지 주요 아키텍처 패러다임을 대표하기 위해서입니다. Faster R-CNN(2-Stage CNN), YOLOv8(1-Stage CNN), RT-DETR(Transformer). 이를 통해 아키텍처 자체가 열화 강건성에 미치는 영향을 비교할 수 있었습니다. 실제로 Transformer(RT-DETR)의 전역 어텐션이 CNN 대비 상대적으로 강건하다는 것을 확인했습니다.

### Q3. "Corruption Augmentation이 왜 효과적인가요?"

모델이 학습 과정에서 열화된 이미지를 경험하면, 열화에 불변하는(invariant) 특징을 학습하게 됩니다. 예를 들어, 블러된 자동차의 전체적인 형태와 맥락 정보를 활용하는 법을 학습하여, 선명한 텍스처에 과도하게 의존하지 않게 됩니다. 또한 일종의 정규화(regularization) 효과도 있어 RT-DETR의 경우 Clean 성능까지 +4.2%p 향상되었습니다.

### Q4. "Image Restoration이 Noise에서 실패한 이유는?"

U-Net이 노이즈를 제거하는 과정에서 객체의 텍스처와 미세한 엣지 정보까지 함께 제거하는 over-smoothing 현상이 발생했기 때문입니다. 특히 소형 객체(보행자, 오토바이)에서 치명적이었습니다. 이는 단일 모델이 서로 다른 성격의 열화(공간적 블러 vs 주파수적 노이즈)를 동시에 최적으로 처리하기 어렵다는 본질적 한계입니다.

### Q5. "실제 방산 시스템에 적용한다면?"

드론의 주 열화 원인은 카메라 진동에 의한 blur입니다. 이 경우 U-Net restoration 전처리가 가장 효과적입니다. 다만 다양한 열화가 혼합되는 환경이라면 Corruption Augmentation이 더 안전한 선택입니다. 이상적으로는 입력 영상의 열화 유형을 먼저 분류한 후, 조건별 최적 전략을 적용하는 adaptive pipeline이 최선입니다.

### Q6. "프로젝트에서 가장 어려웠던 점은?"

RT-DETR 모델의 GPU 디바이스 에러를 해결하는 과정이 가장 어려웠습니다. 에러 메시지만으로는 원인을 파악할 수 없어서 Ultralytics 프레임워크의 소스코드(RTDETRDecoder)를 직접 분석해야 했습니다. 클래스 레벨 속성이 `.to(device)` 호출에 의해 자동으로 이동하지 않는다는 것을 발견하고, 해당 속성을 초기화하는 workaround를 적용했습니다. 이 경험을 통해 오픈소스 프레임워크를 "블랙박스"가 아닌 "수정 가능한 코드"로 다루는 역량을 키웠습니다.

### Q7. "한계점이 있다면?"

세 가지 한계가 있습니다. 첫째, 열화 파라미터가 고정값(sigma=15, kernel=9 등)이어서, 다양한 severity에 대한 분석이 부족합니다. 둘째, 안개, 비, 야간 등 방산 특화 열화 조건은 다루지 못했습니다. 셋째, Augmented 모델과 Restoration을 조합하는 전략(Augmented + U-Net)은 실험하지 못했는데, 이것이 Noise와 Blur 모두에서 최적일 가능성이 있습니다.

---

## 8. 프로젝트 규모 요약

| 항목 | 수치 |
|---|---|
| 학습한 모델 수 | 7개 (Baseline 3 + Augmented 3 + U-Net 1) |
| 총 평가 횟수 | 36회 |
| 생성한 시각화 | 차트 9개 + 데모 이미지 15장 |
| 코드 파일 수 | 17개 스크립트 |
| 문서 | 7개 (보고서 포함) |
| 사용 프레임워크 | PyTorch, torchvision, Ultralytics, pycocotools |
| GPU | NVIDIA RTX 3070 Ti (8GB) |

---

## 9. 기술 스택

```
Language     : Python 3.11
Deep Learning: PyTorch 2.5.1, torchvision, Ultralytics 8.3.209
Models       : Faster R-CNN (ResNet-50 FPN v2), RT-DETR-L, YOLOv8m
Evaluation   : pycocotools (COCO mAP)
Restoration  : Custom U-Net (L1 + SSIM Loss)
Visualization: matplotlib, seaborn
Data         : VisDrone-DET, OpenCV
GPU          : NVIDIA RTX 3070 Ti (CUDA 12.1)
VCS          : Git, GitHub
```
