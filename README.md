# 🛡️ Face Anti-Spoofing Project
> **멀티태스크 학습을 활용한 얼굴 위조 판별 및 공격 유형 분류 시스템**

본 프로젝트는 보안 시스템의 취약점인 '얼굴 위조 공격(Print, Replay, Mask)'을 탐지하기 위해 **CelebA-Spoof 데이터셋**을 분석하고, 멀티태스크 MobileNetV2와 3계층 XAI 파이프라인을 구축하는 것을 목표로 합니다.

## 🛠️ 개발 환경
- **Environment:** Google Colab (T4 GPU)
- **Language:** Python 3.x
- **Libraries:** TensorFlow 2.x, OpenCV, NumPy, Matplotlib, Kaggle API
- **Dataset:** [CelebA-Spoof (Kaggle)](https://www.kaggle.com/datasets/mabdullahsajid/celeba-spoofing)

---

## 📊 현재 성능 요약

| 모델 | Binary Acc | Spoof Type Acc | 비고 |
|------|-----------|---------------|------|
| 멀티태스크 MobileNetV2 | **96%** | **80%** | 최종 판정 모델 |
| 픽셀 단독 (FFT+Laplacian) | 50.3% | — | 판정 불가, XAI 전용 |
| CNN + Pixel 앙상블 실험 | 98.67% | — | CNN 단독과 동일 (Δ +0.00%p) |

---

## 🏗️ 시스템 아키텍처

```
[입력 이미지]
     ↓
[전처리] Haar Cascade 얼굴 크롭 → 224×224 / 정규화
     ↓
[MobileNetV2 멀티태스크]
  ├─ Binary Head: Real/Fake (96%)
  └─ Spoof Head: Live/Print/Replay/Mask (80%)
     ↓
[3계층 XAI 설명]
  ├─ Layer 1: Grad-CAM (logit 기반) — 어디를 봤는가
  ├─ Layer 2: 수치 앵커링 (FFT/Laplacian) — 얼마나 강한 신호인가
  └─ Layer 3: LLaVA 자연어 캡션 — 왜 그렇게 판단했는가 (예정)
     ↓
[최종 출력] 판정 + 신뢰도 + Grad-CAM 히트맵 + 수치 근거
```

---

## 📅 작업 단위 활동 로그

### [2026.04.29] 교수님 피드백 및 전략 수정

#### 1. 주요 피드백
- **"Black-box 문제":** 단순 Binary Classification은 판단 근거가 불분명
- CelebA-Spoof는 XAI 제작에 다소 무리가 있음 → 하이브리드 수치 보강 방향 제안

#### 2. 수정된 연구 방향
- **Hybrid Detection:** 딥러닝(MobileNetV2) + 수치 분석(FFT, Laplacian) 결합
- **Smart Sampling:** `intra_test` 프로토콜 기반 핵심 샘플 6,000장 선별
- **Explainable AI:** 판정 근거를 수치와 시각화로 설명하는 3계층 XAI 파이프라인

---

### [2026-05-07] 데이터 가용성 이슈 해결 및 메타데이터 파싱

- **Issue:** 62만 장(200GB) 전체 다운로드 불가, 외부 링크 404 에러
- **Solution:** `train_label.txt` 메타데이터 우선 확보 → 선택적 다운로드 파이프라인 구축
- **Result:** `data/subset/` 분석용 데이터 구축 완료

---

### [2026-05-11] 공격 유형별 수치 탐색 및 샘플 검증

#### Spoof Type 매핑 확정

| Type | 공격 유형 |
|------|----------|
| 0 | Live |
| 1, 2, 6, 8 | Print (인쇄 사진) |
| 3, 5, 7 | Replay (화면 재촬영) |
| 4, 9 | Paper Cut (오려낸 마스크) |
| 10 | 3D Mask (Toy) |

#### FFT + Laplacian 예비 탐색 결과 (20장)

| 카테고리 | FFT 고주파 에너지 | Laplacian 분산 |
|---------|----------------|--------------|
| Live | 645 | 350 |
| Print | 800 | 395 |
| Replay | 475 | 135 |
| Mask | 1490 | 1420 |

---

### [2026-05-12] Google Drive 연동 및 서브셋 다운로드 파이프라인 구축

- `BASE = '/content/drive/MyDrive/face-anti-spoofing'` 경로 고정으로 데이터 영구 유지
- 카테고리별 균형 샘플링 (random.seed(42) 고정)

#### 중간발표용 서브셋 구축 완료

| 카테고리 | 장수 | Spoof Type |
|---------|------|-----------|
| live | 1,500 | Type 0 |
| print | 1,500 | Type 1, 2, 6, 8 |
| replay | 1,500 | Type 3, 5, 7 |
| mask | 1,500 | Type 4, 9, 10 |

---

### [2026-05-12] RAM 부족 이슈 — MTCNN → Haar Cascade 전환

- **원인:** MTCNN이 이미지 1장당 500MB~1GB RAM 점유 → Colab 12GB 초과
- **해결:** Haar Cascade(OpenCV 내장)로 전환 → RAM 1/10, 속도 10배 향상
- **대가:** 정면 얼굴 검출 정확도 소폭 감소, 미검출 시 전체 이미지 224×224 리사이즈로 대체

---

### [2026-05-12] 픽셀 모듈 실험 및 구조적 문제점 점검

#### 픽셀 단독 성능 (6,000장)

| 카테고리 | Laplacian 평균 | FFT 평균 |
|---------|--------------|---------|
| Live | 383.2 | 1134.1 |
| Print | 318.4 | 1042.3 |
| Replay | 319.2 | 944.8 |
| Mask | 480.1 | 1134.6 |

- **픽셀 단독 Accuracy: 50.3% (랜덤 수준)**
- **교훈:** 카테고리 간 수치 분포가 겹쳐 단순 임계값으로 구분 불가

#### 트러블슈팅 목록 (TS)

| ID | 문제 | 해결 |
|----|------|------|
| TS-01 | MTCNN RAM 초과 | Haar Cascade 전환 |
| TS-02 | 데이터 불균형 (1:3) | class_weight 적용 |
| TS-03 | Laplacian/FFT 스케일 차이 | 정규화 후 concat |
| TS-04 | 픽셀 단독 판정 실패 (50.3%) | 역할 재정의 → XAI 수치 앵커링 전용 |

---

### [2026-05-12] Grad-CAM XAI 시각화 (중간발표)

#### 공격 유형별 Grad-CAM 판단 근거

| 공격 유형 | 집중 영역 | 판정 확률 |
|---------|---------|---------|
| Print Attack | 눈·코 주변 반사광 패턴 | 98~100% |
| Replay Attack | 이마·눈 픽셀 격자 패턴 | 74~100% |
| 3D Mask Attack | 마스크 경계선 텍스처 | 100% |
| Live (Real) | 히트맵 없음 (정상 판정) | 0% |

#### 중간발표 제출 결과물
- 멀티태스크 MobileNetV2 (Binary 96% / Spoof Type 80%)
- Grad-CAM 시각화 (공격 유형별 히트맵)
- Streamlit MVP (기본 동작)

---

### [2026-05-18] Phase 4-A: Logit 기반 Grad-CAM 개선

#### 문제: Sigmoid 포화 (중간발표 이슈)
- FAKE 100% 케이스에서 sigmoid 출력 기반 Grad-CAM → gradient ≈ 0 → 히트맵 미생성

#### 해결: Logit 기반 Grad-CAM
- sigmoid 직전 logit 값으로 gradient 계산 → FAKE 100% 케이스에서도 히트맵 정상 생성
- sigmoid vs logit 정성 비교 완료 (안경/마스크/조명 케이스)

#### 추가 구현: 수치 앵커링 선행 작업
- `anchored_pixel_stats()`: Grad-CAM 활성 영역(상위 30%) 마스크에 국소적 FFT/Laplacian 측정
- `src/gradcam_logit.py`로 모듈화 완료

---

### [2026-05-18] Phase 4-B: 하이브리드 앙상블 실험

#### 실험 설계
- CNN shared(256-d) 피처 + 픽셀 피처(3-d) concat → 앙상블 헤드 학습
- 목적: 픽셀 피처가 CNN 판정을 보완하는지 검증

#### 실험 결과 (동일 test set 300장 기준)

| 모델 | Accuracy | ROC-AUC |
|------|---------|--------|
| CNN 피처만 (새 헤드) | 98.67% | 0.9961 |
| CNN + Pixel concat | 98.67% | 0.9959 |
| **Δ** | **+0.00%p** | **-0.0002** |

#### 결론 (TS-04 최종 확정)
> MobileNetV2가 학습 과정에서 주파수 도메인 패턴을 이미 256차원 표현 공간에 내재화했음을 확인.  
> **픽셀 피처(FFT/Laplacian)는 판정에 사용하지 않고, Grad-CAM 활성 영역의 수치 앵커링(XAI Layer 2)으로만 활용.**

---

---

### [2026-05-19] Phase 4-C: 공격 유형별 FAR 분석

#### 최종 성능 결과

| 지표 | 결과 | 목표 |
|------|------|------|
| Binary Accuracy | 99.67% | > 95% |
| ROC-AUC | 1.0000 | > 0.95 |
| FAR (전체 Spoof) | 0.44% | < 5% |
| FRR (Live) | 0.00% | < 15% |
| Print FAR | 1.33% | < 5% |
| Replay FAR | 0.00% | < 10% |
| Mask FAR | 0.00% | < 8% |
| Replay→Mask 혼동 | 6.7% (5/75) | < 15% |

#### 조명 조건별 FAR

| 조명 | FAR |
|------|-----|
| Indoor-normal | 0.00% |
| Indoor-strong | 0.00% |
| Outdoor-normal | 0.00% |
| Outdoor-strong | 0.00% |
| Outdoor-extreme | 5.88% ⚠️ |

#### 트러블슈팅

| ID | 문제 | 해결 |
|----|------|------|
| TS-05 | test set 전처리 누락 — `/255`만 적용, ImageNet 정규화 없음 → FRR 46.7% | `(img/255 - IMAGENET_MEAN) / IMAGENET_STD` 추가 후 FRR 0.0% |
| TS-06 | Replay→Mask 혼동 5건 — Laplacian 낮고(291 vs 319) FFT 고주파 에너지 높음(3370만 vs 3013만) | 블러 패턴이 원인으로 확인, 수치 앵커링 근거로 활용 |

#### 주요 발견
- Outdoor-extreme 조명 조건에서 FAR 5.88% — 극단적 야외 조명이 시스템 취약점
- Replay→Mask 혼동 케이스는 픽셀 수치(Laplacian/FFT)로 원인 설명 가능 → XAI Layer 2 근거 확보
- MobileNetV2가 ImageNet 정규화에 강하게 의존함을 확인 (전처리 불일치 시 성능 급락)

#### 생성 결과물
- `notebooks/09_far_analysis.ipynb`
- `reports/phase4/roc_curve.png`
- `reports/phase4/far_by_attack.png`
- `reports/phase4/spoof_confusion_matrix.png`
- `reports/phase4/replay_mask_confusion_pixel.png`
- `reports/phase4/replay_mask_sample_grid.png`
- `reports/phase4/far_cross_heatmap.png`


## 🗂️ 디렉토리 구조

```
face-anti-spoofing/
├── data/
│   └── cropped/             # Haar Cascade 크롭 이미지 (live/print/replay/mask)
├── src/
│   ├── gradcam_logit.py     # Logit 기반 Grad-CAM + 수치 앵커링 (Phase 4-A)
│   └── ensemble.py          # 수치 앵커링 전용 모듈 (Phase 4-B 결론)
├── models/
│   ├── stage1_best.h5       # Head 학습 결과
│   └── stage2_best.h5       # Fine-tune 최종 모델
├── reports/
│   └── phase4/
│       ├── 07_gradcam_logit_comparison.png
│       ├── 08_ensemble_comparison.png
│       └── 08_feature_importance.png
├── notebooks/
│   ├── 01_colob_setup.ipynb
│   ├── 03_subset_download.ipynb
│   ├── 04_preprocess_train.ipynb
│   ├── 05_pixel_module.ipynb
│   ├── 06_gradcam.ipynb
│   ├── 07_gradcam_logit.ipynb   # Phase 4-A ✅
│   └── 08_ensemble.ipynb        # Phase 4-B ✅
│   ├── 09_far_analysis.ipynb    # Phase 4-C ✅
└── app/
    └── streamlit_app.py
```

---

## 🚀 진행 현황

| 페이즈 | 내용 | 상태 |
|--------|------|------|
| Phase 1 | 환경 구축 & 데이터 준비 | ✅ 완료 |
| Phase 2 | 전처리 & 멀티태스크 모델 학습 | ✅ 완료 |
| Phase 3 | 픽셀 모듈 & Grad-CAM & 중간발표 | ✅ 완료 |
| Phase 4-A | Logit 기반 Grad-CAM 개선 | ✅ 완료 |
| Phase 4-B | 하이브리드 앙상블 실험 | ✅ 완료 |
| Phase 4-C | 공격 유형별 FAR 분석 | ✅ 완료 |
| Phase 4-D | LLaVA 자연어 캡션 PoC | 🔲 예정 |
| Phase 4-E | 3계층 XAI 통합 + Streamlit 완성 | 🔲 예정 |
| Phase 5 | 최종 발표 준비 | 🔲 예정 |
