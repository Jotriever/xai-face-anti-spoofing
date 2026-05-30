# 🛡️ XAI Face Anti-Spoofing

> **AI 보안 에이전트 — 3계층 XAI 기반 얼굴 위조 공격 탐지 시스템**  
> 멀티태스크 MobileNetV2 + Grad-CAM + 수치 앵커링 + 이미지 기반 자연어 설명

---

## 📌 프로젝트 개요

얼굴 인증 시스템의 핵심 보안 취약점인 **얼굴 위조 공격(Face Anti-Spoofing)** 을 탐지하고,  
AI 모델의 판단 근거를 **3계층 XAI**로 설명하는 시스템입니다.

| 공격 유형 | 설명 | FAR |
|----------|------|-----|
| Print Attack | 종이 인쇄물 | 1.33% |
| Replay Attack | 모니터 재촬영 | 0.00% |
| Mask Attack (3D) | 입체 마스크 | 0.00% |
| **전체** | | **0.44%** |

---

## 🏗️ 시스템 아키텍처

```
입력 이미지 (224×224)
        │
        ▼
┌─────────────────────────────────┐
│  MobileNetV2 Backbone (frozen)  │
│  + Multitask Head               │
│    ├── binary head (REAL/FAKE)  │
│    └── spoof head (4-class)     │
└─────────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
[Layer 1]  [Layer 2]
Grad-CAM   Laplacian + FFT
히트맵      수치 앵커링
   │         │
   └────┬────┘
        ▼
   [Layer 3]
   이미지 기반
   자연어 설명
```

**핵심 알고리즘:**
- **Grad-CAM (logit 기반):** sigmoid 포화 문제 해결, 얼굴 마스크 적용으로 배경 노이즈 제거
- **수치 앵커링:** Laplacian 분산 + FFT 고주파 에너지를 Live 평균 대비 % 로 표시
- **spoof_type 후처리:** ft_model Replay 확률 + Laplacian 기준값(210)으로 Print/Mask 구분
- **마스크 Fine-tuning:** 본인 칸예 마스크 57장 × CelebA mask 100장으로 spoof head 추가 학습

---

## 🤖 AI 도구 활용 전략 (Prompting Log)

이 프로젝트는 **Claude (Anthropic)** 를 AI 코딩 에이전트로 활용하여, 설계부터 배포까지 전 과정을
함께 진행했습니다. 단순 코드 생성이 아니라 **"실행 결과를 다시 입력하고 다음 단계를 지시하는"
반복 루프**로 디렉팅했으며, 그 과정에서 발생한 **22건의 트러블슈팅**을 모두 AI와의 협업으로 해결했습니다.

### 디렉팅 원칙

| 원칙 | 내용 |
|------|------|
| **맥락 우선** | 이전 노트북·코드·실행 로그를 항상 먼저 공유하고 이어서 작업 |
| **결과 기반 반복** | 실행 결과(에러·수치·confusion matrix)를 붙여넣고 다음 단계를 지시 |
| **명확한 제약** | "Colab T4 환경", "외부 라이브러리 없이", "기존 구조 유지" 등 조건을 명시 |
| **점진적 개선** | 작동하는 버전 확보 → 버그 수정 → 보강 순서로 단계화 |
| **단계별 역할** | 설계 / 구현 / 디버깅 / XAI 설계 / 배포로 역할을 나눠 디렉팅 |

### 대표 프롬프팅 사례

실제 작업에서 **프로젝트 방향을 바꾼** 핵심 디렉팅 4건을 정리합니다.
(전체 사례는 [`README_LOG.md`](./README_LOG.md) 참조)

**① 설계 판단 — 실패한 모듈을 버릴 것인가 (TS-04)**
> **상황:** 픽셀 분석(FFT+Laplacian) 단독 판정 정확도가 6,000장 테스트에서 50.3%(랜덤 수준)로 측정.
>
> **프롬프트(요지):** *"FFT+Laplacian 픽셀 분석 단독 정확도가 50.3%로 나왔어. 카테고리별 수치 분포를
> 붙여넣을게. 이 모듈을 폐기해야 할지, 아니면 다른 방식으로 살릴 수 있는지 트레이드오프를 분석해줘."*
>
> **결과:** 판정 도구로는 폐기하되 **XAI 수치 앵커링의 근거**로 역할을 재정의(B안 채택).
> → 실패를 설계 자산으로 전환.

**② 디버깅 루프 — 비정상 FRR 추적 (TS-05)**
> **상황:** 학습은 정상인데 테스트셋 FRR이 46.7%로 비정상 급등.
>
> **프롬프트(요지):** *"train은 정상인데 test에서 FRR 46.7%가 나와. train/test 전처리 코드를 둘 다
> 붙여넣을게. 차이를 찾아줘."*
>
> **결과:** 테스트셋에 ImageNet 정규화가 누락된 것을 발견 → 추가 후 **FRR 46.7% → 0.0%**.

**③ 결과 기반 반복 — 웹캠 도메인 갭 3차 정복 (TS-10 → 11 → 12 → 13)**
> **상황:** CelebA로 학습한 모델이 웹캠으로 찍은 실제 얼굴을 Replay로 오탐(도메인 갭).
>
> **반복 루프:**
> - 1차: *"웹캠 Live가 Replay로 오탐돼. 웹캠 51장 수집했어, fine-tuning 코드 짜줘"* → **v1**
> - 결과 입력: *"v1 후 웹캠 Live는 맞는데 CelebA Live가 75%로 떨어졌어"* → **v2** (LCC-FASD 200장 + 오버샘플링 1:1.4)
> - 결과 입력: *"v2 후 이번엔 공격 탐지율이 떨어졌어"* → threshold 0.65 최적화 + **웹캠 공격 이미지 직접 수집** → **v3**
>
> **결과:** 3차에 걸친 데이터·하이퍼파라미터 재조정으로 도메인 갭 해소. 발표 방식도 웹캠 실시간 →
> 이미지 업로드로 전환해 시연 안정성 확보. *(한 번에 풀리지 않은 문제를 결과 기반으로 3회 반복 개선한 핵심 협업 서사)*

**④ XAI 설계 — 확신할수록 설명이 안 되는 역설 (TS-05/Grad-CAM)**
> **상황:** FAKE 확률 100% 케이스에서 Grad-CAM 히트맵이 전혀 생성되지 않음(sigmoid 포화).
>
> **프롬프트(요지):** *"Grad-CAM이 FAKE 100% 케이스에서 히트맵을 못 만들어. sigmoid 출력 기반인데
> 포화 때문인 것 같아. 어떻게 해결하지?"*
>
> **결과:** sigmoid 직전 **logit 기반 Grad-CAM**으로 전환 → 모든 케이스에서 히트맵 생성.

### 정량적 협업 성과

| 항목 | 수치 |
|------|------|
| 트러블슈팅 해결 | **22건** (TS-01 ~ TS-22) |
| 완주 페이즈 | **Phase 1 ~ 6** |
| 작성 노트북 | **11종** + 데모/검증 노트북 |
| 최종 산출물 | 멀티태스크 모델 · 3계층 XAI 파이프라인 · Streamlit 앱 |

---

## 🚀 실행 방법 (How to Run)

### 로컬 실행
```bash
# 1. 레포지토리 클론
git clone https://github.com/Jotriever/xai-face-anti-spoofing.git
cd xai-face-anti-spoofing

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 모델 파일 배치 (Google Drive에서 다운로드)
# models/stage2_webcam_v3.h5
# models/stage2_mask_ft.h5
# 다운로드: https://drive.google.com/drive/folders/1dElsPLNOkIAlmb-hvHh2cuQ-ITJbnIVt

# 4. 앱 실행
streamlit run app.py
```

### Colab 환경 (학습/실험)

노트북은 Phase 순서대로 실행하도록 구성되어 있습니다.

```
notebooks/
# ── Phase 1 · 환경 구축 & 데이터 ─────────────────────────
├── 01_colob_setup.ipynb         # Colab 환경 설정 + 디렉토리 구조
├── 03_subset_download.ipynb     # CelebA-Spoof 서브셋 다운로드 (6,000장)
# ── Phase 2 · 전처리 & 학습 ──────────────────────────────
├── 04_preprocess_train.ipynb    # Haar Cascade 전처리 + 멀티태스크 학습
# ── Phase 3 · 픽셀 모듈 & Grad-CAM (중간 발표) ───────────
├── 05_pixel_module.ipynb        # FFT + Laplacian 픽셀 분석
├── 06_gradcam.ipynb             # Grad-CAM 시각화 (1차)
# ── Phase 4 · XAI 고도화 ─────────────────────────────────
├── 05b_hybrid_retrain.ipynb     # 하이브리드(CNN+픽셀) 재학습 실험
├── 07_gradcam_logit.ipynb       # logit 기반 Grad-CAM (sigmoid 포화 해결)
├── 08_ensemble.ipynb            # 앙상블 실험
├── 09_far_analysis.ipynb        # 공격 유형별 FAR 분석
├── 10_llava_caption.ipynb       # LLaVA 자연어 캡션 PoC
├── 11_xai_integration.ipynb     # 3계층 XAI 통합 (xai_explainer.py 생성)
# ── Phase 5 · 웹캠 Fine-tuning & 앱 고도화 ───────────────
│   # (12~19) 웹캠 도메인 갭 Fine-tuning v1~v3 + Streamlit v2
│  
# ── Phase 6 · 최종 발표 데모 ─────────────────────────────
├── 20_image_upload_demo.ipynb   # 이미지 업로드 발표 데모 앱
└── 21_my_face_validation.ipynb  # 본인 4종 검증 + 마스크 Fine-tuning
```

---

## 📁 프로젝트 구조

```
xai-face-anti-spoofing/
├── app.py                    # Streamlit 메인 앱
├── requirements.txt
├── README.md
├── README_LOG.md             # 상세 작업 로그 (Phase별 활동 + 트러블슈팅 22건)
├── src/
│   └── xai_explainer.py      # 3계층 XAI 파이프라인
├── models/
│   ├── stage2_webcam_v3.h5   # 메인 모델 (웹캠 fine-tuned)
│   └── stage2_mask_ft.h5     # 마스크 fine-tuned 모델
└── data/
    └── demo_images/
        ├── demo_meta.json
        ├── 01_webcam_live.jpg
        ├── 02_print_attack.jpg
        ├── 03_replay_attack.jpg
        └── 04_mask_attack.jpg
```

---

## 📊 성능 요약

| 공격 유형 | 테스트 수 | 정확도 | FAR | 목표 | 결과 |
|----------|----------|--------|-----|------|------|
| Print Attack | 75 | 98.67% | 1.33% | < 5% | ✅ |
| Replay Attack | 75 | 100% | 0.00% | < 10% | ✅ |
| 3D Mask Attack | 75 | 100% | 0.00% | < 8% | ✅ |
| Live (FRR) | 75 | 100% | 0.00% | < 15% | ✅ |
| **전체** | **300** | **96%** | **0.44%** | — | ✅ |

> 모델 크기: **13.3 MB** (MobileNetV2 기반, 엣지 배포 적합)

---

## 🔮 한계점 및 향후 과제

**기술적 한계**
- **Grad-CAM 히트맵:** backbone frozen으로 마스크 경계면 집중 미흡 → backbone fine-tuning 필요
- **도메인 갭:** 웹캠 후처리(샤프닝/압축)로 인한 오탐 → 도메인 적응 기법 적용 필요
- **spoof_type 구분:** Print/Mask 수치 유사성으로 혼동 → 더 다양한 공격 유형 데이터 필요

**보안 관점 한계 및 인사이트**
- **피험자 검증 한계:** 마스크 Fine-tuning과 4종 데모가 모두 10인 미만의 얼굴 기반 →
  실제 배포에는 다인종·다연령 데이터 확보가 필수
- **Unseen attack 일반화:** 알려진 4종 공격만 방어 → 딥페이크 영상 등 신종 공격에는 미대응
- **XAI의 운영 가치:** 설명이 보안 담당자의 실제 의사결정을 얼마나 보조하는지는 별도 사용자 평가 필요
- **인사이트:** "탐지"와 "설명"을 분리해 설계한 결과, 픽셀 분석처럼 단독 성능이 낮은 모듈도
  설명 근거로 재활용할 수 있었음 → XAI 시스템에서 "실패한 신호"의 재해석 가능성 확인
