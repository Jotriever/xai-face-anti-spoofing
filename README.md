# 🛡️ XAI Face Anti-Spoofing

> **AI 보안 에이전트 — 3계층 XAI 기반 얼굴 위조 공격 탐지 시스템**  
> 멀티태스크 MobileNetV2 + Grad-CAM + 수치 앵커링 + 이미지 기반 자연어 설명

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📌 프로젝트 개요

얼굴 인증 시스템의 핵심 보안 취약점인 **얼굴 위조 공격(Face Anti-Spoofing)** 을 탐지하고,  
AI 모델의 판단 근거를 3계층 XAI로 설명하는 시스템입니다.

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

이 프로젝트는 **Claude (Anthropic)** 를 AI 코딩 에이전트로 활용하여 전 과정을 진행했습니다.

### 주요 활용 전략

| 단계 | 활용 방식 |
|------|---------|
| 설계 | 아키텍처 논의 → 트레이드오프 분석 → 방향 결정 |
| 구현 | Jupyter Notebook 단위로 셀 단위 생성 요청 |
| 디버깅 | 오류 메시지 붙여넣기 → 원인 분석 → 수정 코드 |
| XAI 설계 | 3계층 구조 아이디어 → 구체적 구현 방법 도출 |
| 배포 | Streamlit Cloud 배포 파일 세트 자동 생성 |

### 프롬프팅 원칙
1. **맥락 우선:** 이전 결과물을 항상 공유하고 이어서 작업
2. **결과 기반 반복:** 실행 결과를 붙여넣고 다음 단계 요청
3. **명확한 제약:** "외부 라이브러리 없이", "Colab 환경에서" 등 조건 명시
4. **점진적 개선:** 작동하는 버전 → 버그 수정 → 보강 순서

---

## 🚀 실행 방법 (How to Run)

### Streamlit Cloud (권장)
앱 URL: `https://your-app-url.streamlit.app`

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
```
notebooks/
├── 01_colob_setup.ipynb         # 환경 설정
├── 04_preprocess_train.ipynb    # 데이터 전처리 + 학습
├── 06_gradcam.ipynb             # Grad-CAM 실험
├── 08_ensemble.ipynb            # 앙상블
├── 11_xai_integration.ipynb    # XAI 통합
├── 20_image_upload_demo.ipynb  # 발표 데모 앱
├── 21_my_face_validation.ipynb # 본인 4종 검증 + 마스크 fine-tuning
└── 22_explainer_v2.ipynb       # spoof_type 후처리 보강
```

---

## 📁 프로젝트 구조

```
xai-face-anti-spoofing/
├── app.py                    # Streamlit 메인 앱
├── requirements.txt
├── README.md
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

| 지표 | 값 |
|------|-----|
| 전체 정확도 | 96% |
| FAR (False Acceptance Rate) | 0.44% |
| Live → REAL 정확도 | 100% (웹캠 도메인) |
| FAKE → FAKE 정확도 | 96% |
| 모델 크기 | 13.3 MB |

---

## 🔮 한계점 및 향후 과제

- **Grad-CAM 히트맵:** backbone frozen으로 마스크 경계면 집중 미흡 → backbone fine-tuning 필요
- **도메인 갭:** 웹캠 후처리(샤프닝/압축)로 인한 오탐 → 도메인 적응 기법 적용 필요
- **spoof_type 구분:** Print/Mask 수치 유사성으로 혼동 → 더 다양한 공격 유형 데이터 필요
