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

---

## 📅 작업 단위 활동 로그

*(Phase 1~5-B 로그는 기존 README_LOG.md 참조)*

---

### [2026-05-28~29] Phase 6: 이미지 업로드 기반 최종 발표 데모 완성

#### 전략 전환: 웹캠 실시간 → 이미지 업로드 방식

**배경:** 웹캠 실시간 입력은 카메라 후처리(샤프닝/압축)로 인한 Laplacian/FFT 수치 왜곡 → 발표 환경 오탐 리스크 존재

**전환 이유:**
- 발표의 목적은 "웹캠 실시간성"이 아닌 "XAI 시스템 검증"
- 이미지 업로드 방식: 미리 선별한 케이스로 안정적 시연 보장
- 발표 당일 변수(네트워크/조명/ngrok) 제거

#### 본인 4종 시연 이미지 구성 (전부 본인 얼굴)

| 번호 | 이미지 | 촬영 방법 | 기대 판정 |
|------|-------|---------|---------|
| ① | Live | 본인 얼굴 정면 촬영 | REAL ✅ |
| ② | Print Attack | 본인 사진 프린트 → 들고 촬영 | FAKE 🚨 |
| ③ | Replay Attack | 모니터에 본인 사진 띄우고 촬영 | FAKE 🚨 |
| ④ | Mask Attack | **칸예 웨스트 마스크 착용** → 촬영 | FAKE 🚨 |

#### 버그 수정 3종

| 버그 | 원인 | 수정 |
|------|------|------|
| REAL인데 spoof_type=Replay | verdict와 spoof_type head가 독립 | REAL 판정 시 'Live (실제 얼굴)'로 강제 |
| Layer 3 캡션 전부 없음 | img_path=None → JSON 조회 불가 | 이미지 특성 기반 fallback 캡션 생성 |
| Grad-CAM 레이어 이름 오류 | 텐서 이름으로 잡혀 히트맵 엉뚱한 곳 | Conv_1_bn / binary 하드코딩으로 수정 |

#### spoof_type 후처리 보정 로직 (22번 노트북)

**문제:** 원본 모델이 Print/Mask를 혼동 (본인 이미지 기준)

**해결:** ft_model Replay 확률 + Laplacian 기준값으로 3-way 분류
```
if ft_probs[Replay] >= 0.40  → Replay
elif Laplacian < 210         → Print
else                         → 3D Mask
```

#### 마스크 Fine-tuning (stage2_mask_ft.h5)

**목적:** Mask Attack Grad-CAM 히트맵 경계면 개선  
**데이터:** 본인 칸예 마스크 57장 × CelebA mask 100장 × CelebA live 100장  
**결과:** val_spoof_accuracy 96.4% (6 epoch)  
**한계:** spoof head만 학습(backbone frozen)으로 Grad-CAM 자체는 변화 없음  
→ **"FAKE 판정은 97.9%로 정확, 히트맵 개선은 향후 backbone fine-tuning 과제"** 로 발표

#### 이미지 기반 Layer 3 캡션 (build_image_caption)

LLaVA 없이 이미지 특성으로 캡션 자동 생성:
- Grad-CAM 히트맵 활성 위치 감지 (9×9 그리드 → upper/mid/lower × left/center/right)
- Laplacian/FFT를 Live 평균 대비 % 로 표현
- spoof_type별 도메인 특화 문장 조합

#### 최종 앱 (streamlit_demo_v4.py)

| 기능 | 내용 |
|------|------|
| 기본 화면 | 업로드 창이 메인에 바로 표시 |
| 업로드 플로우 | ① 원본 → ② 얼굴 크롭(224×224) → ③ XAI 분석 3단계 시각화 |
| 수치 표시 | Laplacian/FFT + Live 평균 대비 % + delta 게이지 |
| 4종 시연 | 사이드바 라디오 버튼으로 선택 |
| FAR 대시보드 | 사이드바 고정 표시 |

#### 최종 검증 결과 (본인 4종)

| 카테고리 | 판정 | spoof_type | Laplacian |
|---------|------|-----------|---------|
| Live | REAL ✅ | Live (실제 얼굴) | 266 |
| Print | FAKE ✅ | Print Attack | 196 |
| Replay | FAKE ✅ | Replay Attack | 146 |
| Mask | FAKE ✅ | 3D Mask | 737 |

#### 트러블슈팅

| ID | 문제 | 해결 |
|----|------|------|
| TS-17 | MTCNN Python 3.12 + joblib lz4 충돌 | OpenCV Haar Cascade로 대체 |
| TS-18 | facenet-pytorch PIL._util 충돌 | Haar Cascade 확정 |
| TS-19 | xai_explainer.py 없음 | src/ 폴더 미생성 → 코드로 직접 생성 |
| TS-20 | ft_model spoof_type 전부 Mask로 분류 | oversampling ×3 → ×1로 재학습 |
| TS-21 | 재학습 후에도 Print가 Mask 98% | Lap+ft_model 조합 후처리로 해결 |
| TS-22 | Grad-CAM Before/After 동일 | backbone frozen이라 원리상 불변 — 한계점 처리 |

#### 생성 결과물
- `notebooks/20_image_upload_demo.ipynb` — 최종 발표 데모 앱 실행
- `notebooks/21_my_face_validation.ipynb` — 본인 4종 검증 + 마스크 fine-tuning
- `notebooks/22_explainer_v2.ipynb` — spoof_type 후처리 + 이미지 기반 캡션
- `app/streamlit_demo_v4.py` — 최종 Streamlit 앱
- `src/xai_explainer.py` — Grad-CAM 레이어 하드코딩 수정 (Conv_1_bn / binary)
- `models/stage2_mask_ft.h5` — 마스크 fine-tuned 모델 (13.3MB)
- `data/my_mask_images/` — 본인 칸예 마스크 촬영 57장
- `data/demo_images/` — 4종 시연 이미지 + demo_meta.json

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
| Phase 4-D | LLaVA 자연어 캡션 PoC | ✅ 완료 |
| Phase 4-E | 3계층 XAI 통합 + xai_explainer.py | ✅ 완료 |
| Phase 5 | Streamlit v2 + 웹캠 Fine-tuning | ✅ 완료 |
| Phase 5-B | 웹캠 도메인 갭 심화 해결 + Fine-tuning v3 | ✅ 완료 |
| **Phase 6** | **이미지 업로드 데모 + 마스크 Fine-tuning + Streamlit Cloud 배포** | **✅ 완료** |

---

## 🐛 전체 트러블슈팅 목록

| ID | 문제 | 해결 |
|----|------|------|
| TS-01 | MTCNN RAM 초과 | Haar Cascade 전환 |
| TS-02 | 데이터 불균형 (1:3) | class_weight 적용 |
| TS-03 | Laplacian/FFT 스케일 차이 | 정규화 후 concat |
| TS-04 | 픽셀 단독 판정 실패 (50.3%) | 역할 재정의 → XAI 수치 앵커링 전용 |
| TS-05 | test set 전처리 누락 — FRR 46.7% | ImageNet 정규화 추가 후 FRR 0.0% |
| TS-06 | Replay→Mask 혼동 5건 | 수치 앵커링 근거로 활용 |
| TS-07 | LlavaNextProcessor KeyError | AutoProcessor로 교체 |
| TS-08 | apply_chat_template() v1.5 비호환 | USER/ASSISTANT 포맷 직접 사용 |
| TS-09 | 한글 폰트 깨짐 | fonts-nanum 설치 |
| TS-10 | 웹캠 Live → Replay 오탐 (도메인 갭 v1) | 웹캠 51장 수집 → Fine-tuning v1 |
| TS-11 | v1 fine-tuning 후 CelebA Live 75%로 하락 | LCC FASD 200장 + 오버샘플링 1:1.4 → v2 |
| TS-12 | v2 fine-tuning 후 공격 탐지율 하락 | threshold 0.65 최적화 |
| TS-13 | 웹캠 환경 Replay/Print 오탐 (73%) | 웹캠 공격 이미지 직접 수집 → Fine-tuning v3 |
| TS-14 | class_weight 멀티출력 모델 미지원 | sample_weight → 오버샘플링으로 대체 |
| TS-15 | MobileNetV2 sub-model backbone unfreeze 실패 | 레이어 직접 탐색으로 해결 |
| TS-16 | Grad-CAM 배경/가장자리 활성화 | 얼굴 마스크 후처리 적용 (부분 해결) |
| TS-17 | MTCNN Python 3.12 + joblib lz4 충돌 | OpenCV Haar Cascade로 대체 |
| TS-18 | facenet-pytorch PIL._util 충돌 | Haar Cascade 확정 |
| TS-19 | xai_explainer.py 없음 (src/ 폴더 미생성) | 코드로 직접 생성 |
| TS-20 | ft_model spoof_type 전부 Mask로 분류 | oversampling ×3 → ×1 + print/replay 균형 데이터 추가 재학습 |
| TS-21 | 재학습 후에도 Print가 Mask 98% | ft_model Replay 확률 + Laplacian 210 기준 후처리 조합 |
| TS-22 | Grad-CAM Before/After fine-tuning 동일 | backbone frozen 원리상 불변 → 한계점으로 발표 처리 |