# 🛡️ Face Anti-Spoofing Project
> **멀티태스크 학습을 활용한 얼굴 위조 판별 및 공격 유형 분류 시스템**

본 프로젝트는 보안 시스템의 취약점인 '얼굴 위조 공격(Print, Replay)'을 탐지하기 위해 **CelebA-Spoof 데이터셋**을 분석하고, 주파수(FFT) 및 선명도(Laplacian) 수치를 기반으로 한 하이브리드 탐지 모델을 구축하는 것을 목표로 합니다.

## 🛠️ 개발 환경
- **Environment:** Google Colab
- **Language:** Python 3.x
- **Libraries:** OpenCV, NumPy, Matplotlib, Kaggle API, PyTorch
- **Dataset:** [CelebA-Spoofing (Kaggle)](https://www.kaggle.com/datasets/mabdullahsajid/celeba-spoofing)

---

## 📅 작업 단위 활동 로그 (4/29 ~ 5/7)

### [2026.04.29] 교수님 피드백 및 전략 수정

#### 1. 주요 피드백 (Professor's Feedback)
- **"Deep Learning의 한계 지적":** 단순 Binary Classification 모델은 판단 근거가 불분명한 'Black-box' 문제가 있음.
- 현재의 CelebA-spoof는 설명 가능한 AI(XAI)를 제작하기에는 다소 무리가 있음


#### 2. 수정된 연구 방향 (Adjusted Research Plan)
교수님의 피드백을 바탕으로, 단순 모델 학습에서 '물리적 수치 분석 기반의 하이브리드 탐지'로 프로젝트 방향을 고도화함.

- **Hybrid Detection Logic:** - 딥러닝 모델(MobileNet/ResNet 등)과 수치 분석 알고리즘(FFT, Laplacian)을 결합.
    - 모델이 놓칠 수 있는 미세한 위조 패턴을 주파수 도메인에서 이중 검증.
- **Smart Sampling Strategy:**
    - 전체 62만 장 데이터 중, 조명과 환경 변수가 통제된 `intra_test` 프로토콜 기반의 핵심 샘플 6만 장을 선별.
    - 데이터 양보다 **데이터 질(Quality)**과 **설명 가능한 보안(Explainable AI)**에 집중.



### [2026-05-07] 데이터 가용성 이슈 해결 및 메타데이터 파싱
- **Issue:** 대용량 데이터셋(62만 장, 약 200GB)의 코랩 환경 내 전체 다운로드 불가 및 외부 링크 404 에러 발생.
- **Solution:** 1. **Metadata-First 전략:** 전체 데이터를 받는 대신, `intra_test` 프로토콜의 `train_label.txt` 메타데이터를 우선적으로 확보.
    2. **Selective Extraction:** 확보된 지도를 바탕으로 분석에 필요한 핵심 샘플(진짜 얼굴, 인쇄 공격, 화면 공격)만 선별적으로 다운로드하는 스마트 샘플링 파이프라인 구축.
- **Result:** 분석용 데이터 `data/subset/` 구축 완료 및 수치 분석 준비 완료.

---

## 🔬 핵심 분석 논리 (Work in Progress)

위조 공격의 물리적 특성을 포착하기 위해 다음 두 가지 수치 앵커를 사용:

1. **FFT (Fast Fourier Transform):** - **진짜 얼굴:** 자연스러운 저주파 성분이 강함.
    - **위조 공격:** 인쇄물의 망점이나 디스플레이의 격자 무늬로 인해 특정 고주파 영역에서 비정상적인 에너지 검출.
2. **Laplacian Variance:**
    - 재촬영 및 인쇄 과정에서 발생하는 경계면의 뭉개짐(Blurring)을 선명도 수치로 정량화.

---

### [2026-05-11] 공격 유형별 수치 탐색 및 샘플 검증

#### 1. Spoof Type 매핑 확정
- `train_label.json` 파싱을 통해 전체 11개 타입(0~10) 분포 확인
- 샘플 이미지 육안 검증으로 실제 타입 매핑 확정:

| Type | 공격 유형 |
|---|---|
| 0 | Live |
| 1, 2, 6, 8 | Print (인쇄 사진) |
| 3, 5, 7 | Replay (화면 재촬영) |
| 4, 9 | Paper Cut (오려낸 마스크) |
| 10 | 3D Mask (Toy) |

#### 2. 카테고리별 샘플 구축
- Kaggle API `-f` 옵션 기반 선택적 다운로드 파이프라인 완성
- `live` / `print` / `replay` / `toy` 각 5장씩 총 20장 수집
- 저장 경로: `./samples/{category}/`

#### 3. FFT + Laplacian 수치 탐색 결과

| 카테고리 | FFT 고주파 에너지 | Laplacian 분산 |
|---|---|---|
| Live | 645 | 350 |
| Print | 800 | 395 |
| **Replay** | **475** | **135** |
| Toy | 1490 | 1420 |

- **핵심 발견:** Replay Attack이 FFT·Laplacian 두 지표 모두에서 최저값 → 화면 재촬영 특유의 블러 패턴이 수치로 확인됨
- **한계:** 3D Mask 샘플은 마스크를 손에 들고 촬영한 이미지 다수 포함 → 배경/엣지 노이즈로 수치 과대 측정. BB.txt 기반 얼굴 크롭 적용 시 개선 예상.
---

### [2026-05-12] Google Drive 연동 및 서브셋 다운로드 파이프라인 구축

#### 1. Google Drive 영구 연동
- 기존 문제: 데이터를 `/content/`에 저장하여 런타임 재시작 시 초기화되는 문제 발생
- 해결: `BASE = '/content/drive/MyDrive/face-anti-spoofing'` 경로 고정으로 데이터 영구 유지
- `kaggle.json` Drive에 저장 → 이후 런타임 재시작 시 재업로드 불필요

#### 2. 서브셋 다운로드 파이프라인 완성 (`03_subset_download.ipynb`)
- `train_label.json` 기반 Spoof Type별 경로 추출 자동화
- 카테고리별 균형 샘플링 (random.seed(42) 고정)
- 런타임 끊겨도 이어받기 가능 (이미 받은 파일 자동 스킵)

#### 3. 중간발표용 서브셋 구축 완료
- 전략: 6만장 전체 대신 중간발표용 6,000장 우선 확보 → 최종발표 전 확장
- 카테고리별 1,500장 × 4 = 총 6,000장

| 카테고리 | 장수 | Spoof Type |
|----------|------|------------|
| live     | 1,500 | Type 0 |
| print    | 1,500 | Type 1, 2, 6, 8 |
| replay   | 1,500 | Type 3, 5, 7 |
| mask     | 1,500 | Type 4, 9, 10 |

## 🚀 향후 계획
- [x] 공격 유형별 수치 임계값 탐색 완료 (예비 분석)
- [ ] BB.txt 바운딩박스 기반 얼굴 크롭 후 수치 재측정
- [ ] 6만 장 전체 서브셋 다운로드 및 멀티태스크 MobileNetV2 학습 시작