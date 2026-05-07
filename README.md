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

## 🚀 향후 계획
- [ ] 추출된 샘플 데이터 기반 FFT 주파수 분포 시각화
- [ ] 공격 유형별(index 40) 수치 임계값(Threshold) 도출
- [ ] 딥러닝 백본 모델과 수치 분석을 결합한 하이브리드 분류기 설계