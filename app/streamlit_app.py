"""
app/streamlit_app.py  —  Phase 5 v2
페이스페이 위조 공격 방어 시스템 (3계층 XAI)

실행:
    streamlit run app/streamlit_app.py
"""

import sys, os, json
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE, "src")
sys.path.insert(0, SRC_DIR)

# ── xai_explainer import ──────────────────────────────────
try:
    from xai_explainer import explain, load_model_once
    XAI_READY = True
except ImportError:
    XAI_READY = False

# ── FAR 데이터 ────────────────────────────────────────────
FAR_DATA = {
    "Print":  {"far": None, "target": 5.0},
    "Replay": {"far": None, "target": 10.0},
    "Mask":   {"far": None, "target": 8.0},
}

def load_far_results():
    far_path = os.path.join(BASE, "results", "phase4", "far_analysis.json")
    try:
        with open(far_path) as f:
            data = json.load(f)
        for cat in FAR_DATA:
            key = cat.lower() + "_far"
            if key in data:
                FAR_DATA[cat]["far"] = round(data[key] * 100, 2)
    except FileNotFoundError:
        pass

load_far_results()

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="FAS — 위조 공격 방어 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ FAS v2")
    st.caption("단국대학교 AI Security & Application")
    st.markdown("---")

    st.subheader("⚙️ 모델 설정")
    threshold = st.slider(
        "판정 임계값 (Spoof Prob > X → FAKE)",
        min_value=0.3, max_value=0.9, value=0.5, step=0.05
    )
    show_raw = st.checkbox("수치 상세 표시", value=True)

    st.markdown("---")
    st.subheader("📊 공격 유형별 FAR")
    for cat, info in FAR_DATA.items():
        far_val = info["far"]
        target  = info["target"]
        if far_val is not None:
            status = "✅" if far_val <= target else "⚠️"
            st.metric(
                label=status + " " + cat,
                value=str(far_val) + "%",
                delta="목표 < " + str(target) + "%",
                delta_color="inverse"
            )
        else:
            st.metric(
                label="⏳ " + cat,
                value="N/A",
                delta="목표 < " + str(target) + "%"
            )

    st.markdown("---")
    st.caption("Phase 4 FAR 분석 결과 자동 반영")
    st.markdown("---")
    st.subheader("🗂️ 데이터 수집")
    collect_mode = st.toggle("수집 모드 (Live 이미지 저장)", value=False)
    webcam_dir = os.path.join(BASE, "data", "webcam_live")
    os.makedirs(webcam_dir, exist_ok=True)
    saved_count = len(list(Path(webcam_dir).glob("*.jpg")))
    st.caption("저장된 이미지: " + str(saved_count) + "장 / 목표 50장")
    st.progress(min(saved_count / 50, 1.0))


# ── 메인 ──────────────────────────────────────────────────
st.title("🛡️ 페이스페이 위조 공격 방어 시스템")
st.caption("MobileNetV2 멀티태스크 + 3계층 XAI (Grad-CAM · 수치 앵커링 · 자연어 설명)")

if not XAI_READY:
    st.error("⚠️ src/xai_explainer.py 를 찾을 수 없습니다. Phase 4-E 완료 후 재시작하세요.")
    st.stop()

# 모델 1회 로드
@st.cache_resource
def init_model():
    load_model_once()
    return True

init_model()

# ── 이미지 입력 (웹캠) ────────────────────────────────────
st.markdown("### 📷 얼굴 촬영")
st.caption("카메라로 얼굴을 촬영하거나 데모 모드를 사용하세요.")

input_mode = st.radio(
    "입력 방식",
    ["📸 웹캠 촬영", "🗂️ 데모 모드"],
    horizontal=True
)

img_bgr = None

if input_mode == "📸 웹캠 촬영":
    camera_photo = st.camera_input("촬영 버튼을 눌러 얼굴을 찍으세요")
    if camera_photo is not None:
        file_bytes = np.frombuffer(camera_photo.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

else:
    demo_cats = ["live", "print", "replay", "mask"]
    demo_cat  = st.selectbox("샘플 카테고리 선택", demo_cats)
    demo_dir  = os.path.join(BASE, "data", "cropped", demo_cat)
    demo_imgs = sorted(Path(demo_dir).glob("*.jpg")) if os.path.isdir(demo_dir) else []
    if demo_imgs:
        img_bgr = cv2.imread(str(demo_imgs[0]))
        st.info("샘플 이미지: " + demo_cat)
    else:
        st.warning("샘플 이미지 없음 — 데이터 경로를 확인하세요.")

# ── 분석 실행 ─────────────────────────────────────────────

    # 수집 모드: 촬영 이미지를 webcam_live 폴더에 저장
    if collect_mode and img_bgr is not None and input_mode == "📸 웹캠 촬영":
        import time
        save_path = os.path.join(BASE, "data", "webcam_live",
                                 "webcam_" + str(int(time.time())) + ".jpg")
        cv2.imwrite(save_path, img_bgr)
        st.sidebar.success("💾 저장: " + os.path.basename(save_path))
if img_bgr is not None:
    with st.spinner("🔍 분석 중..."):
        try:
            is_webcam = (input_mode == "📸 웹캠 촬영")
            result = explain(img_bgr, threshold=threshold, webcam=is_webcam)
        except Exception as e:
            st.error("분석 오류: " + str(e))
            st.stop()

    verdict    = result["verdict"]
    spoof_prob = result["spoof_prob"]
    spoof_type = result["spoof_type_name"]

    if verdict == "FAKE":
        st.error("🚨 FAKE — 위조 공격 탐지됨 (신뢰도 " + "{:.1%}".format(spoof_prob) + ") | 유형: " + spoof_type)
    else:
        st.success("✅ REAL — 실제 얼굴로 판정됨 (신뢰도 " + "{:.1%}".format(1 - spoof_prob) + ")")

    col1, col2, col3 = st.columns(3)

    # Layer 1: Grad-CAM
    with col1:
        st.markdown("#### 🔥 Layer 1: Grad-CAM")
        st.caption("모델이 주목한 얼굴 영역")
        st.image(
            result["heatmap_overlay"],
            use_column_width=True,
            caption="Spoof Prob: " + "{:.1%}".format(spoof_prob) + " | Type: " + spoof_type
        )
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_column_width=True, caption="원본 이미지")

    # Layer 2: 수치 앵커링
    with col2:
        st.markdown("#### 📐 Layer 2: 수치 앵커링")
        st.caption("Grad-CAM 활성 영역의 FFT / Laplacian 수치")

        anchor  = result["anchor_stats"]
        lap_val = anchor.get("laplacian", "N/A")
        fft_val = anchor.get("fft_high",  "N/A")

        REF = {
            "live":   {"lap": 383, "fft": 1134},
            "print":  {"lap": 318, "fft": 1042},
            "replay": {"lap": 319, "fft": 944},
            "mask":   {"lap": 480, "fft": 1134},
        }

        m1, m2 = st.columns(2)
        m1.metric("Laplacian",
                  "{:.0f}".format(lap_val) if isinstance(lap_val, float) else str(lap_val),
                  help="활성화 영역 선명도")
        m2.metric("FFT High-Freq",
                  "{:.0f}".format(fft_val) if isinstance(fft_val, float) else str(fft_val),
                  help="활성화 영역 고주파 에너지")

        st.markdown("**📋 참조 기준값 (전체 평균)**")
        rows = ["| 유형 | Laplacian | FFT |", "|---|---|---|"]
        for cat, vals in REF.items():
            rows.append("| " + cat.capitalize() + " | " + str(vals["lap"]) + " | " + str(vals["fft"]) + " |")
        st.markdown("\n".join(rows))

        if show_raw:
            with st.expander("수치 상세"):
                st.json(anchor)

        if isinstance(lap_val, (int, float)):
            fig_bar, ax = plt.subplots(figsize=(4, 2.5))
            bar_cats   = list(REF.keys()) + ["입력 이미지"]
            bar_vals   = [REF[c]["lap"] for c in REF] + [lap_val]
            bar_colors = ["#95A5A6"] * 4 + ["#E74C3C" if verdict == "FAKE" else "#27AE60"]
            ax.barh(bar_cats, bar_vals, color=bar_colors)
            ax.set_xlabel("Laplacian")
            ax.set_title("Laplacian 비교", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

    # Layer 3: 자연어 설명
    with col3:
        st.markdown("#### 💬 Layer 3: 자연어 설명")
        st.caption("XAI 통합 설명 + LLaVA 캡션")

        xai_text = result.get("xai_text", "설명 생성 실패")
        if verdict == "FAKE":
            st.warning("🔍 XAI 판정 근거\n\n" + xai_text)
        else:
            st.info("🔍 XAI 판정 근거\n\n" + xai_text)

        llava_caption = result.get("llava_caption", "")
        st.markdown("**🤖 LLaVA 시각적 분석:**")
        if llava_caption and llava_caption != "(none)":
            st.success(llava_caption)
        else:
            st.caption("(LLaVA 캡션 DB에 해당 이미지 없음)")

        st.markdown("**📌 공격 유형 해설:**")
        TYPE_DESC = {
            "Live (실제 얼굴)":           "✅ 실제 얼굴. 위조 패턴 없음.",
            "Print Attack (인쇄 공격)":   "🖨️ 인쇄 공격. 평면적 반사광 패턴 검출.",
            "Replay Attack (화면 재촬영)": "📺 재촬영 공격. 모아레 패턴 검출.",
            "3D Mask (입체 마스크)":       "😷 3D 마스크. 경계선 텍스처 불연속 검출.",
        }
        st.markdown(TYPE_DESC.get(spoof_type, "❓ 유형 불명."))

    # FAR 대시보드
    st.markdown("---")
    st.markdown("### 📊 공격 유형별 FAR 대시보드")
    st.caption("Phase 4 FAR 분석 결과 | 목표치 대비 달성 여부")

    has_far = any(v["far"] is not None for v in FAR_DATA.values())
    if has_far:
        fig_far, ax_far = plt.subplots(figsize=(8, 3))
        f_cats    = list(FAR_DATA.keys())
        f_actuals = [FAR_DATA[c]["far"] or 0 for c in f_cats]
        f_targets = [FAR_DATA[c]["target"] for c in f_cats]
        f_colors  = [
            "#27AE60" if (FAR_DATA[c]["far"] or 999) <= FAR_DATA[c]["target"]
            else "#E74C3C"
            for c in f_cats
        ]
        x    = np.arange(len(f_cats))
        bars = ax_far.bar(x - 0.2, f_actuals, 0.35, label="실측 FAR", color=f_colors, alpha=0.85)
        ax_far.bar(x + 0.2, f_targets, 0.35, label="목표 FAR", color="#BDC3C7", alpha=0.7)
        ax_far.set_xticks(x)
        ax_far.set_xticklabels(f_cats, fontsize=11)
        ax_far.set_ylabel("FAR (%)")
        ax_far.set_title("공격 유형별 FAR: 실측 vs 목표", fontsize=12)
        ax_far.legend()
        for bar, val in zip(bars, f_actuals):
            ax_far.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                "{:.1f}%".format(val),
                ha="center", fontsize=9
            )
        plt.tight_layout()
        st.pyplot(fig_far)
        plt.close(fig_far)
    else:
        st.info("FAR 분석 결과 파일(results/phase4/far_analysis.json)이 없습니다. Phase 4-C 완료 후 자동으로 표시됩니다.")

else:
    st.markdown("---")
    st.info(
        "📷 카메라로 얼굴을 촬영하거나 데모 모드를 선택하세요.\n\n"
        "지원 공격 유형: Print · Replay · 3D Mask · Live\n\n"
        "3계층 XAI:\n"
        "- Layer 1: logit 기반 Grad-CAM (어디를 봤는가)\n"
        "- Layer 2: 활성 영역 수치 앵커링 (얼마나 강한 신호인가)\n"
        "- Layer 3: LLaVA 자연어 설명 (왜 그렇게 판단했는가)"
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Binary Accuracy", "96%",    "목표 >95% ✅")
    col_b.metric("Spoof Type Acc",  "80%",    "목표 >80% ✅")
    col_c.metric("데이터셋",         "6,000장", "CelebA-Spoof")
