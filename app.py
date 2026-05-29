"""
app.py — Face Anti-Spoofing XAI Demo
Streamlit Cloud 배포용 메인 앱
"""

import os, sys, json, gdown
import numpy as np
import cv2
import streamlit as st
from pathlib import Path
import tensorflow as tf

# ── 경로 ─────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
SRC_DIR       = os.path.join(BASE, "src")
DEMO_DIR      = os.path.join(BASE, "data", "demo_images")
MODEL_PATH    = os.path.join(BASE, "models", "stage2_webcam_v3.h5")
FT_MODEL_PATH = os.path.join(BASE, "models", "stage2_mask_ft.h5")
META_PATH     = os.path.join(DEMO_DIR, "demo_meta.json")
sys.path.insert(0, SRC_DIR)

# ── Google Drive 모델 다운로드 (클라우드 배포 시) ──────────
GDRIVE_IDS = {
    MODEL_PATH:    "1JJ5Z5TfQbNmvJrXjofsaneWj03tQ7ezi",
    FT_MODEL_PATH: "1NzxToe4xIQyc9hnJIOrlkQ9_qHYO4Fu9",
}

def ensure_models():
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    for path, file_id in GDRIVE_IDS.items():
        if not os.path.exists(path):
            with st.spinner(f"모델 다운로드 중: {Path(path).name}..."):
                gdown.download(id=file_id, output=path, quiet=False)

ensure_models()

from xai_explainer import explain

# ── 상수 ─────────────────────────────────────────────────
LIVE_MEAN = {"laplacian": 383.0, "fft_high": 1134.0}
LAP_THRESHOLD_PRINT = 210

SPOOF_KO = {
    0: "Live (실제 얼굴)",
    1: "Print Attack (인쇄 공격)",
    2: "Replay Attack (화면 재촬영)",
    3: "3D Mask (입체 마스크)",
}
REGION_MAP = {
    "upper-center": "forehead region", "upper-left": "forehead-left",
    "upper-right":  "forehead-right",  "mid-center": "nose and cheek area",
    "mid-left":     "left cheek",      "mid-right":  "right cheek",
    "lower-center": "mouth and chin",  "lower-left": "lower-left jaw",
    "lower-right":  "lower-right jaw", "full-face":  "entire face",
    "none":         "no concentrated region",
}

# ── 캐시 리소스 ───────────────────────────────────────────
@st.cache_resource
def get_models():
    orig = tf.keras.models.load_model(MODEL_PATH, compile=False)
    ft   = tf.keras.models.load_model(FT_MODEL_PATH, compile=False)
    return orig, ft

@st.cache_resource
def get_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

@st.cache_data
def load_demo_meta():
    if os.path.exists(META_PATH):
        return json.load(open(META_PATH, encoding="utf-8"))
    return {}

# ── 유틸 함수 ─────────────────────────────────────────────
def detect_heatmap_region(heatmap_raw, threshold=0.5):
    if heatmap_raw is None: return "unknown"
    hm = cv2.resize(heatmap_raw, (9, 9))
    active = hm >= threshold
    if active.mean() > 0.6: return "full-face"
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]
    if len(rows) == 0: return "none"
    v = "upper" if rows.mean() < 3 else ("lower" if rows.mean() > 6 else "mid")
    h = "left"  if cols.mean() < 3 else ("right" if cols.mean() > 6 else "center")
    return f"{v}-{h}"

def anchor_detail(lap, fft):
    lap_pct = lap / LIVE_MEAN["laplacian"] * 100
    fft_pct = fft / LIVE_MEAN["fft_high"]  * 100
    if   lap < LIVE_MEAN["laplacian"] * 0.5:
        lap_interp = f"매우 낮음 — 종이/화면 질감 특성 (Live 평균의 {lap_pct:.0f}%)"
    elif lap < LIVE_MEAN["laplacian"] * 0.8:
        lap_interp = f"낮음 — 표면 아티팩트 가능성 (Live 평균의 {lap_pct:.0f}%)"
    elif lap > LIVE_MEAN["laplacian"] * 1.2:
        lap_interp = f"높음 — 마스크 경계선 특성 (Live 평균의 {lap_pct:.0f}%)"
    else:
        lap_interp = f"정상 — Live 범위 내 (Live 평균의 {lap_pct:.0f}%)"
    if   fft < LIVE_MEAN["fft_high"] * 0.6:
        fft_interp = f"매우 낮음 — 압축/재촬영 아티팩트 (Live 평균의 {fft_pct:.0f}%)"
    elif fft < LIVE_MEAN["fft_high"] * 0.85:
        fft_interp = f"낮음 — 화면/인쇄 감쇠 특성 (Live 평균의 {fft_pct:.0f}%)"
    else:
        fft_interp = f"정상 — 고주파 충분 (Live 평균의 {fft_pct:.0f}%)"
    return lap_interp, fft_interp

def build_image_caption(verdict, spoof_type_idx, anchor_stats, heatmap_raw):
    lap = anchor_stats.get("laplacian", 0)
    fft = anchor_stats.get("fft_high", 0)
    region_desc = REGION_MAP.get(detect_heatmap_region(heatmap_raw), "face region")
    lap_desc = (
        f"very low sharpness (Lap={lap:.0f}, {lap/LIVE_MEAN['laplacian']*100:.0f}% of live avg)"
        if lap < LIVE_MEAN["laplacian"] * 0.5 else
        f"reduced sharpness (Lap={lap:.0f}, {lap/LIVE_MEAN['laplacian']*100:.0f}% of live avg)"
        if lap < LIVE_MEAN["laplacian"] * 0.8 else
        f"high edge contrast (Lap={lap:.0f}, {lap/LIVE_MEAN['laplacian']*100:.0f}% of live avg)"
        if lap > LIVE_MEAN["laplacian"] * 1.2 else
        f"normal sharpness (Lap={lap:.0f}, {lap/LIVE_MEAN['laplacian']*100:.0f}% of live avg)"
    )
    fft_desc = (
        f"low high-freq energy (FFT={fft:.0f}, {fft/LIVE_MEAN['fft_high']*100:.0f}% of live avg)"
        if fft < LIVE_MEAN["fft_high"] * 0.6 else
        f"suppressed high-freq (FFT={fft:.0f}, {fft/LIVE_MEAN['fft_high']*100:.0f}% of live avg)"
        if fft < LIVE_MEAN["fft_high"] * 0.85 else
        f"normal high-freq energy (FFT={fft:.0f}, {fft/LIVE_MEAN['fft_high']*100:.0f}% of live avg)"
    )
    type_hints = {
        0: "No spoofing artifacts — classified as live face.",
        1: "Paper-based attack: flat texture and ink dot pattern visible.",
        2: "Screen replay: digital display interference pattern observed.",
        3: "3D mask: rigid boundary and synthetic texture inconsistency.",
    }
    return (f"Model focused on {region_desc}. "
            f"Texture: {lap_desc}, {fft_desc}. "
            f"{type_hints.get(spoof_type_idx, '')}")

def explain_v2(img_bgr, img_path_str, thr=0.75):
    _, ft_model = get_models()
    r = explain(img_bgr, img_path=img_path_str, threshold=thr)
    if r["verdict"] == "FAKE":
        lap = r["anchor_stats"]["laplacian"]
        inp = np.expand_dims(
            cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (224, 224)).astype("float32") / 255.0, 0
        )
        ft_preds = ft_model.predict(inp, verbose=0)
        ft_probs = {i: float(p) for i, p in enumerate(ft_preds[1][0])}
        if   ft_probs.get(2, 0) >= 0.40: final_idx = 2
        elif lap < LAP_THRESHOLD_PRINT:   final_idx = 1
        else:                             final_idx = 3
        r["spoof_type_idx"]  = final_idx
        r["spoof_type_name"] = SPOOF_KO[final_idx]
    if r["verdict"] == "REAL":
        r["spoof_type_name"] = "Live (실제 얼굴)"
    r["llava_caption"] = build_image_caption(
        r["verdict"], r["spoof_type_idx"], r["anchor_stats"], r["heatmap_raw"]
    )
    return r

def crop_face(img_bgr, target_size=224, margin_ratio=0.3):
    cascade = get_cascade()
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
    if len(faces) == 0:
        h, w = img_bgr.shape[:2]; s = min(h, w)
        return cv2.resize(img_bgr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], (target_size, target_size)), False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    ih, iw = img_bgr.shape[:2]
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    x1, y1 = max(0, x-mx), max(0, y-my)
    x2, y2 = min(iw, x+w+mx), min(ih, y+h+my)
    return cv2.resize(img_bgr[y1:y2, x1:x2], (target_size, target_size)), True

# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="FAS XAI Demo", page_icon="🛡️", layout="wide")

st.title("🛡️ Face Anti-Spoofing — 3-Layer XAI Demo")
st.caption("멀티태스크 MobileNetV2 + Grad-CAM + 수치 앵커링 + 이미지 기반 자연어 설명")

demo_meta = load_demo_meta()

LABELS = {
    "upload"          : "📤 직접 업로드 (크롭 포함)",
    "01_webcam_live"  : "① 본인 Live — REAL 기대",
    "02_print_attack" : "② Print Attack — FAKE 기대",
    "03_replay_attack": "③ Replay Attack — FAKE 기대",
    "04_mask_attack"  : "④ Mask Attack — FAKE 기대",
}

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    thr = st.slider("판정 임계값", 0.50, 0.99, 0.75, 0.01)
    st.divider()
    st.subheader("🎬 시연 모드")
    mode = st.radio("입력 방식", list(LABELS.values()), index=0)
    st.divider()
    st.subheader("📊 FAR 대시보드")
    import pandas as pd
    st.dataframe(pd.DataFrame({
        "공격 유형": ["Print", "Replay", "Mask", "전체"],
        "FAR (%)":  [1.33, 0.00, 0.00, 0.44],
        "목표치 (%)": [5, 10, 8, 5],
    }).set_index("공격 유형"), use_container_width=True)
    st.divider()
    st.caption(f"Live 평균 기준값\nLaplacian: {LIVE_MEAN['laplacian']:.0f}\nFFT High: {LIVE_MEAN['fft_high']:.0f}")

# ── 이미지 로드 ───────────────────────────────────────────
img_bgr, cur_path = None, ""

if mode == "📤 직접 업로드 (크롭 포함)":
    st.subheader("📤 이미지 업로드")
    st.caption("얼굴이 잘 보이는 사진을 업로드하면 자동으로 크롭 후 분석합니다.")
    uploaded_file = st.file_uploader(
        "얼굴 이미지 선택 (.jpg / .png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        arr     = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_raw is None:
            st.error("❌ 이미지 디코딩 실패")
        else:
            col1, col2, col3 = st.columns([4, 1, 4])
            with col1:
                st.markdown("**① 업로드 원본**")
                st.image(cv2.cvtColor(cv2.resize(img_raw, (224, 224)), cv2.COLOR_BGR2RGB),
                         caption=f"{img_raw.shape[1]}×{img_raw.shape[0]}", use_container_width=True)
            with st.spinner("얼굴 감지 중..."):
                img_cropped, detected = crop_face(img_raw)
            with col2:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown("### →")
            with col3:
                st.markdown("**② 얼굴 크롭 (224×224)**")
                st.image(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB),
                         caption="✅ 얼굴 감지 성공" if detected else "⚠️ center crop fallback",
                         use_container_width=True)
            st.markdown("**③ XAI 분석 결과** ↓")
            st.divider()
            img_bgr  = img_cropped
            cur_path = uploaded_file.name
    else:
        st.info("👆 위 버튼을 눌러 얼굴 이미지를 업로드하세요.")
else:
    key_map = {v: k for k, v in LABELS.items()}
    key     = key_map.get(mode)
    if key and key in demo_meta:
        img_path = demo_meta[key]["dst"]
        # 상대경로로 변환 (클라우드 환경 대응)
        fname = Path(img_path).name
        local_path = os.path.join(DEMO_DIR, fname)
        if os.path.exists(local_path):
            img_bgr  = cv2.imread(local_path)
            cur_path = local_path
        else:
            st.warning(f"시연 이미지 없음: {fname}")
    else:
        st.warning("시연 이미지가 없습니다.")

# ── XAI 결과 ─────────────────────────────────────────────
if img_bgr is not None:
    with st.spinner("🔍 분석 중..."):
        r = explain_v2(img_bgr, cur_path, thr)

    verdict = r["verdict"]
    prob    = r["spoof_prob"]
    stype   = r["spoof_type_name"]
    caption = r["llava_caption"]
    lap     = r["anchor_stats"]["laplacian"]
    fft     = r["anchor_stats"]["fft_high"]
    lap_interp, fft_interp = anchor_detail(lap, fft)

    if verdict == "REAL":
        st.success(f"✅ REAL (Live) — spoof_prob: {prob:.1%}  |  threshold: {thr:.2f}")
    else:
        st.error(f"🚨 FAKE ({stype}) — spoof_prob: {prob:.1%}  |  threshold: {thr:.2f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📷 원본 이미지")
        st.image(cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB),
                 use_container_width=True)
        st.caption(f"유형: {stype}")
    with c2:
        st.subheader("🔥 Layer 1: Grad-CAM")
        st.image(r["heatmap_overlay"], use_container_width=True)
        region = detect_heatmap_region(r["heatmap_raw"])
        st.caption(f"활성 영역: {REGION_MAP.get(region, region)}")
    with c3:
        st.subheader("📐 Layer 2: 수치 앵커링")
        m1, m2 = st.columns(2)
        m1.metric("Laplacian", f"{lap:.0f}",
                  delta=f"{lap/LIVE_MEAN['laplacian']*100:.0f}% of Live avg",
                  delta_color="normal" if verdict == "REAL" else "inverse")
        m2.metric("FFT High",  f"{fft:.0f}",
                  delta=f"{fft/LIVE_MEAN['fft_high']*100:.0f}% of Live avg",
                  delta_color="normal" if verdict == "REAL" else "inverse")
        st.info(f"**Laplacian:** {lap_interp}")
        st.info(f"**FFT High:** {fft_interp}")
        st.subheader("💬 Layer 3: 자연어 설명")
        st.write(caption)

    with st.expander("🔬 상세 수치"):
        st.json({
            "verdict": verdict, "spoof_prob": round(float(prob), 4),
            "spoof_type": stype, "threshold": thr,
            "laplacian": {"value": lap, "live_avg": LIVE_MEAN["laplacian"],
                          "ratio_pct": round(lap/LIVE_MEAN["laplacian"]*100, 1)},
            "fft_high":  {"value": fft, "live_avg": LIVE_MEAN["fft_high"],
                          "ratio_pct": round(fft/LIVE_MEAN["fft_high"]*100, 1)},
            "layer3_caption": caption,
        })