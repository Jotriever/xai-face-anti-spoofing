"""
src/xai_explainer.py — Phase 4-E (v2)
3계층 XAI 통합 파이프라인
"""

import os, json, random
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE, "models", "stage2_webcam_v3.h5")
CAPTION_JSON = os.path.join(BASE, "results", "phase4", "llava_captions.json")

# ── 레이어 이름 ───────────────────────────────────────────
TARGET_LAYER = 'Conv_1_bn'
LOGIT_LAYER  = 'binary'
SPOOF_LAYER  = 'spoof'

# ── 전역 상태 ─────────────────────────────────────────────
_model        = None
_caption_db   = {}
_caption_pool = {"live": [], "print": [], "replay": [], "mask": []}

# ── 기준값 ────────────────────────────────────────────────
ANCHOR_BASELINE = {
    "live":   {"laplacian": 383, "fft_high": 1134},
    "print":  {"laplacian": 318, "fft_high": 1042},
    "replay": {"laplacian": 319, "fft_high":  944},
    "mask":   {"laplacian": 480, "fft_high": 1134},
}

SPOOF_KO = {
    0: "Live (실제 얼굴)",
    1: "Print Attack (인쇄 공격)",
    2: "Replay Attack (화면 재촬영)",
    3: "3D Mask (입체 마스크)",
}
SPOOF_EN_HINT = {
    1: "인쇄물 특유의 평탄한 피부 질감과 낮은 고주파 에너지가 관측됩니다.",
    2: "화면 재촬영 특유의 모아레 패턴 및 고주파 에너지 감소가 감지됩니다.",
    3: "마스크 경계부에 비정상적 선명도 및 피부 질감 불일치가 나타납니다.",
}
ILLUMINATION_NAMES = {0: "자연광", 1: "실내 형광등", 2: "역광", 3: "저조도"}
ENVIRONMENT_NAMES  = {0: "실내", 1: "실외"}


def load_model_once(model_path=MODEL_PATH, caption_json=CAPTION_JSON):
    global _model, _caption_db, _caption_pool
    if _model is None:
        _model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ 모델 로드:", model_path)
    if not _caption_db and Path(caption_json).exists():
        with open(caption_json) as f:
            records = json.load(f)
        for rec in records.get("results", []):
            stem    = Path(rec["img_path"]).stem
            caption = rec.get("caption", "")
            _caption_db[stem] = caption
            cat = rec.get("category", "")
            if cat in _caption_pool and caption:
                _caption_pool[cat].append(caption)
        print("✅ 캡션 DB 로드:", len(_caption_db), "개")
    return _model


def denoise_webcam(img_bgr):
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.2)
    return cv2.bilateralFilter(blurred, d=5, sigmaColor=30, sigmaSpace=30)


def preprocess(img_bgr, size=224):
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res  = cv2.resize(img_rgb, (size, size))
    return np.expand_dims(img_res.astype("float32") / 255.0, 0)


def get_gradcam_logit(model, img_array, conv_layer_name, logit_layer_name):
    try:
        logit_output = model.get_layer(logit_layer_name).output
    except ValueError:
        return np.zeros((7, 7))
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, logit_output],
    )
    with tf.GradientTape() as tape:
        inp            = tf.cast(img_array, tf.float32)
        conv_out, pred = grad_model(inp)
        p_clipped      = tf.clip_by_value(pred[:, 0], 1e-7, 1 - 1e-7)
        logit          = tf.math.log(p_clipped / (1.0 - p_clipped))
    grads = tape.gradient(logit, conv_out)
    if grads is None:
        return np.zeros((7, 7))
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img_bgr, heatmap, alpha=0.45):
    h, w  = img_bgr.shape[:2]
    hmr   = cv2.resize(heatmap, (w, h))

    # ── 얼굴 마스크 적용 ──────────────────────────────────
    _xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "haarcascade_frontalface_default.xml")
    if not os.path.exists(_xml):
        _xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    
    _cascade = cv2.CascadeClassifier()
    _cascade.load(_xml)
    
    if not _cascade.empty():
        # 기존 얼굴 마스크 로직
        _gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _faces = _cascade.detectMultiScale(_gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(40,40))
        if len(_faces) > 0:
            _mask = np.zeros((h, w), dtype=np.float32)
            _x, _y, _fw, _fh = sorted(_faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            _mx, _my = int(_fw*0.4), int(_fh*0.5)
            _x1,_y1  = max(0,_x-_mx),    max(0,_y-_my)
            _x2,_y2  = min(w,_x+_fw+_mx), min(h,_y+_fh+_my)
            _mask[_y1:_y2, _x1:_x2] = 1.0
            hmr = hmr * _mask
            _thresh = np.percentile(hmr[hmr > 0], 70) if hmr.max() > 0 else 0
            hmr = np.where(hmr >= _thresh, hmr, 0)
            if hmr.max() > 0:
                hmr = hmr / hmr.max()

    hmc   = cv2.applyColorMap(np.uint8(255 * hmr), cv2.COLORMAP_JET)
    img_r = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hmc_r = cv2.cvtColor(hmc,     cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_r, 1 - alpha, hmc_r, alpha, 0)


def compute_pixel_features(img_bgr, mask_224=None):
    img_r = cv2.resize(img_bgr, (224, 224))
    img_g = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    if mask_224 is not None and mask_224.sum() >= 16:
        lap     = cv2.Laplacian(img_g, cv2.CV_64F)
        lap_var = float(lap[mask_224].var())
    else:
        lap     = cv2.Laplacian(img_g, cv2.CV_64F)
        lap_var = float(lap.var())
        img_g   = img_g
    sub      = img_g.astype(np.float32)
    f        = np.fft.fftshift(np.fft.fft2(sub))
    mag      = np.abs(f)
    h, w     = mag.shape
    cy, cx   = h // 2, w // 2
    r        = min(h, w) // 6
    Y, X     = np.ogrid[:h, :w]
    d2       = (Y - cy)**2 + (X - cx)**2
    fft_high = float(mag[d2 > r**2].mean())
    fft_low  = float(mag[d2 <= r].mean())
    return {"laplacian": round(lap_var, 1), "fft_high": round(fft_high, 1), "fft_low": round(fft_low, 1)}


def interpret_anchors(stats, category="unknown"):
    base     = ANCHOR_BASELINE.get(category, ANCHOR_BASELINE["live"])
    lines    = []
    lap_diff = stats["laplacian"] - base["laplacian"]
    if   lap_diff < -40: lines.append("질감 평탄 (선명도 낮음)")
    elif lap_diff >  40: lines.append("경계선 강조 (선명도 높음)")
    else:                lines.append("선명도 보통")
    fft_diff = stats["fft_high"] - base["fft_high"]
    if   fft_diff < -80: lines.append("고주파 에너지 부재 (압축/재촬영 흔적)")
    elif fft_diff >  80: lines.append("고주파 에너지 과잉 (경계 아티팩트)")
    else:                lines.append("고주파 에너지 정상")
    return " / ".join(lines)


def get_llava_caption(img_path, category=None):
    stem = Path(img_path).stem if img_path else ""
    if stem in _caption_db:
        return _caption_db[stem]
    if category and category in _caption_pool and _caption_pool[category]:
        return random.choice(_caption_pool[category])
    return ""


def build_xai_text(verdict, spoof_prob, spoof_type_idx,
                   anchor_stats, anchor_interp,
                   llava_caption="", illum_idx=None, env_idx=None, webcam=False):
    lines      = []
    verdict_ko = "🟢 실제 얼굴 (REAL)" if verdict == "REAL" else "🔴 위조 공격 감지 (FAKE)"
    spoof_ko   = SPOOF_KO.get(spoof_type_idx, "유형 " + str(spoof_type_idx))
    lines.append("**" + verdict_ko + "** — 신뢰도 " + "{:.1%}".format(spoof_prob))
    lines.append("**예측 공격 유형:** " + spoof_ko)
    if webcam: lines.append("**입력:** 웹캠 (고주파 노이즈 보정 적용)")
    lines.append("")
    lines.append("**[Layer 2 — 수치 앵커링]**")
    lines.append("> Laplacian: **" + str(anchor_stats["laplacian"]) + "**  |  FFT 고주파: **" + str(anchor_stats["fft_high"]) + "**")
    lines.append("> 해석: " + anchor_interp)
    hint = SPOOF_EN_HINT.get(spoof_type_idx)
    if hint and verdict == "FAKE": lines.append("> ✏️ " + hint)
    lines.append("")
    lines.append("**[Layer 3 — VLM 자연어 분석]**")
    if llava_caption: lines.append("> " + llava_caption)
    else: lines.append("> *(캡션 없음)*")
    return "\n".join(lines)


def explain(img_bgr, img_path=None, category=None,
            threshold=0.75, heatmap_threshold=0.4,
            webcam=False, illum_idx=None, env_idx=None):
    model = load_model_once()
    img_for_stats = img_bgr
    img_for_model = denoise_webcam(img_bgr) if webcam else img_bgr
    inp   = preprocess(img_for_model)
    preds = model.predict(inp, verbose=0)
    if isinstance(preds, list) and len(preds) >= 2:
        spoof_prob     = float(preds[0][0][0])
        spoof_type_idx = int(np.argmax(preds[1][0]))
    else:
        spoof_prob     = float(preds[0][0]) if hasattr(preds[0], "__len__") else float(preds[0])
        spoof_type_idx = 0
    verdict         = "FAKE" if spoof_prob >= threshold else "REAL"
    spoof_type_name = SPOOF_KO.get(spoof_type_idx, "유형 " + str(spoof_type_idx))
    heatmap_raw     = get_gradcam_logit(model, inp, TARGET_LAYER, LOGIT_LAYER)
    heatmap_overlay = overlay_heatmap(img_bgr, heatmap_raw)
    cat_key         = {1: "print", 2: "replay", 3: "mask"}.get(spoof_type_idx, "live")
    hm_224          = cv2.resize(heatmap_raw, (224, 224))
    mask_224        = hm_224 >= heatmap_threshold
    anchor_stats    = compute_pixel_features(img_for_stats, mask_224 if mask_224.any() else None)
    anchor_interp   = interpret_anchors(anchor_stats, category=cat_key)
    llava_caption   = get_llava_caption(img_path, category=category) if img_path else ""
    xai_text        = build_xai_text(verdict, spoof_prob, spoof_type_idx,
                                     anchor_stats, anchor_interp, llava_caption,
                                     illum_idx, env_idx, webcam=webcam)
    return {
        "verdict": verdict, "spoof_prob": spoof_prob,
        "spoof_type_idx": spoof_type_idx, "spoof_type_name": spoof_type_name,
        "heatmap_raw": heatmap_raw, "heatmap_overlay": heatmap_overlay,
        "anchor_stats": anchor_stats, "anchor_interp": anchor_interp,
        "llava_caption": llava_caption, "xai_text": xai_text,
    }
