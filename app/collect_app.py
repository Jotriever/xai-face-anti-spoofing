"""
app/collect_app.py — 웹캠 Live 이미지 수집 전용 앱
실행: streamlit run app/collect_app.py
"""

import os, time
import numpy as np
import cv2
import streamlit as st
from pathlib import Path

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR   = os.path.join(BASE, "data", "webcam_live")
os.makedirs(SAVE_DIR, exist_ok=True)

st.set_page_config(page_title="Live 이미지 수집", page_icon="📷", layout="centered")
st.title("📷 웹캠 Live 이미지 수집")

saved = list(Path(SAVE_DIR).glob("*.jpg"))
st.metric("저장된 이미지", str(len(saved)) + "장", "목표 50장")
st.progress(min(len(saved) / 50, 1.0))
st.markdown("---")

photo = st.camera_input("얼굴을 촬영하세요")

if photo is not None:
    arr    = np.frombuffer(photo.read(), np.uint8)
    img    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    fname  = "webcam_" + str(int(time.time())) + ".jpg"
    fpath  = os.path.join(SAVE_DIR, fname)
    cv2.imwrite(fpath, img)

    saved  = list(Path(SAVE_DIR).glob("*.jpg"))
    st.success("✅ 저장 완료! 총 " + str(len(saved)) + "장")
    st.image(
        cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB),
        caption=fname, width=224
    )

st.markdown("---")
st.caption("촬영 후 Clear photo 누르고 다시 찍으면 연속 수집 가능합니다.")
