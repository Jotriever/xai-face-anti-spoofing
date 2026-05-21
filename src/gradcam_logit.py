
"""gradcam_logit.py — Phase 4-A: logit 기반 Grad-CAM (sigmoid 포화 해결)"""
import numpy as np
import cv2
import tensorflow as tf


def get_gradcam_logit(model, img_array, conv_layer_name, logit_layer_name):
    """sigmoid 직전 logit 기반 Grad-CAM — FAKE 100% 케이스도 히트맵 생성"""
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.get_layer(logit_layer_name).output
        ]
    )
    with tf.GradientTape() as tape:
        inp = tf.cast(img_array, tf.float32)
        conv_out, pred = grad_model(inp)
        pred_clipped = tf.clip_by_value(pred[:, 0], 1e-7, 1 - 1e-7)
        logit = tf.math.log(pred_clipped / (1.0 - pred_clipped))
        loss = logit
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img_bgr, heatmap, alpha=0.45):
    h, w = img_bgr.shape[:2]
    hm_r = cv2.resize(heatmap, (w, h))
    hm_c = cv2.applyColorMap(np.uint8(255 * hm_r), cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hm_rgb = cv2.cvtColor(hm_c, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, hm_rgb, alpha, 0)


def extract_active_mask(heatmap, img_size=224, top_percent=30):
    hm_resized = cv2.resize(heatmap, (img_size, img_size))
    threshold = np.percentile(hm_resized, 100 - top_percent)
    return hm_resized >= threshold


def anchored_pixel_stats(img_bgr, mask):
    img_gray = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2GRAY)
    mask_uint8 = mask.astype(np.uint8) * 255
    masked = cv2.bitwise_and(img_gray, img_gray, mask=mask_uint8)
    lap = cv2.Laplacian(masked.astype(np.float64), cv2.CV_64F)
    laplacian_var = lap[mask].var() if mask.any() else 0.0
    f = np.fft.fftshift(np.fft.fft2(masked))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2; r = min(h,w)//6
    Y, X = np.ogrid[:h, :w]
    hfm = (Y-cy)**2 + (X-cx)**2 > r**2
    fft_energy = mag[hfm].mean() if hfm.any() else 0.0
    return {'laplacian': float(laplacian_var), 'fft_energy': float(fft_energy)}
