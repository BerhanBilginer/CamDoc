# src/predictive/features.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

# CamDoc çekirdekleri
from src.core.camqa import CamQA, resize_short
try:
    # Opsiyonel: BRISQUE/NIQE/TV için
    from src.core.advanced_metrics import AdvancedQualityMetrics  # type: ignore
    _ADV_AVAILABLE = True
except Exception:
    AdvancedQualityMetrics = None  # type: ignore
    _ADV_AVAILABLE = False


# --------------------------
# Temporal durum nesnesi
# --------------------------
@dataclass
class TemporalState:
    prev_gray: Optional[np.ndarray] = None
    prev_meanY: Optional[float] = None
    prev_ts: Optional[float] = None

    # Kayan pencereli akümülatör (flicker için minik dizi)
    # (çok basit: son N meanY değerinin var/yığımı)
    meanY_buffer: Optional[np.ndarray] = None
    meanY_ptr: int = 0
    meanY_len: int = 30  # ~1s @30 FPS, ayarlanabilir

    def push_meanY(self, m: float):
        if self.meanY_buffer is None:
            self.meanY_buffer = np.full((self.meanY_len,), m, dtype=np.float32)
            self.meanY_ptr = 1
        else:
            self.meanY_buffer[self.meanY_ptr % self.meanY_len] = m
            self.meanY_ptr += 1

    def meanY_stats(self) -> Tuple[float, float]:
        if self.meanY_buffer is None:
            return 0.0, 0.0
        arr = self.meanY_buffer
        return float(np.mean(arr)), float(np.std(arr))


# --------------------------
# Yardımcı temporal metrikler
# --------------------------
def _optical_flow_jitter(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Lucas–Kanade sparse optical flow ile kaba titreşim/jitter skoru.
    Çıktı: piksel başına ortalama hareket büyüklüğü (px/frame).
    """
    try:
        # Özellik noktaları (shi-tomasi)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7
        )
        if prev_pts is None or len(prev_pts) < 10:
            return 0.0

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        good_prev = prev_pts[st == 1]
        good_next = next_pts[st == 1] if next_pts is not None else None
        if good_next is None or len(good_next) == 0:
            return 0.0

        disp = good_next - good_prev
        mag = np.sqrt((disp ** 2).sum(axis=1))
        return float(np.mean(mag))
    except Exception:
        return 0.0


def _frame_diff_variance(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Frame farkının varyansı (hareket / ani değişim göstergesi).
    """
    try:
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.var(diff))
    except Exception:
        return 0.0


# --------------------------
# Ana API
# --------------------------
def build_features(
    bgr: np.ndarray,
    prev_state: Optional[TemporalState] = None,
    use_advanced: bool = False,
    use_temporal: bool = False,
    camqa: Optional[CamQA] = None,
) -> Tuple[Dict[str, float], TemporalState]:
    """
    Tek kareden (ve opsiyonel temporal durumdan) özellik vektörü çıkarır.

    Args:
        bgr: BGR frame (np.ndarray, uint8)
        prev_state: TemporalState veya None
        use_advanced: BRISQUE/NIQE/TV ekle
        use_temporal: optical-flow jitter, flicker vb. ekle
        camqa: dışarıdan verilmiş CamQA örneği (yoksa içinde oluşturur)

    Returns:
        (features: dict[str, float], new_state: TemporalState)
    """
    t0 = time.time()
    if prev_state is None:
        prev_state = TemporalState()

    # CamQA çalıştır
    qa = camqa if camqa is not None else CamQA(short_side=720, ema=0.2, mode='auto')
    # CamQA.analyze(bgr) senin sürümünde ya (bgr, out) ya da out döndürebiliyor;
    # aşağıyı kendi imzanla uyumlu tut (gerekirse bgr_, res = qa.analyze(bgr))
    res = qa.analyze(bgr) if isinstance(qa.analyze(bgr), dict) else qa.analyze(bgr)[1]

    # Çekirdek metrikleri topla
    raw = res["raw"]
    scores = res["scores"]
    mode = res["mode"]

    features: Dict[str, float] = {
        # --- CamQA skorları (0–1)
        "camqa_sharpness": float(scores["sharpness"]),
        "camqa_brightness": float(scores["brightness"]),
        "camqa_color_intensity": float(scores["color_intensity"]),
        "camqa_lens_cleanliness": float(scores["lens_cleanliness"]),
        # --- CamQA ham metrikleri
        "vol": float(raw["vol"]),
        "tenengrad": float(raw["tenengrad"]),
        "edge_density": float(raw["edge_density"]),
        "edge_density_p10": float(raw["edge_density_p10"]),
        "vol_p10": float(raw["vol_p10"]),
        "meanY": float(raw["meanY"]),
        "dyn_range": float(raw["dyn_range"]),
        "clip_sum": float(raw["clip_sum"]),
        "colorfulness": float(raw["colorfulness"]),
        "hsv_s": float(raw["hsv_s"]),
        "dark_channel_mean": float(raw["dark_channel_mean"]),
        "static_dark_ratio": float(raw["static_dark_ratio"]),
        # --- sahne modu (one-hot benzeri)
        "scene_day": 1.0 if mode == "day" else 0.0,
        "scene_night": 1.0 if mode == "night" else 0.0,
        "scene_twilight": 1.0 if mode == "twilight" else 0.0,
    }

    # --------------------------
    # Advanced (opsiyonel)
    # --------------------------
    if use_advanced:
        if not _ADV_AVAILABLE:
            # Kitaplık yoksa alanları doldur ama 0 yapma; None değil float bekliyoruz → NaN bırak
            features.update({
                "brisque_raw": np.nan,
                "niqe_raw": np.nan,
                "tv_raw": np.nan,
                "brisque_q": np.nan,  # normalize edilmiş kalite (0–1, yüksek iyi)
                "niqe_q": np.nan,
                "tv_q": np.nan,
                "adv_overall_q": np.nan,
            })
        else:
            try:
                adv = AdvancedQualityMetrics(device="cpu")
                adv_res = adv.analyze(bgr)
                # Raw (lower is better) + normalized (higher is better)
                features.update({
                    "brisque_raw": float(adv_res["raw"]["brisque"]),
                    "niqe_raw": float(adv_res["raw"]["niqe"]),
                    "tv_raw": float(adv_res["raw"]["total_variation"]),
                    "brisque_q": float(adv_res["normalized"]["brisque_quality"]),
                    "niqe_q": float(adv_res["normalized"]["niqe_quality"]),
                    "tv_q": float(adv_res["normalized"]["tv_quality"]),
                    "adv_overall_q": float(adv_res["overall_quality"]),
                })
            except Exception:
                # Hata toleransı: alanları NaN ile doldur
                features.update({
                    "brisque_raw": np.nan,
                    "niqe_raw": np.nan,
                    "tv_raw": np.nan,
                    "brisque_q": np.nan,
                    "niqe_q": np.nan,
                    "tv_q": np.nan,
                    "adv_overall_q": np.nan,
                })

    # --------------------------
    # Temporal (opsiyonel)
    # --------------------------
    gray = cv2.cvtColor(resize_short(bgr, qa.short_side), cv2.COLOR_BGR2GRAY)

    jitter = 0.0
    fdiff_var = 0.0
    flicker_std = 0.0
    meanY_mean = float(raw["meanY"])  # default

    if use_temporal and prev_state.prev_gray is not None:
        try:
            jitter = _optical_flow_jitter(prev_state.prev_gray, gray)
            fdiff_var = _frame_diff_variance(prev_state.prev_gray, gray)
        except Exception:
            jitter, fdiff_var = 0.0, 0.0

    # meanY buffer ile flicker std
    prev_state.push_meanY(float(raw["meanY"]))
    meanY_mean, flicker_std = prev_state.meanY_stats()

    features.update({
        "jitter_px": float(jitter),               # px/frame
        "frame_diff_var": float(fdiff_var),       # 0–+∞
        "meanY_flicker_std": float(flicker_std),  # luminance kararlılığı
        "meanY_running_mean": float(meanY_mean),
    })

    # Durumu güncelle
    prev_state.prev_gray = gray
    prev_state.prev_meanY = float(raw["meanY"])
    prev_state.prev_ts = t0

    # Basit süre metriği (profiling için)
    features["feature_extract_ms"] = (time.time() - t0) * 1000.0

    return features, prev_state
