import cv2, numpy as np
from collections import deque

def resize_short(bgr, short=720):
    h, w = bgr.shape[:2]
    if min(h, w) == short:
        return bgr
    if h < w:
        nw = int(w * (short / h)); nh = short
    else:
        nh = int(h * (short / w)); nw = short
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def variance_of_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def tenengrad(gray, ksize=3):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx*gx + gy*gy))

def edge_density(gray, low=80, high=160):
    e = cv2.Canny(gray, low, high)
    return float(np.count_nonzero(e))/e.size

def percentile(arr, q):
    arr = np.asarray(arr, dtype=np.float64)
    return float(np.percentile(arr, q))

def dark_channel_mean(bgr, k=15):
    # hızlı DCP approx: per-pixel min(R,G,B) sonra min filtresi ~ erozyon
    m = np.min(bgr, axis=2)
    k = k if k % 2 == 1 else k+1
    dc = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    return float(np.mean(dc))

def colorfulness_hs(img_bgr):
    # Hasler–Süsstrunk “colorfulness”
    b,g,r = cv2.split(img_bgr.astype(np.float32))
    rg = np.abs(r - g); yb = np.abs(0.5*(r+g) - b)
    mean_rg, std_rg = np.mean(rg), np.std(rg)
    mean_yb, std_yb = np.mean(yb), np.std(yb)
    cf = np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2)
    # HSV S ortalaması da dönelim
    s = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()
    return float(cf), float(s)

def luma_hist_stats(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    cdf = np.cumsum(hist) / (hist.sum() + 1e-9)
    p1 = np.searchsorted(cdf, 0.01)
    p99 = np.searchsorted(cdf, 0.99)
    clip_lo = float(cdf[10])          # ~%4 proxy
    clip_hi = float(1.0 - cdf[-11])   # ~%4 proxy
    return dict(meanY=float(gray.mean()), dyn_range=int(p99-p1),
                clip_lo=clip_lo, clip_hi=clip_hi)

def grid(gray, gy=3, gx=3):
    H,W = gray.shape
    ys = np.linspace(0,H,gy+1,dtype=int); xs = np.linspace(0,W,gx+1,dtype=int)
    for i in range(gy):
        for j in range(gx):
            yield i,j, gray[ys[i]:ys[i+1], xs[j]:xs[j+1]]

class RunningMin:
    """ Statik lekeleri yakalamak için yavaş bir arkaplan akümülatörü. """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.init = False
        self.acc = None
    def update(self, gray):
        g = gray.astype(np.float32)
        if not self.init:
            self.acc = g.copy(); self.init=True
        else:
            cv2.accumulateWeighted(g, self.acc, self.alpha)
        return self.acc.astype(np.uint8)

class CamQA:
    """
    Dört hedef metrik: sharpness, brightness, color_intensity, lens_cleanliness
    Ayrıca 0–1 arası skorlar döndürür.
    """
    def __init__(self, short_side=720, ema=0.2, mode='auto'):
        self.short_side = short_side
        self.ema = ema
        self.mode = mode  # 'day' | 'night' | 'auto'
        self.min_accum = RunningMin(alpha=0.01)   # lens kirliliği için
        self.buf = {}     # EMA cache
        # referans normalizasyon sabitleri (sahada kalibre edilebilir)
        self.ref_vol_day  = 300.0
        self.ref_ten_day  = 1200.0
        self.ref_vol_night= 180.0
        self.ref_ten_night= 800.0

    def _ema(self, k, v):
        if k not in self.buf: self.buf[k]=float(v)
        else: self.buf[k] = (1-self.ema)*self.buf[k]+self.ema*float(v)
        return self.buf[k]

    def _scene_mode(self, meanY):
        if self.mode in ('day','night'): return self.mode
        # auto
        if meanY >= 110: return 'day'
        if meanY <= 75:  return 'night'
        return 'twilight'

    def analyze(self, bgr):
        bgr = resize_short(bgr, self.short_side)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        vol = variance_of_laplacian(gray)
        ten = tenengrad(gray)
        ed_full = edge_density(gray)

        # 3x3 patch: lokal bulanıklık/kir riskini görmek için P10’lar
        vols, eds = [], []
        for _,_,p in grid(gray,3,3):
            if p.size==0: continue
            vols.append(variance_of_laplacian(p))
            eds.append(edge_density(p))
        vol_p10 = percentile(vols,10) if vols else vol
        ed_p10  = percentile(eds,10)  if eds  else ed_full

        # parlaklık/kontrast/clipping
        lstats = luma_hist_stats(gray)
        meanY, dynr, clip_lo, clip_hi = lstats["meanY"], lstats["dyn_range"], lstats["clip_lo"], lstats["clip_hi"]

        # renk yoğunluğu
        cf, hsv_s = colorfulness_hs(bgr)

        # dark channel & statik leke maskesi
        dc_mean = dark_channel_mean(bgr, k=15)
        bg_est  = self.min_accum.update(gray)
        diff    = cv2.absdiff(bg_est, gray)
        dark    = bg_est < 40
        stable  = diff < 6
        static_mask = (dark & stable).astype(np.uint8)
        static_ratio = float(static_mask.sum())/static_mask.size

        # EMA ile yumuşat
        vol_s   = self._ema("vol", vol)
        ten_s   = self._ema("ten", ten)
        ed_s    = self._ema("ed", ed_full)
        ed10_s  = self._ema("ed10", ed_p10)
        vol10_s = self._ema("vol10", vol_p10)
        meanY_s = self._ema("meanY", meanY)
        dynr_s  = self._ema("dynr", dynr)
        clip_s  = self._ema("clip", clip_lo+clip_hi)
        cf_s    = self._ema("cf", cf)
        sat_s   = self._ema("sat", hsv_s)
        dc_s    = self._ema("dc", dc_mean)
        stat_s  = self._ema("stat", static_ratio)

        # --- skorlar ---
        # 1) Keskinlik (day/night referansına göre normalize)
        mode = self._scene_mode(meanY_s)
        if mode == 'night':
            s_vol = np.clip(vol_s / (self.ref_vol_night+1e-9), 0, 1)
            s_ten = np.clip(ten_s / (self.ref_ten_night+1e-9), 0, 1)
        else:
            s_vol = np.clip(vol_s / (self.ref_vol_day+1e-9), 0, 1)
            s_ten = np.clip(ten_s / (self.ref_ten_day+1e-9), 0, 1)
        # lokal P10 ile kir/blur cezalandır
        s_vol10 = np.clip(vol10_s / (0.6*(self.ref_vol_day if mode!='night' else self.ref_vol_night)+1e-9), 0, 1)
        sharp_score = 0.55*s_vol + 0.30*s_ten + 0.15*s_vol10

        # 2) Parlaklık (hedef bandına yakınlık + clipping cezası)
        # hedef bant: day: 110–140, night: 60–90
        if mode=='night':
            target_lo, target_hi = 60, 90
            good_dr = 120.0
        else:
            target_lo, target_hi = 110, 140
            good_dr = 180.0
        within = 1.0 - np.clip(abs((meanY_s - (0.5*(target_lo+target_hi))))/(0.5*(target_hi-target_lo)+1e-9), 0, 1)
        s_dr   = np.clip(dynr_s / good_dr, 0, 1)
        clip_pen = np.clip(clip_s / 0.28, 0, 1) # %28 toplam clipping “kötü”
        brightness_score = np.clip(0.6*within + 0.4*s_dr - 0.35*clip_pen, 0, 1)

        # 3) Renk yoğunluğu (saturation + colorfulness; aşırı parlaklıkta tavan yaptırma)
        # normalize: HSV S: 0–255 → /200; colorfulness ~ 0–150+ → /100
        s_sat = np.clip(sat_s/200.0, 0, 1)
        s_cf  = np.clip(cf_s/100.0, 0, 1)
        # aşırı clipping varsa (parlama), renk güvenini düşür
        color_score = np.clip(0.6*s_sat + 0.4*s_cf, 0, 1) * (1.0 - 0.4*clip_pen)

        # 4) Mercek temizliği: düşük edge/lap P10 + yüksek dark channel + statik maske oranı cezalandırılır
        s_edge10 = np.clip(ed10_s/0.08, 0, 1)  # %8 patch-edge iyi kabul
        s_dc     = np.clip(1.0 - (dc_s-20.0)/100.0, 0, 1)  # dc 20→iyi, 120→kötü
        s_static = np.clip(1.0 - (stat_s/0.02), 0, 1)      # %2 statik koyu alan toleransı
        lens_clean_score = np.clip(0.5*s_edge10 + 0.25*s_dc + 0.25*s_static, 0, 1)

        return {
            "mode": mode,
            "raw": {
                "vol": vol_s, "tenengrad": ten_s, "vol_p10": vol10_s,
                "edge_density": ed_s, "edge_density_p10": ed10_s,
                "meanY": meanY_s, "dyn_range": dynr_s,
                "clip_sum": clip_s,
                "colorfulness": cf_s, "hsv_s": sat_s,
                "dark_channel_mean": dc_s, "static_dark_ratio": stat_s
            },
            "scores": {
                "sharpness": float(sharp_score),
                "brightness": float(brightness_score),
                "color_intensity": float(color_score),
                "lens_cleanliness": float(lens_clean_score)
            }
        }
