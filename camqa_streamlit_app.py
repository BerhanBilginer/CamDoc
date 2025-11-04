import cv2, numpy as np, time, io
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# -------------------- Core CamQA (embedded) --------------------
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
    m = np.min(bgr, axis=2)
    k = k if k % 2 == 1 else k+1
    dc = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    return float(np.mean(dc))

def colorfulness_hs(img_bgr):
    b,g,r = cv2.split(img_bgr.astype(np.float32))
    rg = np.abs(r - g); yb = np.abs(0.5*(r+g) - b)
    mean_rg, std_rg = np.mean(rg), np.std(rg)
    mean_yb, std_yb = np.mean(yb), np.std(yb)
    cf = np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2)
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
                clip_lo=clip_lo, clip_hi=clip_hi, p1=int(p1), p99=int(p99), hist=hist)

def grid(gray, gy=3, gx=3):
    H,W = gray.shape
    ys = np.linspace(0,H,gy+1,dtype=int); xs = np.linspace(0,W,gx+1,dtype=int)
    for i in range(gy):
        for j in range(gx):
            yield i,j, gray[ys[i]:ys[i+1], xs[j]:xs[j+1]]

class RunningMin:
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
    def __init__(self, short_side=720, ema=0.2, mode='auto'):
        self.short_side = short_side
        self.ema = ema
        self.mode = mode
        self.min_accum = RunningMin(alpha=0.01)
        self.buf = {}
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
        if meanY >= 110: return 'day'
        if meanY <= 75:  return 'night'
        return 'twilight'
    def analyze(self, bgr):
        bgr = resize_short(bgr, self.short_side)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        vol = variance_of_laplacian(gray)
        ten = tenengrad(gray)
        ed_full = edge_density(gray)
        vols, eds = [], []
        for _,_,p in grid(gray,3,3):
            if p.size==0: continue
            vols.append(variance_of_laplacian(p))
            eds.append(edge_density(p))
        vol_p10 = percentile(vols,10) if vols else vol
        ed_p10  = percentile(eds,10)  if eds  else ed_full
        lstats = luma_hist_stats(gray)
        meanY, dynr, clip_lo, clip_hi = lstats["meanY"], lstats["dyn_range"], lstats["clip_lo"], lstats["clip_hi"]
        cf, hsv_s = colorfulness_hs(bgr)
        dc_mean = dark_channel_mean(bgr, k=15)
        bg_est  = self.min_accum.update(gray)
        diff    = cv2.absdiff(bg_est, gray)
        dark    = bg_est < 40
        stable  = diff < 6
        static_mask = (dark & stable).astype(np.uint8)
        static_ratio = float(static_mask.sum())/static_mask.size
        # EMA
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
        mode = self._scene_mode(meanY_s)
        if mode == 'night':
            s_vol = np.clip(vol_s / (self.ref_vol_night+1e-9), 0, 1)
            s_ten = np.clip(ten_s / (self.ref_ten_night+1e-9), 0, 1)
        else:
            s_vol = np.clip(vol_s / (self.ref_vol_day+1e-9), 0, 1)
            s_ten = np.clip(ten_s / (self.ref_ten_day+1e-9), 0, 1)
        s_vol10 = np.clip(vol10_s / (0.6*(self.ref_vol_day if mode!='night' else self.ref_vol_night)+1e-9), 0, 1)
        sharp_score = 0.55*s_vol + 0.30*s_ten + 0.15*s_vol10
        if mode=='night':
            target_lo, target_hi = 60, 90
            good_dr = 120.0
        else:
            target_lo, target_hi = 110, 140
            good_dr = 180.0
        within = 1.0 - np.clip(abs((meanY_s - (0.5*(target_lo+target_hi))))/(0.5*(target_hi-target_lo)+1e-9), 0, 1)
        s_dr   = np.clip(dynr_s / good_dr, 0, 1)
        clip_pen = np.clip(clip_s / 0.28, 0, 1)
        brightness_score = np.clip(0.6*within + 0.4*s_dr - 0.35*clip_pen, 0, 1)
        s_sat = np.clip(sat_s/200.0, 0, 1)
        s_cf  = np.clip(cf_s/100.0, 0, 1)
        color_score = np.clip(0.6*s_sat + 0.4*s_cf, 0, 1) * (1.0 - 0.4*clip_pen)
        s_edge10 = np.clip(ed10_s/0.08, 0, 1)
        s_dc     = np.clip(1.0 - (dc_s-20.0)/100.0, 0, 1)
        s_static = np.clip(1.0 - (stat_s/0.02), 0, 1)
        lens_clean_score = np.clip(0.5*s_edge10 + 0.25*s_dc + 0.25*s_static, 0, 1)

        out = {
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
        return bgr, out

# -------------------- Plot helpers --------------------
def plot_gauge(value, title="Overall Health"):
    v = float(np.clip(value, 0, 1))*100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={'suffix': "%", 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2E8B57"},
            'steps': [
                {'range': [0, 40], 'color': '#ff6b6b'},
                {'range': [40, 60], 'color': '#ffd166'},
                {'range': [60, 80], 'color': '#ffe89a'},
                {'range': [80, 100], 'color': '#b7e4c7'},
            ],
        },
        domain={'x': [0,1], 'y':[0,1]},
        title={'text': title}
    ))
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_radar(labels, values, title="Quality Radar"):
    theta = labels + [labels[0]]
    r = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', name='Scores'))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1], showticklabels=True)),
                      showlegend=False, height=350, margin=dict(l=30,r=30,t=40,b=10), title=title)
    return fig

def lens_pie(lens_score):
    clean = float(np.clip(lens_score,0,1))
    dirty = 1.0 - clean
    fig = go.Figure(data=[go.Pie(labels=['Clean','Dirty'], values=[clean, dirty], hole=0.45)])
    fig.update_traces(textinfo='label+percent', pull=[0.05, 0], marker=dict(colors=['#74c69d','#ef476f']))
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
    return fig

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="CamQA Dashboard", layout="wide")
st.title("📷 CamQA — Camera Health Dashboard")

with st.sidebar:
    st.header("Settings")
    src_mode = st.selectbox("Source", ["Sample image", "Upload image"])
    short_side = st.slider("Resize short side", 480, 1080, 720, 20)
    ema = st.slider("EMA smoothing", 0.0, 0.9, 0.2, 0.05)
    force_mode = st.selectbox("Scene mode", ["auto", "day", "night"])
    uploaded = None
    if src_mode == "Upload image":
        uploaded = st.file_uploader("Upload a frame (JPEG/PNG)", type=["jpg","jpeg","png"])
    st.caption("Overall Health = weighted mean of four scores")

# Sample image (placeholder)
sample_path = "test/images/img1.jpg"

# Load image
if src_mode == "Upload image" and uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    bgr = cv2.imread(sample_path)

if bgr is None:
    st.error("Could not load image. Please upload a valid image."); st.stop()

qa = CamQA(short_side=short_side, ema=ema, mode=force_mode)
vis_img, res = qa.analyze(bgr)

# Convert to RGB for display
rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

# ---- Layout ----
col_img, col_gauge = st.columns([3,2], vertical_alignment="top")

with col_img:
    st.subheader("Source Image")
    st.image(rgb, use_column_width=True)

with col_gauge:
    # Overall Health: weighted mean (you can tweak weights in sidebar in an advanced version)
    s = res["scores"]
    overall = 0.30*s["sharpness"] + 0.30*s["brightness"] + 0.20*s["color_intensity"] + 0.20*s["lens_cleanliness"]
    st.plotly_chart(plot_gauge(overall, title=f"Overall Health ({res['mode']})"), use_container_width=True)
    st.metric("Sharpness", f"{s['sharpness']:.3f}")
    st.metric("Brightness", f"{s['brightness']:.3f}")
    st.metric("Color Intensity", f"{s['color_intensity']:.3f}")
    st.metric("Lens Cleanliness", f"{s['lens_cleanliness']:.3f}")

# Second row: score bars & radar
st.markdown("---")
c1, c2, c3 = st.columns([2,2,2])

with c1:
    st.subheader("Score Breakdown")
    labels = ["Sharpness","Brightness","Color Intensity","Lens Cleanliness"]
    vals = [s["sharpness"], s["brightness"], s["color_intensity"], s["lens_cleanliness"]]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation='h'))
    fig.update_layout(xaxis=dict(range=[0,1]), height=300, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Quality Radar")
    st.plotly_chart(plot_radar(labels, vals), use_container_width=True)

with c3:
    st.subheader("Lens Cleanliness (Pie)")
    st.plotly_chart(lens_pie(s["lens_cleanliness"]), use_container_width=True)
    st.caption("Pie approximates clean vs. dirty proportion inferred from static dark / edge P10 / dark channel.")

# Third row: hist + raw metrics
st.markdown("---")
r1, r2 = st.columns([3,2])

with r1:
    st.subheader("Luminance Histogram (Y)")
    raw = res["raw"]
    hist = np.array(raw["hist"] if "hist" in raw else [])
    if hist.size == 0:
        gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    x = np.arange(256)
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=x, y=hist, mode='lines'))
    fig_h.add_vline(x=raw.get("p1", 1), line_dash='dash')
    fig_h.add_vline(x=raw.get("p99", 255), line_dash='dash')
    fig_h.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Intensity", yaxis_title="Count")
    st.plotly_chart(fig_h, use_container_width=True)

with r2:
    st.subheader("Raw Metrics")
    show = {
        "VoL": f"{raw['vol']:.1f}",
        "Tenengrad": f"{raw['tenengrad']:.1f}",
        "VoL (P10)": f"{raw['vol_p10']:.1f}",
        "Edge density": f"{raw['edge_density']:.3f}",
        "Edge density (P10)": f"{raw['edge_density_p10']:.3f}",
        "Y_mean": f"{raw['meanY']:.1f}",
        "Dyn range": f"{int(raw['dyn_range'])}",
        "Clipping sum": f"{raw['clip_sum']:.3f}",
        "Colorfulness": f"{raw['colorfulness']:.1f}",
        "HSV-S mean": f"{raw['hsv_s']:.1f}",
        "Dark channel mean": f"{raw['dark_channel_mean']:.1f}",
        "Static dark ratio": f"{raw['static_dark_ratio']*100:.2f}%",
    }
    st.table(show)

st.markdown("---")
st.caption("Tip: Sidebar'dan 'Scene mode' ve 'Resize short side' ayarlarını değiştirerek hem gece/gündüz normalizasyonunu hem de hesaplama ölçeğini kontrol edebilirsin.")
