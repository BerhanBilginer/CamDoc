"""Streamlit Dashboard for Camera Quality Assessment"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from src.core import CamQA, AdvancedQualityMetrics
from src.utils import ConfigLoader


# -------------------- Plot Helpers --------------------
def plot_gauge(value, title="Overall Health"):
    """Create a gauge chart for quality visualization"""
    v = float(np.clip(value, 0, 1)) * 100.0
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
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title}
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_radar(labels, values, title="Quality Radar"):
    """Create a radar chart for multi-dimensional quality scores"""
    theta = labels + [labels[0]]
    r = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', name='Scores'))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True)),
        showlegend=False, height=350, margin=dict(l=30, r=30, t=40, b=10),
        title=title
    )
    return fig


def lens_pie(lens_score):
    """Create a pie chart for lens cleanliness"""
    clean = float(np.clip(lens_score, 0, 1))
    dirty = 1.0 - clean
    fig = go.Figure(data=[go.Pie(
        labels=['Clean', 'Dirty'],
        values=[clean, dirty],
        hole=0.45
    )])
    fig.update_traces(
        textinfo='label+percent',
        pull=[0.05, 0],
        marker=dict(colors=['#74c69d', '#ef476f'])
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_color_histogram(bgr):
    """Plot RGB color histogram"""
    colors = ['blue', 'green', 'red']
    fig = go.Figure()
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([bgr], [i], None, [256], [0, 256]).ravel()
        fig.add_trace(go.Scatter(
            x=np.arange(256), y=hist,
            mode='lines', name=color.upper(),
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Intensity",
        yaxis_title="Pixel Count",
        title="RGB Histogram",
        showlegend=True
    )
    return fig


def plot_advanced_radar(adv_res):
    """Create radar chart for advanced metrics"""
    labels = ['BRISQUE', 'NIQE', 'TV']
    values = [
        adv_res['normalized']['brisque_quality'],
        adv_res['normalized']['niqe_quality'],
        adv_res['normalized']['tv_quality']
    ]
    
    theta = labels + [labels[0]]
    r = values + [values[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r, theta=theta,
        fill='toself',
        name='Advanced Metrics',
        line=dict(color='#8b5cf6', width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True)),
        showlegend=False,
        height=300,
        margin=dict(l=30, r=30, t=40, b=10),
        title="Advanced Quality Radar"
    )
    return fig


def plot_comparison_bar(camqa_score, adv_score):
    """Compare CamQA vs Advanced metrics"""
    fig = go.Figure(data=[
        go.Bar(name='CamQA', x=['Quality'], y=[camqa_score], marker_color='#2E8B57'),
        go.Bar(name='Advanced', x=['Quality'], y=[adv_score], marker_color='#8b5cf6')
    ])
    
    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        title="CamQA vs Advanced Quality",
        showlegend=True
    )
    return fig


def visualize_edge_map(gray):
    """Visualize edge detection map"""
    edges = cv2.Canny(gray, 80, 160)
    return edges


def visualize_gradient_magnitude(gray):
    """Visualize gradient magnitude"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
    return mag


def visualize_saturation(bgr):
    """Visualize saturation channel"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]


def visualize_dark_channel(bgr, k=15):
    """Visualize dark channel"""
    m = np.min(bgr, axis=2)
    k = k if k % 2 == 1 else k + 1
    dc = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return dc


# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="CamQA Dashboard", layout="wide")
    st.title("📷 CamQA — Camera Health Dashboard")
    
    # Initialize analyzers
    config_loader = ConfigLoader()
    
    with st.sidebar:
        st.header("Settings")
        
        # Source selection
        src_mode = st.selectbox("Source", ["Sample image", "Upload image"])
        
        # CamQA settings
        st.subheader("CamQA Settings")
        short_side = st.slider("Resize short side", 480, 1080, 720, 20)
        ema = st.slider("EMA smoothing", 0.0, 0.9, 0.2, 0.05)
        force_mode = st.selectbox("Scene mode", ["auto", "day", "night"])
        
        # Advanced metrics toggle
        st.subheader("Advanced Metrics")
        use_advanced = st.checkbox("Enable BRISQUE/NIQE/TV", value=False)
        
        # Visualization options
        st.subheader("Visualizations")
        show_detailed = st.checkbox("Show Detailed Analysis", value=True)
        show_visualizations = st.checkbox("Show Image Visualizations", value=False)
        
        # Upload
        uploaded = None
        if src_mode == "Upload image":
            uploaded = st.file_uploader(
                "Upload a frame (JPEG/PNG/BMP)",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]
            )
        
        if use_advanced:
            st.caption("Overall Quality = CamQA (56%) + Advanced metrics (44%)")
            st.caption("Advanced: BRISQUE, NIQE (PyIQA), Total Variation (PIQ)")
        else:
            st.caption("Overall Quality = CamQA metrics only (Sharpness, Brightness, Color, Lens)")
    
    # Sample image path
    sample_path = project_root / "test/images/img1.jpg"
    
    # Load image
    if src_mode == "Upload image" and uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        if sample_path.exists():
            bgr = cv2.imread(str(sample_path))
        else:
            st.error(f"Sample image not found at {sample_path}. Please upload an image.")
            st.stop()
    
    if bgr is None:
        st.error("Could not load image. Please upload a valid image.")
        st.stop()
    
    # -------- CamQA Analysis --------
    qa = CamQA(short_side=short_side, ema=ema, mode=force_mode)
    res = qa.analyze(bgr)
    raw = res["raw"]  # Extract raw metrics for use throughout
    
    # Convert to RGB for display
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # ---- Layout: Image + Gauge ----
    col_img, col_gauge = st.columns([3, 2], vertical_alignment="top")
    
    with col_img:
        st.subheader("Source Image")
        st.image(rgb, width='stretch')
    
    with col_gauge:
        # Scores
        s = res["scores"]
        camqa_only = (
            0.30 * s["sharpness"] +
            0.30 * s["brightness"] +
            0.20 * s["color_intensity"] +
            0.20 * s["lens_cleanliness"]
        )
        overall = camqa_only  # Will be updated if advanced metrics enabled
        
        # Placeholder for gauge and status (will be rendered after advanced metrics if enabled)
        gauge_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Individual scores
        st.metric("Sharpness", f"{s['sharpness']:.3f}")
        st.metric("Brightness", f"{s['brightness']:.3f}")
        st.metric("Color Intensity", f"{s['color_intensity']:.3f}")
        st.metric("Lens Cleanliness", f"{s['lens_cleanliness']:.3f}")
    
    # -------- Advanced Metrics (Optional) --------
    if use_advanced:
        st.markdown("---")
        st.subheader("🔬 Advanced Quality Metrics (PyIQA + PIQ)")
        
        try:
            adv_metrics = AdvancedQualityMetrics(device='cpu')
            adv_res = adv_metrics.analyze(bgr)
            
            # Main metrics display with gauges
            col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
            
            with col_adv1:
                st.plotly_chart(
                    plot_gauge(adv_res['normalized']['brisque_quality'], "BRISQUE"),
                    width='stretch'
                )
                st.caption(f"Raw: {adv_res['raw']['brisque']:.2f}")
            
            with col_adv2:
                st.plotly_chart(
                    plot_gauge(adv_res['normalized']['niqe_quality'], "NIQE"),
                    width='stretch'
                )
                st.caption(f"Raw: {adv_res['raw']['niqe']:.2f}")
            
            with col_adv3:
                st.plotly_chart(
                    plot_gauge(adv_res['normalized']['tv_quality'], "Total Variation"),
                    width='stretch'
                )
                st.caption(f"Raw: {adv_res['raw']['total_variation']:.6f}")
            
            with col_adv4:
                st.markdown("#### Metric Weights")
                st.markdown("""
                **CamQA (56%)**
                - Sharpness: 16%
                - Brightness: 16%
                - Color: 12%
                - Lens: 12%
                
                **Advanced (44%)**
                - BRISQUE: 18%
                - NIQE: 18%
                - TV: 12%
                """)
            
            # Calculate Overall Quality (CamQA + Advanced combined)
            overall = (
                # CamQA metrics - 56% weight
                0.16 * s["sharpness"] +
                0.16 * s["brightness"] +
                0.12 * s["color_intensity"] +
                0.12 * s["lens_cleanliness"] +
                # Advanced metrics - 44% weight
                0.18 * adv_res['normalized']['brisque_quality'] +
                0.18 * adv_res['normalized']['niqe_quality'] +
                0.12 * adv_res['normalized']['tv_quality']
            )
            
            # Update quality class based on combined score
            quality_class = config_loader.classify_quality(overall)
            needs_maint = config_loader.needs_maintenance(overall)
            disable_analytics = config_loader.should_disable_analytics(overall)
            
            # Render the gauge with combined score
            with gauge_placeholder:
                st.plotly_chart(
                    plot_gauge(overall, title=f"Overall Quality ({res['mode']})"),
                    width='stretch'
                )
            
            # Render status indicators with updated quality class
            with status_placeholder:
                st.metric("Quality Class", quality_class.upper())
                if needs_maint:
                    st.warning("⚠️ Maintenance Recommended")
                if disable_analytics:
                    st.error("🚫 Analytics Should Be Disabled")
            
            # Show breakdown
            st.markdown("### 📊 Score Breakdown")
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                st.metric("CamQA Only", f"{camqa_only:.3f}", help="Fast metrics: Sharpness, Brightness, Color, Lens")
            with col_summary2:
                st.metric("Advanced Only", f"{adv_res['overall_quality']:.3f}", help="BRISQUE, NIQE, Total Variation")
            
            # Advanced metrics visualization
            st.markdown("### 📊 Advanced Metrics Radar")
            st.plotly_chart(
                plot_advanced_radar(adv_res),
                width='stretch'
            )
            
            # Detailed metrics table
            if show_detailed:
                st.markdown("### 📋 Advanced Metrics Details")
                
                adv_details = {
                    "Metric": ["BRISQUE", "NIQE", "Total Variation"],
                    "Raw Value": [
                        f"{adv_res['raw']['brisque']:.2f}",
                        f"{adv_res['raw']['niqe']:.2f}",
                        f"{adv_res['raw']['total_variation']:.6f}"
                    ],
                    "Quality Score": [
                        f"{adv_res['normalized']['brisque_quality']:.3f}",
                        f"{adv_res['normalized']['niqe_quality']:.3f}",
                        f"{adv_res['normalized']['tv_quality']:.3f}"
                    ],
                    "Interpretation": [
                        "Excellent quality" if adv_res['raw']['brisque'] < 25 else "Good quality" if adv_res['raw']['brisque'] < 40 else "Consider enhancement",
                        "Natural quality" if adv_res['raw']['niqe'] < 4 else "Acceptable" if adv_res['raw']['niqe'] < 6 else "Quality issues detected",
                        "Optimal balance" if 0.015 <= adv_res['raw']['total_variation'] <= 0.030 else "Too smooth" if adv_res['raw']['total_variation'] < 0.010 else "High variation/noise"
                    ]
                }
                
                st.table(adv_details)
                
                # Add weight information
                st.info(
                    f"**Weights used:** BRISQUE: {adv_metrics.weights['brisque']:.2f}, "
                    f"NIQE: {adv_metrics.weights['niqe']:.2f}, "
                    f"TV: {adv_metrics.weights['tv']:.2f}"
                )
            
        except Exception as e:
            st.error(f"Advanced metrics error: {e}")
            st.info("Make sure PyIQA and torch are installed: `pip install pyiqa torch`")
    else:
        # Render gauge with CamQA-only score when advanced metrics are disabled
        with gauge_placeholder:
            st.plotly_chart(
                plot_gauge(overall, title=f"Overall Quality ({res['mode']})"),
                width='stretch'
            )
        
        # Render status indicators with CamQA-only score
        quality_class = config_loader.classify_quality(overall)
        needs_maint = config_loader.needs_maintenance(overall)
        disable_analytics = config_loader.should_disable_analytics(overall)
        
        with status_placeholder:
            st.metric("Quality Class", quality_class.upper())
            if needs_maint:
                st.warning("⚠️ Maintenance Recommended")
            if disable_analytics:
                st.error("🚫 Analytics Should Be Disabled")
    
    # -------- Score Breakdown --------
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 2, 2])
    
    with c1:
        st.subheader("Score Breakdown")
        labels = ["Sharpness", "Brightness", "Color Intensity", "Lens Cleanliness"]
        vals = [s["sharpness"], s["brightness"], s["color_intensity"], s["lens_cleanliness"]]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation='h'))
        fig.update_layout(xaxis=dict(range=[0, 1]), height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    
    with c2:
        st.subheader("Quality Radar")
        st.plotly_chart(plot_radar(labels, vals), width='stretch')
    
    with c3:
        st.subheader("Lens Cleanliness")
        st.plotly_chart(lens_pie(s["lens_cleanliness"]), width='stretch')
        st.caption(
            "Pie approximates clean vs. dirty proportion inferred from "
            "static dark / edge P10 / dark channel."
        )
    
    # -------- Histogram + Raw Metrics --------
    st.markdown("---")
    
    if show_detailed:
        st.subheader("📈 Detailed Histograms & Statistics")
        
        # Compute histogram data
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        cdf = np.cumsum(hist) / (hist.sum() + 1e-9)
        p1 = int(np.searchsorted(cdf, 0.01))
        p99 = int(np.searchsorted(cdf, 0.99))
        
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            st.markdown("#### Luminance Histogram (Y)")
            x = np.arange(256)
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=x, y=hist, mode='lines', name='Histogram', fill='tozeroy'))
            fig_h.add_vline(x=p1, line_dash='dash', annotation_text='P1', line_color='red')
            fig_h.add_vline(x=p99, line_dash='dash', annotation_text='P99', line_color='red')
            fig_h.add_vline(x=raw['meanY'], line_dash='dot', annotation_text='Mean', line_color='green')
            fig_h.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Intensity",
                yaxis_title="Pixel Count"
            )
            st.plotly_chart(fig_h, width='stretch')
        
        with col_hist2:
            st.markdown("#### RGB Color Histogram")
            st.plotly_chart(plot_color_histogram(bgr), width='stretch')
    else:
        # Compact version
        col_hist1, col_hist2 = st.columns([3, 2])
        
        with col_hist1:
            st.subheader("Luminance Histogram (Y)")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            cdf = np.cumsum(hist) / (hist.sum() + 1e-9)
            p1 = int(np.searchsorted(cdf, 0.01))
            p99 = int(np.searchsorted(cdf, 0.99))
            
            x = np.arange(256)
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=x, y=hist, mode='lines', name='Histogram'))
            fig_h.add_vline(x=p1, line_dash='dash', annotation_text='P1')
            fig_h.add_vline(x=p99, line_dash='dash', annotation_text='P99')
            fig_h.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Intensity",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_h, width='stretch')
        
        with col_hist2:
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
    
    # Extended raw metrics in detailed mode
    if show_detailed:
        st.markdown("#### 📊 Complete Raw Metrics Table")
        raw_metrics_df = {
            "Metric Category": [
                "Sharpness", "Sharpness", "Sharpness", "Sharpness",
                "Brightness", "Brightness", "Brightness",
                "Color", "Color",
                "Lens Quality", "Lens Quality"
            ],
            "Metric Name": [
                "Variance of Laplacian", "Tenengrad", "VoL (P10)", "Edge Density",
                "Mean Luminance (Y)", "Dynamic Range", "Clipping Sum",
                "Colorfulness (H-S)", "HSV Saturation Mean",
                "Dark Channel Mean", "Static Dark Ratio"
            ],
            "Value": [
                f"{raw['vol']:.1f}", f"{raw['tenengrad']:.1f}", f"{raw['vol_p10']:.1f}", f"{raw['edge_density']:.3f}",
                f"{raw['meanY']:.1f}", f"{raw['dyn_range']}", f"{raw['clip_sum']:.3f}",
                f"{raw['colorfulness']:.1f}", f"{raw['hsv_s']:.1f}",
                f"{raw['dark_channel_mean']:.1f}", f"{raw['static_dark_ratio']*100:.2f}%"
            ],
            "Interpretation": [
                "Higher = sharper", "Higher = sharper", "Higher = uniform sharpness", "Higher = more edges",
                f"{'Good' if 110 <= raw['meanY'] <= 140 else 'Review needed'}",
                f"{'Good' if raw['dyn_range'] > 180 else 'Limited'}",
                f"{'Good' if raw['clip_sum'] < 0.15 else 'High clipping'}",
                f"{'Vivid' if raw['colorfulness'] > 50 else 'Muted'}",
                f"{'Saturated' if raw['hsv_s'] > 100 else 'Desaturated'}",
                f"{'Clear' if raw['dark_channel_mean'] < 80 else 'Hazy'}",
                f"{'Clean' if raw['static_dark_ratio'] < 0.05 else 'Check lens'}"
            ]
        }
        
        st.table(raw_metrics_df)
    
    # -------- Image Visualizations --------
    if show_visualizations:
        st.markdown("---")
        st.subheader("🔍 Image Analysis Visualizations")
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        viz_col1, viz_col2, viz_col3, viz_col4 = st.columns(4)
        
        with viz_col1:
            st.markdown("##### Edge Map (Canny)")
            edges = visualize_edge_map(gray)
            st.image(edges, width='stretch', clamp=True)
            st.caption(f"Edge Density: {raw['edge_density']:.3f}")
        
        with viz_col2:
            st.markdown("##### Gradient Magnitude")
            grad_mag = visualize_gradient_magnitude(gray)
            st.image(grad_mag, width='stretch', clamp=True)
            st.caption("Sobel gradient magnitude")
        
        with viz_col3:
            st.markdown("##### Saturation Map")
            sat_map = visualize_saturation(bgr)
            st.image(sat_map, width='stretch', clamp=True)
            st.caption(f"Mean Sat: {raw['hsv_s']:.1f}")
        
        with viz_col4:
            st.markdown("##### Dark Channel")
            dc_map = visualize_dark_channel(bgr)
            st.image(dc_map, width='stretch', clamp=True)
            st.caption(f"DC Mean: {raw['dark_channel_mean']:.1f}")
    
    # -------- Footer --------
    st.markdown("---")
    st.caption(
        "💡 **Tip:** Sidebar'dan 'Scene mode' ve 'Resize short side' ayarlarını "
        "değiştirerek hem gece/gündüz normalizasyonunu hem de hesaplama ölçeğini "
        "kontrol edebilirsin."
    )


if __name__ == "__main__":
    main()
