"""CamDoc Predictive Features Analysis - Streamlit Dashboard"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import CamQA, AdvancedQualityMetrics
from src.utils import ConfigLoader, PredictiveConfig
from src.predictive.features import build_features, TemporalState

# Initialize session state
if 'temporal_state' not in st.session_state:
    st.session_state.temporal_state = TemporalState()
if 'feature_history' not in st.session_state:
    st.session_state.feature_history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0


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


def plot_temporal_gauge(value, title="Temporal Quality"):
    """Create a gauge chart for temporal quality"""
    v = float(np.clip(value, 0, 1)) * 100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={'suffix': "%", 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#8b5cf6"},
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


def plot_predictive_radar(features):
    """Create radar chart for predictive features"""
    labels = ['Sharpness', 'Brightness', 'Color', 'Lens', 'Stability', 'Consistency']
    values = [
        features.get('camqa_sharpness', 0),
        features.get('camqa_brightness', 0),
        features.get('camqa_color_intensity', 0),
        features.get('camqa_lens_cleanliness', 0),
        max(0, 1 - features.get('jitter_px', 0) / 10),  # Inverse jitter
        max(0, 1 - features.get('meanY_flicker_std', 0) / 20)  # Inverse flicker
    ]
    
    theta = labels + [labels[0]]
    r = values + [values[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r, theta=theta,
        fill='toself',
        name='Predictive Metrics',
        line=dict(color='#8b5cf6', width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True)),
        showlegend=False,
        height=350,
        margin=dict(l=30, r=30, t=40, b=10),
        title="Predictive Quality Radar"
    )
    return fig


def load_configurations():
    """Load all configuration objects"""
    try:
        config = ConfigLoader()
        pred_config = PredictiveConfig()
        return config, pred_config
    except Exception as e:
        st.error(f"Error loading configurations: {e}")
        st.stop()


def create_temporal_charts(history: List[Dict]) -> go.Figure:
    """Create temporal analysis charts"""
    
    if len(history) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for temporal analysis", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Temporal Stability Metrics")
        return fig
    
    df = pd.DataFrame(history)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Jitter (px/frame)', 'Flicker Std', 'Sharpness', 'Brightness'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(x=df['frame_count'], y=df['jitter_px'], name='Jitter', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['frame_count'], y=df['meanY_flicker_std'], name='Flicker', line=dict(color='orange')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df['frame_count'], y=df['camqa_sharpness'], name='Sharpness', line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['frame_count'], y=df['camqa_brightness'], name='Brightness', line=dict(color='green')),
        row=2, col=2
    )
    
    fig.update_layout(height=500, title_text="Temporal Analysis Dashboard", showlegend=False)
    return fig


# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="CamDoc Predictive Analysis", layout="wide")
    st.title("🔮 CamQA — Predictive Features Dashboard")
    
    # Initialize analyzers
    config_loader, pred_config = load_configurations()
    
    with st.sidebar:
        st.header("Settings")
        
        # Source selection
        src_mode = st.selectbox("Source", ["Sample image", "Upload image", "Video analysis"])
        
        # CamQA settings
        st.subheader("CamQA Settings")
        short_side = st.slider("Resize short side", 480, 1080, 720, 20)
        ema = st.slider("EMA smoothing", 0.0, 0.9, 0.2, 0.05)
        force_mode = st.selectbox("Scene mode", ["auto", "day", "night"])
        
        # Predictive features toggle
        st.subheader("Predictive Features")
        use_predictive = st.checkbox("Enable Predictive Analysis", value=True)
        use_temporal = st.checkbox("Enable Temporal Analysis", value=False, disabled=src_mode != "Video analysis")
        
        # Advanced metrics toggle
        st.subheader("Advanced Metrics")
        use_advanced = st.checkbox("Enable BRISQUE/NIQE/TV", value=False)
        
        # Visualization options
        st.subheader("Visualizations")
        show_detailed = st.checkbox("Show Detailed Analysis", value=True)
        show_predictive_radar = st.checkbox("Show Predictive Radar", value=True)
        
        # Upload
        uploaded = None
        if src_mode == "Upload image":
            uploaded = st.file_uploader(
                "Upload a frame (JPEG/PNG/BMP)",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]
            )
        elif src_mode == "Video analysis":
            uploaded = st.file_uploader(
                "Upload a video file",
                type=["mp4", "avi", "mov", "mkv"]
            )
        
        # Clear temporal history
        if st.button("Clear Temporal History"):
            st.session_state.temporal_state = TemporalState()
            st.session_state.feature_history = []
            st.session_state.analysis_count = 0
            st.success("History cleared!")
        
        if use_predictive:
            st.caption("Predictive features: Temporal stability, jitter detection, flicker analysis")
        if use_advanced:
            st.caption("Advanced: BRISQUE, NIQE (PyIQA), Total Variation (PIQ)")
    
    # Sample image path
    sample_path = project_root / "test/images/img1.jpg"
    
    # Load image based on mode
    if src_mode == "Video analysis" and uploaded is not None:
        # Video analysis mode
        st.subheader("📹 Video Analysis with Temporal Features")
        
        # Save uploaded video temporarily
        temp_video_path = f"/tmp/uploaded_video_{int(time.time())}.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded.read())
        
        # Video processing controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_frames = st.number_input("Max frames to process", min_value=10, max_value=500, value=100)
        
        with col2:
            frame_skip = st.number_input("Frame skip interval", min_value=1, max_value=10, value=2)
        
        with col3:
            if st.button("Start Video Analysis"):
                # Process video
                cap = cv2.VideoCapture(temp_video_path)
                
                if not cap.isOpened():
                    st.error("Could not open video file")
                    return
                
                # Reset temporal state for new video
                st.session_state.temporal_state = TemporalState()
                st.session_state.feature_history = []
                st.session_state.analysis_count = 0
                
                progress_bar = st.progress(0)
                frame_count = 0
                processed_frames = 0
                
                # Get total frames for progress
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_to_process = min(max_frames, total_frames // frame_skip)
                
                while cap.isOpened() and processed_frames < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_skip == 0:
                        # Analyze frame with temporal features
                        try:
                            features, new_state = build_features(
                                frame,
                                prev_state=st.session_state.temporal_state,
                                use_advanced=use_advanced,
                                use_temporal=True,
                                camqa=CamQA(short_side=short_side, ema=ema, mode=force_mode)
                            )
                            
                            # Update temporal state
                            st.session_state.temporal_state = new_state
                            
                            # Add to history
                            st.session_state.feature_history.append({
                                'timestamp': time.time(),
                                'frame_count': st.session_state.analysis_count,
                                **features
                            })
                            
                            processed_frames += 1
                            st.session_state.analysis_count += 1
                            
                            # Update progress
                            progress = processed_frames / frames_to_process
                            progress_bar.progress(progress)
                            
                        except Exception as e:
                            st.error(f"Error processing frame {frame_count}: {e}")
                            break
                    
                    frame_count += 1
                
                cap.release()
                
                # Display temporal analysis results
                if st.session_state.feature_history:
                    st.success(f"Processed {len(st.session_state.feature_history)} frames")
                    
                    # Create temporal charts
                    fig_temporal = create_temporal_charts(st.session_state.feature_history)
                    st.plotly_chart(fig_temporal, use_container_width=True)
                    
                    # Summary statistics
                    df = pd.DataFrame(st.session_state.feature_history)
                    
                    st.subheader("📊 Temporal Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Jitter", f"{df['jitter_px'].mean():.2f} px")
                        st.metric("Max Jitter", f"{df['jitter_px'].max():.2f} px")
                    
                    with col2:
                        st.metric("Avg Flicker", f"{df['meanY_flicker_std'].mean():.2f}")
                        st.metric("Max Flicker", f"{df['meanY_flicker_std'].max():.2f}")
                    
                    with col3:
                        st.metric("Avg Sharpness", f"{df['camqa_sharpness'].mean():.3f}")
                        st.metric("Min Sharpness", f"{df['camqa_sharpness'].min():.3f}")
                    
                    with col4:
                        stability_scores = [pred_config.calculate_temporal_quality_score(
                            row['jitter_px'], row['meanY_flicker_std'], row['frame_diff_var']
                        ) for _, row in df.iterrows()]
                        st.metric("Avg Stability", f"{np.mean(stability_scores):.3f}")
                        st.metric("Min Stability", f"{np.min(stability_scores):.3f}")
        
        # Clean up temp file
        try:
            Path(temp_video_path).unlink(missing_ok=True)
        except:
            pass
        
        return  # Exit early for video mode
    
    # Image analysis mode
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
    
    # -------- Analysis --------
    qa = CamQA(short_side=short_side, ema=ema, mode=force_mode)
    
    if use_predictive:
        # Use predictive features
        features, _ = build_features(
            bgr,
            prev_state=None,
            use_advanced=use_advanced,
            use_temporal=False,  # Single image, no temporal
            camqa=qa
        )
        
        # Get CamQA results from features
        res = {
            'scores': {
                'sharpness': features['camqa_sharpness'],
                'brightness': features['camqa_brightness'],
                'color_intensity': features['camqa_color_intensity'],
                'lens_cleanliness': features['camqa_lens_cleanliness']
            },
            'raw': {
                'vol': features['vol'],
                'tenengrad': features['tenengrad'],
                'edge_density': features['edge_density'],
                'meanY': features['meanY'],
                'dyn_range': features['dyn_range'],
                'colorfulness': features['colorfulness'],
                'hsv_s': features['hsv_s'],
                'dark_channel_mean': features['dark_channel_mean'],
                'static_dark_ratio': features['static_dark_ratio'],
                'clip_sum': features['clip_sum']
            },
            'mode': 'day' if features['scene_day'] else 'night' if features['scene_night'] else 'twilight'
        }
        
        # Get recommendations
        recommendations = pred_config.get_analysis_recommendations(features)
        temporal_quality = recommendations['temporal_quality']
        stability_class = recommendations['stability_class']
        
    else:
        # Standard CamQA analysis
        res = qa.analyze(bgr)
        features = None
        recommendations = None
        temporal_quality = 1.0  # Default for single image
        stability_class = "excellent"
    
    raw = res["raw"]
    
    # Convert to RGB for display
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # ---- Layout: Image + Gauges ----
    col_img, col_gauge = st.columns([3, 2], vertical_alignment="top")
    
    with col_img:
        st.subheader("Source Image")
        st.image(rgb, use_column_width=True)
    
    with col_gauge:
        # Scores
        s = res["scores"]
        camqa_only = (
            0.30 * s["sharpness"] +
            0.30 * s["brightness"] +
            0.20 * s["color_intensity"] +
            0.20 * s["lens_cleanliness"]
        )
        
        # Calculate real overall quality (will be updated if advanced metrics enabled)
        if use_advanced and features and 'brisque_q' in features and not np.isnan(features['brisque_q']):
            # Real overall quality with advanced metrics
            overall = (
                # CamQA metrics - 35% weight
                0.105 * s["sharpness"] +
                0.105 * s["brightness"] +
                0.07 * s["color_intensity"] +
                0.07 * s["lens_cleanliness"] +
                # Advanced metrics - 65% weight
                0.27 * features['brisque_q'] +
                0.27 * features['niqe_q'] +
                0.11 * features['tv_q']
            )
        else:
            # CamQA only
            overall = camqa_only
        
        # Main quality gauge (real overall quality)
        st.plotly_chart(
            plot_gauge(overall, title=f"Overall Quality ({res['mode']})"),
            use_container_width=True
        )
        
        # Predictive temporal gauge (only for video analysis)
        if use_predictive and src_mode == "Video analysis":
            st.plotly_chart(
                plot_temporal_gauge(temporal_quality, title="Temporal Quality"),
                use_container_width=True
            )
        elif use_predictive:
            st.info("💡 **Temporal Quality** sadece video analizi için geçerlidir")
        
        # Individual scores
        st.metric("Sharpness", f"{s['sharpness']:.3f}")
        st.metric("Brightness", f"{s['brightness']:.3f}")
        st.metric("Color Intensity", f"{s['color_intensity']:.3f}")
        st.metric("Lens Cleanliness", f"{s['lens_cleanliness']:.3f}")
        
        if use_predictive:
            st.metric("Stability Class", stability_class.upper())
    
    # -------- Advanced Metrics (Optional) --------
    if use_advanced and features and 'brisque_q' in features and not np.isnan(features['brisque_q']):
        st.markdown("---")
        st.subheader("🔬 Advanced Quality Metrics")
        
        col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
        
        with col_adv1:
            st.plotly_chart(
                plot_gauge(features['brisque_q'], "BRISQUE"),
                use_container_width=True
            )
            st.caption(f"Raw: {features['brisque_raw']:.2f}")
        
        with col_adv2:
            st.plotly_chart(
                plot_gauge(features['niqe_q'], "NIQE"),
                use_container_width=True
            )
            st.caption(f"Raw: {features['niqe_raw']:.2f}")
        
        with col_adv3:
            st.plotly_chart(
                plot_gauge(features['tv_q'], "Total Variation"),
                use_container_width=True
            )
            st.caption(f"Raw: {features['tv_raw']:.6f}")
        
        with col_adv4:
            st.markdown("#### Quality Breakdown")
            
            # Show contribution breakdown
            camqa_contribution = 0.105 * s["sharpness"] + 0.105 * s["brightness"] + 0.07 * s["color_intensity"] + 0.07 * s["lens_cleanliness"]
            adv_contribution = 0.27 * features['brisque_q'] + 0.27 * features['niqe_q'] + 0.11 * features['tv_q']
            
            st.metric("CamQA Contribution", f"{camqa_contribution:.3f}", help="35% weight in overall score")
            st.metric("Advanced Contribution", f"{adv_contribution:.3f}", help="65% weight in overall score")
            st.metric("Combined Total", f"{overall:.3f}", help="Final overall quality score")
    
    # Quality classification
    quality_class = config_loader.classify_quality(overall)
    needs_maint = config_loader.needs_maintenance(overall)
    disable_analytics = config_loader.should_disable_analytics(overall)
    
    # -------- Score Breakdown --------
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 2, 2])
    
    with c1:
        st.subheader("Score Breakdown")
        labels = ["Sharpness", "Brightness", "Color Intensity", "Lens Cleanliness"]
        vals = [s["sharpness"], s["brightness"], s["color_intensity"], s["lens_cleanliness"]]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation='h'))
        fig.update_layout(xaxis=dict(range=[0, 1]), height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        if use_predictive and show_predictive_radar and features:
            st.subheader("Predictive Radar")
            st.plotly_chart(plot_predictive_radar(features), use_container_width=True)
        else:
            st.subheader("Quality Radar")
            st.plotly_chart(plot_radar(labels, vals), use_container_width=True)
    
    with c3:
        st.subheader("Quality Status")
        st.metric("Quality Class", quality_class.upper())
        if needs_maint:
            st.warning("⚠️ Maintenance Recommended")
        if disable_analytics:
            st.error("🚫 Analytics Should Be Disabled")
        
        if use_predictive and recommendations:
            if recommendations['alerts']:
                st.error("⚠️ Quality Alerts")
                for alert in recommendations['alerts']:
                    st.caption(f"• {alert}")
    
    # -------- Detailed Analysis --------
    if show_detailed:
        st.markdown("---")
        st.subheader("📊 Detailed Metrics")
        
        if use_predictive and features:
            # Predictive features table
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.markdown("#### Predictive Features")
                pred_metrics = {
                    "Feature": ["Jitter (px/frame)", "Frame Diff Variance", "Flicker Std", "Extract Time (ms)"],
                    "Value": [
                        f"{features['jitter_px']:.2f}",
                        f"{features['frame_diff_var']:.1f}",
                        f"{features['meanY_flicker_std']:.2f}",
                        f"{features['feature_extract_ms']:.1f}"
                    ],
                    "Status": [
                        pred_config.classify_jitter(features['jitter_px']).upper(),
                        pred_config.classify_frame_diff(features['frame_diff_var']).upper(),
                        pred_config.classify_flicker(features['meanY_flicker_std']).upper(),
                        "Good" if features['feature_extract_ms'] < 200 else "Slow"
                    ]
                }
                st.table(pred_metrics)
            
            with col_pred2:
                st.markdown("#### Scene Classification")
                scene_data = {
                    "Scene Type": ["Day", "Night", "Twilight"],
                    "Detected": [
                        "✓" if features['scene_day'] > 0 else "✗",
                        "✓" if features['scene_night'] > 0 else "✗",
                        "✓" if features['scene_twilight'] > 0 else "✗"
                    ]
                }
                st.table(scene_data)
                
                if recommendations and recommendations['suggestions']:
                    st.markdown("#### Suggestions")
                    for suggestion in recommendations['suggestions']:
                        st.info(f"💡 {suggestion}")
        
        # Raw metrics table (always show)
        st.markdown("#### Raw Metrics")
        raw_show = {
            "VoL": f"{raw['vol']:.1f}",
            "Tenengrad": f"{raw['tenengrad']:.1f}",
            "Edge density": f"{raw['edge_density']:.3f}",
            "Y_mean": f"{raw['meanY']:.1f}",
            "Dyn range": f"{int(raw['dyn_range'])}",
            "Colorfulness": f"{raw['colorfulness']:.1f}",
            "HSV-S mean": f"{raw['hsv_s']:.1f}",
            "Dark channel mean": f"{raw['dark_channel_mean']:.1f}",
            "Static dark ratio": f"{raw['static_dark_ratio']*100:.2f}%",
        }
        st.table(raw_show)
    
    # -------- Footer --------
    st.markdown("---")
    st.caption(
        "🔮 **Predictive Features:** Temporal stability analysis, jitter detection, and quality forecasting. "
        "💡 **Tip:** Enable temporal analysis for video streams to get stability insights."
    )


if __name__ == "__main__":
    main()
