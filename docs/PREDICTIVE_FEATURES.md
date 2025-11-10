# CamDoc Predictive Features System

## Overview

The CamDoc Predictive Features system extends the core camera quality assessment capabilities with advanced temporal analysis and predictive modeling. This system provides comprehensive feature extraction, stability analysis, and quality prediction for both single images and video streams.

## Features

### Core Capabilities

1. **Comprehensive Feature Extraction**
   - CamQA quality scores (sharpness, brightness, color intensity, lens cleanliness)
   - Raw image metrics (variance of Laplacian, Tenengrad, edge density, etc.)
   - Scene classification (day/night/twilight)
   - Advanced quality metrics (BRISQUE, NIQE, Total Variation) - optional

2. **Temporal Analysis**
   - Optical flow-based jitter detection
   - Frame-to-frame difference variance
   - Luminance flicker analysis
   - Running statistics with configurable buffer sizes

3. **Stability Classification**
   - Real-time stability assessment
   - Configurable thresholds for different stability levels
   - Comprehensive recommendations system

4. **Interactive Analysis Interface**
   - Streamlit-based web application
   - Real-time visualization
   - Video stream processing
   - Temporal trend analysis

## System Architecture

```
CamDoc Predictive Features
├── src/predictive/
│   └── features.py              # Core feature extraction
├── src/utils/
│   └── predictive_config.py     # Configuration management
├── config/
│   └── predictive_config.json   # System configuration
├── apps/
│   └── predictive_analysis.py   # Streamlit interface
└── run_predictive_app.py        # Application launcher
```

## Configuration

### Predictive Configuration (`config/predictive_config.json`)

The system uses a comprehensive configuration file that defines:

- **Temporal Settings**: Buffer sizes, optical flow parameters
- **Feature Thresholds**: Classification boundaries for different metrics
- **Quality Weights**: Relative importance of different quality components
- **Analysis Settings**: Default parameters and timeouts
- **Stability Classification**: Thresholds for stability levels

### Key Configuration Sections

```json
{
  "predictive_features": {
    "temporal_settings": {
      "meanY_buffer_size": 30,
      "optical_flow_max_corners": 200,
      // ... other temporal parameters
    },
    "feature_thresholds": {
      "jitter_px": {
        "low": 0.5, "medium": 2.0, "high": 5.0, "critical": 10.0
      },
      // ... other thresholds
    },
    "stability_classification": {
      "excellent": { "jitter_max": 0.5, "flicker_std_max": 2.0 },
      // ... other stability levels
    }
  }
}
```

## Usage

### 1. Command Line Interface

The main CLI has been extended to support predictive features:

```bash
# Basic analysis with predictive features
python main.py --image my_image.jpg --predictive

# Full analysis with all features
python main.py --image my_image.jpg --advanced --predictive --mode auto

# Available options
python main.py --help
```

### 2. Streamlit Web Interface

Launch the interactive web application:

```bash
# Start the Streamlit app
python run_predictive_app.py

# Or directly with streamlit
streamlit run apps/predictive_analysis.py
```

The web interface provides:

- **Single Image Analysis**: Upload and analyze individual images
- **Video Stream Analysis**: Process video files with temporal analysis
- **Real-time Visualization**: Interactive charts and gauges
- **Detailed Reports**: Comprehensive feature tables and recommendations

### 3. Programmatic API

```python
from src.predictive.features import build_features, TemporalState
from src.utils import PredictiveConfig
import cv2

# Load configuration
config = PredictiveConfig()

# Initialize temporal state for video analysis
temporal_state = TemporalState()

# Analyze a frame
frame = cv2.imread("image.jpg")
features, temporal_state = build_features(
    frame,
    prev_state=temporal_state,
    use_advanced=True,
    use_temporal=True
)

# Get recommendations
recommendations = config.get_analysis_recommendations(features)
```

## Feature Descriptions

### CamQA Features
- **Sharpness**: Edge-based sharpness assessment (0-1, higher is better)
- **Brightness**: Luminance-based brightness evaluation (0-1, optimal around 0.7-0.8)
- **Color Intensity**: Color saturation and vibrancy (0-1, higher is better)
- **Lens Cleanliness**: Dirt and obstruction detection (0-1, higher is better)

### Raw Metrics
- **Variance of Laplacian (vol)**: Edge-based sharpness measure
- **Tenengrad**: Gradient-based focus measure
- **Edge Density**: Proportion of edge pixels
- **Mean Luminance (meanY)**: Average brightness value
- **Dynamic Range**: Spread of intensity values
- **Colorfulness**: Perceptual color richness

### Temporal Features
- **Jitter (px/frame)**: Camera shake and vibration (lower is better)
- **Frame Diff Variance**: Frame-to-frame stability (lower is better)
- **Flicker Std**: Luminance stability over time (lower is better)
- **Running Mean Y**: Temporal average of luminance

### Advanced Metrics (Optional)
- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator
- **NIQE**: Natural Image Quality Evaluator
- **Total Variation**: Image smoothness measure

## Stability Classification

The system classifies temporal stability into four levels:

1. **Excellent**: Minimal jitter, stable luminance, consistent quality
2. **Good**: Low jitter, acceptable stability
3. **Fair**: Moderate instability, may affect some applications
4. **Poor**: High instability, requires immediate attention

## Recommendations System

The system provides automated recommendations based on analysis results:

### Alerts
- High camera jitter detection
- Significant luminance flicker
- High frame-to-frame variation
- Poor overall stability

### Suggestions
- Camera mount stabilization
- Lighting condition optimization
- Camera settings adjustment
- Environmental condition verification

## Performance Considerations

### Optimization Tips

1. **Buffer Size**: Adjust `meanY_buffer_size` based on frame rate
2. **Optical Flow**: Reduce `max_corners` for faster processing
3. **Advanced Metrics**: Disable if real-time performance is critical
4. **Temporal Analysis**: Skip for single image analysis

### Typical Performance

- **Single Image**: 50-200ms (depending on resolution and features)
- **Video Stream**: 30-60 FPS (720p, basic features)
- **Advanced Metrics**: +100-300ms per frame

## Integration Examples

### Video Stream Processing

```python
import cv2
from src.predictive.features import build_features, TemporalState

# Initialize
temporal_state = TemporalState()
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Analyze frame
    features, temporal_state = build_features(
        frame, 
        prev_state=temporal_state,
        use_temporal=True
    )
    
    # Process results
    jitter = features['jitter_px']
    stability = features['meanY_flicker_std']
    
    print(f"Jitter: {jitter:.2f} px, Stability: {stability:.2f}")

cap.release()
```

### Real-time Quality Monitoring

```python
from src.utils import PredictiveConfig

config = PredictiveConfig()

# Monitor quality in real-time
def monitor_quality(features):
    recommendations = config.get_analysis_recommendations(features)
    
    if recommendations['alerts']:
        print("⚠️ Quality alerts:")
        for alert in recommendations['alerts']:
            print(f"  - {alert}")
    
    return recommendations['temporal_quality']
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Advanced Metrics Not Available**: Install PyIQA and PyTorch
   ```bash
   pip install pyiqa torch
   ```

3. **Configuration Errors**: Verify JSON syntax in config files

4. **Performance Issues**: 
   - Reduce image resolution
   - Disable advanced metrics
   - Adjust buffer sizes

### Debug Mode

Enable detailed logging by setting environment variable:
```bash
export CAMDOC_DEBUG=1
python main.py --predictive --image test.jpg
```

## Future Enhancements

### Planned Features

1. **Machine Learning Models**: Predictive quality forecasting
2. **Anomaly Detection**: Automated quality degradation detection
3. **Real-time Alerts**: Push notifications for quality issues
4. **Historical Analysis**: Long-term quality trend analysis
5. **Multi-camera Support**: Synchronized analysis across multiple streams

### API Extensions

1. **REST API**: HTTP endpoints for remote analysis
2. **WebSocket Support**: Real-time streaming analysis
3. **Cloud Integration**: Scalable processing in cloud environments
4. **Mobile SDK**: Quality assessment for mobile applications

## Contributing

To contribute to the predictive features system:

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update configuration schemas as needed
4. Document new features in this README
5. Ensure backward compatibility

## License

This predictive features system is part of the CamDoc project and follows the same licensing terms.
