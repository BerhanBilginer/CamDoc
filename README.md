# 📷 CamDoc - Camera Quality Assessment & Health Check

A comprehensive camera quality assessment system that combines fast real-time metrics with advanced perceptual quality models.

**Version**: 1.1.0 | **Last Updated**: 2025-11-06

---

## 📑 İçindekiler

- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Usage Options](#usage-options)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
  - [CLI Interface](#1-command-line-interface)
  - [Streamlit Dashboard](#2-streamlit-dashboard)
  - [Python API](#3-python-api)
- [Metrics Explained](#-metrics-explained)
- [Dashboard Features](#-dashboard-features)
- [Configuration](#️-configuration)
- [Quality Classification](#-quality-classification)
- [Common Use Cases](#-common-use-cases)
- [Migration Guide](#-migration-guide)
- [Troubleshooting](#-troubleshooting)
- [Changelog](#-changelog)
- [Development](#️-development)
- [Roadmap](#-roadmap)
- [Support](#-support)

---

## ✨ Features

### **Fast CamQA Analysis**
Real-time assessment of:
- 🔍 **Sharpness**: Variance of Laplacian, Tenengrad
- 💡 **Brightness**: Luminance, dynamic range, clipping
- 🎨 **Color Intensity**: Colorfulness, saturation
- 🧹 **Lens Cleanliness**: Static detection, dark channel

### **Advanced Metrics** (Optional)
- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator
- **NIQE**: Natural Image Quality Evaluator
- **Total Variation**: Smoothness measure

### **Quality Classification & Health Policies**
- Automatic quality grading (high/medium/low/unusable)
- Maintenance alerts
- Analytics disable recommendations

### **Interactive Dashboard**
- Streamlit-based web interface
- Real-time visualization
- 4 gauge charts, comparison charts, radar plots
- RGB & luminance histograms
- Image visualizations (edge map, gradient, saturation, dark channel)
- Detailed/Compact view modes

## 📁 Project Structure

```
CamDoc/
├── src/                        # Source modules
│   ├── core/                   # Core functionality
│   │   ├── camqa.py           # Fast CamQA analyzer
│   │   └── advanced_metrics.py # BRISQUE/NIQE/TV metrics
│   └── utils/                  # Utilities
│       └── config_loader.py   # Configuration management
├── apps/                       # Applications
│   └── streamlit_app.py       # Streamlit dashboard
├── config/                     # Configuration files
│   └── quality_thresholds.json # Quality thresholds
├── test/                       # Test images
│   └── images/
├── main.py                     # CLI interface
├── requirements.txt            # Dependencies
└── README.md                   # This file

# Legacy files (for reference)
├── camqa.py                    # Original CamQA (now in src/core/)
├── brisque_niqe.py            # Original metrics demo
└── camqa_streamlit_app.py     # Original streamlit app
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd CamDoc

# Install dependencies
pip install -r requirements.txt

# For advanced metrics (optional)
pip install pyiqa torch
```

### Usage

#### 1. Command Line Interface

```bash
# Basic usage
python main.py

# Analyze specific image
python main.py --image path/to/image.jpg

# Enable advanced metrics
python main.py --image path/to/image.jpg --advanced

# Force scene mode (day/night)
python main.py --image path/to/image.jpg --mode night
```

#### 2. Streamlit Dashboard

```bash
streamlit run apps/streamlit_app.py
```

Then open your browser at `http://localhost:8501`

#### 3. Python API

```python
from src.core import CamQA, AdvancedQualityMetrics
from src.utils import ConfigLoader
import cv2

# Load configuration
config = ConfigLoader()

# Basic CamQA analysis
qa = CamQA(short_side=720, ema=0.2, mode='auto')
frame = cv2.imread("test/images/img1.jpg")
result = qa.analyze(frame)

print(result['mode'])  # 'day', 'night', or 'twilight'
print(result['scores'])  # sharpness, brightness, color_intensity, lens_cleanliness

# Quality classification
overall = (0.3 * result['scores']['sharpness'] + 
           0.3 * result['scores']['brightness'] +
           0.2 * result['scores']['color_intensity'] +
           0.2 * result['scores']['lens_cleanliness'])

quality_class = config.classify_quality(overall)
print(f"Quality: {quality_class}")

# Advanced metrics (optional)
adv = AdvancedQualityMetrics(device='cpu')
adv_result = adv.analyze(frame)
print(f"Advanced quality: {adv_result['overall_quality']:.3f}")
```

## 📊 Metrics Explained

### CamQA Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Sharpness** | Edge clarity using Laplacian variance & Tenengrad | 0-1 |
| **Brightness** | Optimal luminance range with clipping penalty | 0-1 |
| **Color Intensity** | Colorfulness & saturation measures | 0-1 |
| **Lens Cleanliness** | Static dark regions & edge degradation | 0-1 |

### Advanced Metrics

| Metric | Description | Type |
|--------|-------------|------|
| **BRISQUE** | No-reference quality based on natural scene statistics | Lower = Better |
| **NIQE** | Quality assessment using multivariate Gaussian model | Lower = Better |
| **Total Variation** | Image smoothness (high TV = noisy/oversharpened) | Lower = Better |

## ⚙️ Configuration

Edit `config/quality_thresholds.json` to customize:

- Day/night thresholds for each metric
- Quality classification boundaries
- Maintenance alert thresholds
- Analytics disable thresholds

## 📈 Quality Classification

| Class | Score Range | Action |
|-------|-------------|--------|
| **High** | ≥ 0.80 | Normal operation |
| **Medium** | 0.60 - 0.80 | Monitor closely |
| **Low** | 0.40 - 0.60 | Maintenance recommended |
| **Unusable** | < 0.40 | Disable analytics |

## 🛠️ Development

### Running Tests

```bash
# Test on sample images
python main.py --image test/images/img1.jpg --advanced

# Test Streamlit app
streamlit run apps/streamlit_app.py
```

### Adding Custom Metrics

1. Create new metric class in `src/core/`
2. Update `src/core/__init__.py` to export it
3. Integrate in `main.py` and `apps/streamlit_app.py`

## 📝 Dependencies

- **Core**: OpenCV, NumPy
- **Dashboard**: Streamlit, Plotly
- **Advanced Metrics**: PyIQA, PyTorch, PIL
- **Optional**: PIQ (for faster Total Variation)

## 🔄 Migration from Legacy Code

The project has been reorganized into a modular structure:

- `camqa.py` → `src/core/camqa.py`
- `brisque_niqe.py` → `src/core/advanced_metrics.py`
- `camqa_streamlit_app.py` → `apps/streamlit_app.py`

Old files are kept for reference but should not be used in new code.

## 📄 License

[Add your license here]

## 👥 Contributing

[Add contribution guidelines here]

## 📞 Support

[Add support information here]
