"""
CamDoc - Camera Quality Assessment Main CLI

This script provides command-line interface for:
- CamQA: Fast camera quality assessment (sharpness, brightness, color, lens cleanliness)
- Advanced Metrics: BRISQUE, NIQE, Total Variation (optional)
- Predictive Features: Temporal analysis, jitter detection, stability classification
- Config Loader: Quality thresholds and health policies

Usage:
    python main.py [--image PATH] [--advanced] [--predictive] [--mode auto|day|night]
"""

# Standard library imports
import sys
import argparse
from pathlib import Path

# Third-party imports
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from src.core import CamQA, AdvancedQualityMetrics
from src.utils import ConfigLoader, PredictiveConfig
from src.predictive.features import build_features, TemporalState


def print_separator(title="", char="=", width=70):
    """Print a formatted separator line"""
    if title:
        print(f"\n{char * width}")
        print(f"{title.center(width)}")
        print(f"{char * width}")
    else:
        print(f"{char * width}")


def print_header(image_path, mode):
    """Print analysis header with image info"""
    print_separator("CamDoc - Camera Quality Assessment")
    print(f"📸 Image: {image_path}")
    print(f"🔧 Mode: {mode}")
    
    # Get image dimensions
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            print(f"📏 Size: {w}x{h}")
    except:
        pass


def analyze_camqa(image_path, mode, config):
    """Perform CamQA analysis and return results"""
    print_separator("CamQA Analysis")
    
    # Initialize CamQA
    qa = CamQA(mode=mode)
    
    # Load and analyze image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    res = qa.analyze(bgr)
    scores = res["scores"]
    
    # Print scene mode
    print(f"\n🌅 Detected Scene Mode: {res['mode'].upper()}")
    
    # Print quality scores
    print(f"\n📊 Quality Scores (0-1, higher is better):")
    print(f"  • Sharpness:        {scores['sharpness']:.3f}")
    print(f"  • Brightness:       {scores['brightness']:.3f}")
    print(f"  • Color Intensity:  {scores['color_intensity']:.3f}")
    print(f"  • Lens Cleanliness: {scores['lens_cleanliness']:.3f}")
    
    # Calculate overall health
    overall = (
        0.30 * scores['sharpness'] +
        0.30 * scores['brightness'] +
        0.20 * scores['color_intensity'] +
        0.20 * scores['lens_cleanliness']
    )
    print(f"\n⭐ Overall Health Score: {overall:.3f}")
    
    # Quality classification
    quality_class = config.classify_quality(overall)
    print(f"📈 Quality Class: {quality_class.upper()}")
    
    # Health policy checks
    if config.needs_maintenance(overall):
        print("⚠️  WARNING: Camera needs maintenance!")
    if config.should_disable_analytics(overall):
        print("🚫 CRITICAL: Analytics should be disabled!")
    
    # Print raw metrics
    raw = res["raw"]
    print(f"\n🔍 Raw Metrics:")
    print(f"  • Variance of Laplacian: {raw['vol']:.1f}")
    print(f"  • Tenengrad:             {raw['tenengrad']:.1f}")
    print(f"  • Edge Density:          {raw['edge_density']:.3f}")
    print(f"  • Mean Luminance (Y):    {raw['meanY']:.1f}")
    print(f"  • Dynamic Range:         {int(raw['dyn_range'])}")
    print(f"  • Colorfulness:          {raw['colorfulness']:.1f}")
    print(f"  • HSV Saturation:        {raw['hsv_s']:.1f}")
    print(f"  • Dark Channel Mean:     {raw['dark_channel_mean']:.1f}")
    print(f"  • Static Dark Ratio:     {raw['static_dark_ratio']*100:.2f}%")
    
    return res, overall, qa


def analyze_advanced_metrics(bgr, overall, scores):
    """Perform advanced metrics analysis and return updated overall score"""
    print_separator("Advanced Metrics (BRISQUE/NIQE/TV)")
    
    try:
        adv = AdvancedQualityMetrics(device="cpu")
        adv_result = adv.analyze(bgr)
        
        # Print raw metrics
        raw_adv = adv_result["raw"]
        print(f"\n📐 Raw Metrics (lower is better):")
        print(f"  • BRISQUE:          {raw_adv['brisque']:.2f}")
        print(f"  • NIQE:             {raw_adv['niqe']:.2f}")
        print(f"  • Total Variation:  {raw_adv['total_variation']:.6f}")
        
        # Print normalized scores
        norm = adv_result["normalized"]
        print(f"\n📊 Normalized Quality Scores (0-1, higher is better):")
        print(f"  • BRISQUE Quality:  {norm['brisque_quality']:.3f}")
        print(f"  • NIQE Quality:     {norm['niqe_quality']:.3f}")
        print(f"  • TV Quality:       {norm['tv_quality']:.3f}")
        
        print(f"\n⭐ Overall Advanced Quality: {adv_result['overall_quality']:.3f}")
        
        # Calculate combined Overall Quality (CamQA + Advanced)
        overall_combined = (
            # CamQA metrics - 35% weight
            0.105 * scores['sharpness'] +
            0.105 * scores['brightness'] +
            0.07 * scores['color_intensity'] +
            0.07 * scores['lens_cleanliness'] +
            # Advanced metrics - 65% weight
            0.27 * norm['brisque_quality'] +
            0.27 * norm['niqe_quality'] +
            0.11 * norm['tv_quality']
        )
        
        # Update the overall score to combined
        overall = overall_combined
        
        print_separator()
        print(f"\n⭐ OVERALL QUALITY: {overall:.3f}")
        
        # Calculate contributions
        camqa_contrib = (0.105 * scores['sharpness'] + 0.105 * scores['brightness'] + 
                        0.07 * scores['color_intensity'] + 0.07 * scores['lens_cleanliness'])
        adv_contrib = (0.27 * norm['brisque_quality'] + 0.27 * norm['niqe_quality'] + 
                      0.11 * norm['tv_quality'])
        
        print(f"   ├─ CamQA contribution (35%):    {camqa_contrib:.3f}")
        print(f"   └─ Advanced contribution (65%): {adv_contrib:.3f}")
        print(f"\n   Weights: Sharpness(10.5%) Brightness(10.5%) Color(7%) Lens(7%)")
        print(f"            BRISQUE(27%) NIQE(27%) TV(11%)")
        
        return overall
        
    except ImportError as e:
        print(f"\n❌ Error: Advanced metrics require PyIQA and PyTorch")
        print(f"   Install with: pip install pyiqa torch")
        return overall
    except Exception as e:
        print(f"\n❌ Error in advanced metrics: {e}")
        return overall


def analyze_predictive_features(bgr, qa, use_advanced):
    """Perform predictive features analysis"""
    print_separator("Predictive Features Analysis")
    
    try:
        pred_config = PredictiveConfig()
        
        # Build features
        features, _ = build_features(
            bgr,
            prev_state=None,
            use_advanced=use_advanced,
            use_temporal=False,  # Single image, no temporal
            camqa=qa
        )
        
        # Print extracted features
        print(f"\n🔍 Extracted Features:")
        print(f"  • CamQA Sharpness:      {features['camqa_sharpness']:.3f}")
        print(f"  • CamQA Brightness:     {features['camqa_brightness']:.3f}")
        print(f"  • CamQA Color Intensity: {features['camqa_color_intensity']:.3f}")
        print(f"  • CamQA Lens Cleanliness: {features['camqa_lens_cleanliness']:.3f}")
        print(f"  • Variance of Laplacian: {features['vol']:.1f}")
        print(f"  • Tenengrad:            {features['tenengrad']:.1f}")
        print(f"  • Edge Density:         {features['edge_density']:.3f}")
        print(f"  • Mean Luminance (Y):   {features['meanY']:.1f}")
        print(f"  • Dynamic Range:        {features['dyn_range']:.1f}")
        print(f"  • Colorfulness:         {features['colorfulness']:.1f}")
        
        # Scene classification
        print(f"\n🌅 Scene Classification:")
        print(f"  • Day Scene:            {'✓' if features['scene_day'] else '✗'}")
        print(f"  • Night Scene:          {'✓' if features['scene_night'] else '✗'}")
        print(f"  • Twilight Scene:       {'✓' if features['scene_twilight'] else '✗'}")
        
        # Temporal features (single image)
        print(f"\n⏱️  Temporal Features (Single Image):")
        print(f"  • Jitter (px/frame):    {features['jitter_px']:.2f}")
        print(f"  • Frame Diff Variance:  {features['frame_diff_var']:.1f}")
        print(f"  • Flicker Std:          {features['meanY_flicker_std']:.2f}")
        print(f"  • Running Mean Y:       {features['meanY_running_mean']:.1f}")
        
        # Performance
        print(f"\n⚡ Performance:")
        print(f"  • Feature Extract Time: {features['feature_extract_ms']:.1f} ms")
        
        # Get recommendations
        recommendations = pred_config.get_analysis_recommendations(features)
        
        print(f"\n📊 Analysis Summary:")
        print(f"  • Temporal Quality:     {recommendations['temporal_quality']:.3f} (tek görüntü için sabit)")
        print(f"  • Stability Class:      {recommendations['stability_class'].upper()}")
        
        if recommendations['suggestions']:
            print(f"\n💡 Suggestions:")
            for suggestion in recommendations['suggestions']:
                print(f"  • {suggestion}")
        
        if recommendations['alerts']:
            print(f"\n⚠️  Alerts:")
            for alert in recommendations['alerts']:
                print(f"  • {alert}")
        
        print(f"\n📝 Not: Temporal analiz (jitter, flicker) sadece video akışı veya çoklu frame'ler için anlamlıdır")
        
    except ImportError as e:
        print(f"\n❌ Error: Predictive features require additional dependencies")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"\n❌ Error in predictive features: {e}")


def analyze_image(image_path, mode='auto', use_advanced=False, use_predictive=False):
    """
    Analyze an image using CamQA and optionally advanced metrics and predictive features.
    
    Args:
        image_path: Path to image file
        mode: 'auto', 'day', or 'night'
        use_advanced: Whether to use BRISQUE/NIQE/TV metrics
        use_predictive: Whether to use predictive features analysis
    """
    # Load configuration
    config = ConfigLoader()
    
    # Validate image path
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    # Print header
    print_header(str(image_path), mode)
    
    try:
        # Perform CamQA analysis
        res, overall, qa = analyze_camqa(str(image_path), mode, config)
        scores = res["scores"]
        bgr = cv2.imread(str(image_path))
        
        # Perform advanced metrics analysis if requested
        if use_advanced:
            overall = analyze_advanced_metrics(bgr, overall, scores)
        
        # Perform predictive features analysis if requested
        if use_predictive:
            analyze_predictive_features(bgr, qa, use_advanced)
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return
    
    # Final separator
    print_separator()


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='CamDoc - Camera Quality Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --image test/images/img1.jpg
  python main.py --image my_image.jpg --advanced
  python main.py --image my_image.jpg --mode night --advanced
  python main.py --image my_image.jpg --predictive
  python main.py --image my_image.jpg --advanced --predictive
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        default='test/images/img1.jpg',
        help='Path to image file (default: test/images/img1.jpg)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['auto', 'day', 'night'],
        default='auto',
        help='Scene mode (default: auto)'
    )
    
    parser.add_argument(
        '--advanced', '-a',
        action='store_true',
        help='Enable advanced metrics (BRISQUE/NIQE/TV)'
    )
    
    parser.add_argument(
        '--predictive', '-p',
        action='store_true',
        help='Enable predictive features analysis'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_image(args.image, mode=args.mode, use_advanced=args.advanced, use_predictive=args.predictive)


if __name__ == "__main__":
    main()
