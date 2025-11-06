"""
CamDoc - Camera Quality Assessment Main Demo

This script demonstrates the integration of:
- CamQA: Fast camera quality assessment (sharpness, brightness, color, lens cleanliness)
- Advanced Metrics: BRISQUE, NIQE, Total Variation (optional)
- Config Loader: Quality thresholds and health policies

Usage:
    python main.py [--image PATH] [--advanced] [--mode auto|day|night]
"""
import sys
import cv2
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import CamQA, AdvancedQualityMetrics
from src.utils import ConfigLoader


def print_separator(title="", char="=", width=70):
    """Print a formatted separator line"""
    if title:
        print(f"\n{char * width}")
        print(f"{title.center(width)}")
        print(f"{char * width}")
    else:
        print(f"{char * width}")


def analyze_image(image_path, mode='auto', use_advanced=False):
    """
    Analyze an image using CamQA and optionally advanced metrics.
    
    Args:
        image_path: Path to image file
        mode: 'auto', 'day', or 'night'
        use_advanced: Whether to use BRISQUE/NIQE/TV metrics
    """
    # Load configuration
    config = ConfigLoader()
    
    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return
    
    print_separator("CamDoc - Camera Quality Assessment")
    print(f"📸 Image: {image_path}")
    print(f"🔧 Mode: {mode}")
    print(f"📏 Size: {frame.shape[1]}x{frame.shape[0]}")
    
    # ========== CamQA Analysis ==========
    print_separator("CamQA Analysis")
    
    qa = CamQA(short_side=720, ema=0.2, mode=mode)
    result = qa.analyze(frame)
    
    print(f"\n🌅 Detected Scene Mode: {result['mode'].upper()}")
    
    print("\n📊 Quality Scores (0-1, higher is better):")
    scores = result['scores']
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
    
    # Raw metrics
    print("\n🔍 Raw Metrics:")
    raw = result['raw']
    print(f"  • Variance of Laplacian: {raw['vol']:.1f}")
    print(f"  • Tenengrad:             {raw['tenengrad']:.1f}")
    print(f"  • Edge Density:          {raw['edge_density']:.3f}")
    print(f"  • Mean Luminance (Y):    {raw['meanY']:.1f}")
    print(f"  • Dynamic Range:         {raw['dyn_range']}")
    print(f"  • Colorfulness:          {raw['colorfulness']:.1f}")
    print(f"  • HSV Saturation:        {raw['hsv_s']:.1f}")
    print(f"  • Dark Channel Mean:     {raw['dark_channel_mean']:.1f}")
    print(f"  • Static Dark Ratio:     {raw['static_dark_ratio']*100:.2f}%")
    
    # ========== Advanced Metrics ==========
    if use_advanced:
        print_separator("Advanced Metrics (BRISQUE/NIQE/TV)")
        
        try:
            adv = AdvancedQualityMetrics(device='cpu')
            adv_result = adv.analyze(frame)
            
            print("\n📐 Raw Metrics (lower is better):")
            print(f"  • BRISQUE:          {adv_result['raw']['brisque']:.2f}")
            print(f"  • NIQE:             {adv_result['raw']['niqe']:.2f}")
            print(f"  • Total Variation:  {adv_result['raw']['total_variation']:.6f}")
            
            print("\n📊 Normalized Quality Scores (0-1, higher is better):")
            norm = adv_result['normalized']
            print(f"  • BRISQUE Quality:  {norm['brisque_quality']:.3f}")
            print(f"  • NIQE Quality:     {norm['niqe_quality']:.3f}")
            print(f"  • TV Quality:       {norm['tv_quality']:.3f}")
            
            print(f"\n⭐ Overall Advanced Quality: {adv_result['overall_quality']:.3f}")
            
            # Calculate combined Overall Quality (CamQA + Advanced)
            overall_combined = (
                # CamQA metrics - 56% weight
                0.16 * scores['sharpness'] +
                0.16 * scores['brightness'] +
                0.12 * scores['color_intensity'] +
                0.12 * scores['lens_cleanliness'] +
                # Advanced metrics - 44% weight
                0.18 * norm['brisque_quality'] +
                0.18 * norm['niqe_quality'] +
                0.12 * norm['tv_quality']
            )
            
            print_separator()
            print(f"\n⭐ OVERALL QUALITY (Combined): {overall_combined:.3f}")
            print(f"   ├─ CamQA contribution (56%):    {overall:.3f}")
            print(f"   └─ Advanced contribution (44%): {adv_result['overall_quality']:.3f}")
            print(f"\n   Weights: Sharpness(16%) Brightness(16%) Color(12%) Lens(12%)")
            print(f"            BRISQUE(18%) NIQE(18%) TV(12%)")
            
        except ImportError as e:
            print(f"\n❌ Error: Advanced metrics require PyIQA and PyTorch")
            print(f"   Install with: pip install pyiqa torch")
        except Exception as e:
            print(f"\n❌ Error in advanced metrics: {e}")
    
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
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_image(args.image, mode=args.mode, use_advanced=args.advanced)


if __name__ == "__main__":
    main()
