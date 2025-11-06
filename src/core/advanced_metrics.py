"""Advanced Image Quality Metrics using PyIQA and PIQ"""
import numpy as np
from PIL import Image

# Optional dependencies
try:
    import torch
    import pyiqa
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False

try:
    import piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False


def load_tensor01(path_or_array):
    """
    Load image as tensor in [0,1] range.
    
    Args:
        path_or_array: Either a file path (str) or numpy array (BGR format)
    
    Returns:
        torch.Tensor: Shape (1, C, H, W) in [0,1] range
    """
    if not HAS_PYIQA:
        raise ImportError("PyTorch is required for advanced metrics. Install with: pip install torch")
    
    if isinstance(path_or_array, str):
        img = Image.open(path_or_array).convert("RGB")
        arr = np.array(img)
    else:
        # Assume BGR numpy array from OpenCV
        import cv2
        arr = cv2.cvtColor(path_or_array, cv2.COLOR_BGR2RGB)
    
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ten


def total_variation_manual(x):
    """
    Manual Total Variation computation.
    
    Args:
        x: Tensor (N, C, H, W) in [0,1] range
    
    Returns:
        float: Total variation value
    """
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return (dx + dy).item()


def normalize(v, lo, hi, invert=False):
    """
    Normalize value to [0,1] range.
    
    Args:
        v: Value to normalize
        lo: Lower bound
        hi: Upper bound
        invert: If True, invert the normalized value (for "lower is better" metrics)
    
    Returns:
        float: Normalized value in [0,1]
    """
    v = float(v)
    s = (v - lo) / (hi - lo + 1e-9)
    s = max(0.0, min(1.0, s))
    return 1.0 - s if invert else s


def normalize_tv(tv_raw, optimal=0.020, min_threshold=0.005, max_good=0.040, max_bad=0.100):
    """
    Normalize Total Variation with optimal range consideration.
    
    TV interpretation:
    - Very low (< 0.005): Too smooth, detail loss → Bad
    - Optimal (0.015-0.025): Good balance → Best
    - High (> 0.040): Noisy or over-processed → Bad
    - Very high (> 0.100): Very noisy → Very bad
    
    Args:
        tv_raw: Raw TV value
        optimal: Optimal TV value (default: 0.020)
        min_threshold: Below this = detail loss (default: 0.005)
        max_good: Above this starts to be noisy (default: 0.040)
        max_bad: Maximum before score goes to 0 (default: 0.100)
    
    Returns:
        float: Quality score in [0,1], higher is better
    """
    tv_raw = float(tv_raw)
    
    if tv_raw < min_threshold:
        # Too smooth: linear from 0 to 0.5
        return 0.5 * (tv_raw / min_threshold)
    elif tv_raw <= max_good:
        # Good range: peak at optimal
        if tv_raw <= optimal:
            # Rising to optimal
            return 0.5 + 0.5 * (tv_raw - min_threshold) / (optimal - min_threshold)
        else:
            # Falling from optimal
            return 1.0 - 0.3 * (tv_raw - optimal) / (max_good - optimal)
    else:
        # Too high: linear decay to 0
        if tv_raw >= max_bad:
            return 0.0
        return 0.7 * (1.0 - (tv_raw - max_good) / (max_bad - max_good))


class AdvancedQualityMetrics:
    """
    Advanced image quality metrics using no-reference quality assessment models.
    Combines BRISQUE, NIQE, and Total Variation for comprehensive quality scoring.
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize quality metrics.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        if not HAS_PYIQA:
            raise ImportError(
                "PyIQA and PyTorch are required for advanced metrics.\n"
                "Install with: pip install pyiqa torch"
            )
        
        self.device = device
        self.brisque_metric = pyiqa.create_metric('brisque', device=device)
        self.niqe_metric = pyiqa.create_metric('niqe', device=device)
        
        # Normalization ranges (calibrated for typical field conditions)
        self.brisque_range = (10, 60)   # lower is better
        self.niqe_range = (2.0, 7.0)    # lower is better
        self.tv_range = (0.005, 0.05)   # lower is better (smoother)
        
        # Weights for combined score
        self.weights = {
            'brisque': 0.5,
            'niqe': 0.35,
            'tv': 0.15
        }
    
    def compute_brisque(self, tensor):
        """Compute BRISQUE score (lower is better)"""
        return self.brisque_metric(tensor).item()
    
    def compute_niqe(self, tensor):
        """Compute NIQE score (lower is better)"""
        return self.niqe_metric(tensor).item()
    
    def compute_tv(self, tensor):
        """Compute Total Variation (lower is smoother)"""
        if HAS_PIQ:
            return piq.total_variation(tensor, reduction='mean').item()
        else:
            return total_variation_manual(tensor)
    
    def analyze(self, image_input):
        """
        Analyze image quality using advanced metrics.
        
        Args:
            image_input: Either file path (str) or BGR numpy array
        
        Returns:
            dict: Contains raw metrics, normalized scores, and combined quality score
        """
        # Load image as tensor
        tensor = load_tensor01(image_input)
        
        # Compute raw metrics
        brisque_raw = self.compute_brisque(tensor)
        niqe_raw = self.compute_niqe(tensor)
        tv_raw = self.compute_tv(tensor)
        
        # Normalize to [0,1] quality scores (higher is better)
        q_brisque = normalize(brisque_raw, *self.brisque_range, invert=True)
        q_niqe = normalize(niqe_raw, *self.niqe_range, invert=True)
        q_tv = normalize_tv(tv_raw)  # Use specialized TV normalization
        
        # Weighted combination
        true_quality = (
            self.weights['brisque'] * q_brisque +
            self.weights['niqe'] * q_niqe +
            self.weights['tv'] * q_tv
        )
        
        return {
            "raw": {
                "brisque": float(brisque_raw),
                "niqe": float(niqe_raw),
                "total_variation": float(tv_raw)
            },
            "normalized": {
                "brisque_quality": float(q_brisque),
                "niqe_quality": float(q_niqe),
                "tv_quality": float(q_tv)
            },
            "overall_quality": float(true_quality)
        }
    
    def set_normalization_ranges(self, brisque_range=None, niqe_range=None, tv_range=None):
        """
        Update normalization ranges for field calibration.
        
        Args:
            brisque_range: Tuple (min, max)
            niqe_range: Tuple (min, max)
            tv_range: Tuple (min, max)
        """
        if brisque_range:
            self.brisque_range = brisque_range
        if niqe_range:
            self.niqe_range = niqe_range
        if tv_range:
            self.tv_range = tv_range
    
    def set_weights(self, brisque=None, niqe=None, tv=None):
        """
        Update weights for combined quality score.
        Weights will be automatically normalized to sum to 1.0.
        
        Args:
            brisque: Weight for BRISQUE
            niqe: Weight for NIQE
            tv: Weight for Total Variation
        """
        if brisque is not None:
            self.weights['brisque'] = brisque
        if niqe is not None:
            self.weights['niqe'] = niqe
        if tv is not None:
            self.weights['tv'] = tv
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
