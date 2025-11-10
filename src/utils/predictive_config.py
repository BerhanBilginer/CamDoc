"""
Predictive Features Configuration Loader

This module provides configuration management for the predictive features system,
including temporal analysis settings, feature thresholds, and stability classification.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class PredictiveConfig:
    """Configuration manager for predictive features system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize predictive configuration loader.
        
        Args:
            config_path: Path to predictive config JSON file. If None, uses default.
        """
        if config_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "predictive_config.json"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Predictive config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_path}: {e}")
    
    @property
    def temporal_settings(self) -> Dict[str, Any]:
        """Get temporal analysis settings"""
        return self._config["predictive_features"]["temporal_settings"]
    
    @property
    def feature_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get feature threshold definitions"""
        return self._config["predictive_features"]["feature_thresholds"]
    
    @property
    def quality_weights(self) -> Dict[str, float]:
        """Get quality calculation weights"""
        return self._config["predictive_features"]["quality_weights"]
    
    @property
    def analysis_settings(self) -> Dict[str, Any]:
        """Get analysis settings"""
        return self._config["predictive_features"]["analysis_settings"]
    
    @property
    def stability_classification(self) -> Dict[str, Dict[str, float]]:
        """Get stability classification thresholds"""
        return self._config["predictive_features"]["stability_classification"]
    
    def classify_jitter(self, jitter_px: float) -> str:
        """
        Classify jitter level based on pixel movement.
        
        Args:
            jitter_px: Jitter in pixels per frame
            
        Returns:
            Classification: 'low', 'medium', 'high', or 'critical'
        """
        thresholds = self.feature_thresholds["jitter_px"]
        
        if jitter_px <= thresholds["low"]:
            return "low"
        elif jitter_px <= thresholds["medium"]:
            return "medium"
        elif jitter_px <= thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    def classify_flicker(self, flicker_std: float) -> str:
        """
        Classify flicker level based on luminance standard deviation.
        
        Args:
            flicker_std: Standard deviation of meanY values
            
        Returns:
            Classification: 'low', 'medium', 'high', or 'critical'
        """
        thresholds = self.feature_thresholds["meanY_flicker_std"]
        
        if flicker_std <= thresholds["low"]:
            return "low"
        elif flicker_std <= thresholds["medium"]:
            return "medium"
        elif flicker_std <= thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    def classify_frame_diff(self, frame_diff_var: float) -> str:
        """
        Classify frame difference variance level.
        
        Args:
            frame_diff_var: Frame difference variance
            
        Returns:
            Classification: 'low', 'medium', 'high', or 'critical'
        """
        thresholds = self.feature_thresholds["frame_diff_var"]
        
        if frame_diff_var <= thresholds["low"]:
            return "low"
        elif frame_diff_var <= thresholds["medium"]:
            return "medium"
        elif frame_diff_var <= thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    def classify_overall_stability(self, jitter_px: float, flicker_std: float, 
                                 frame_diff_var: float) -> str:
        """
        Classify overall temporal stability based on all metrics.
        
        Args:
            jitter_px: Jitter in pixels per frame
            flicker_std: Standard deviation of meanY values
            frame_diff_var: Frame difference variance
            
        Returns:
            Overall stability: 'excellent', 'good', 'fair', or 'poor'
        """
        stability_classes = self.stability_classification
        
        # Check from best to worst
        for level in ["excellent", "good", "fair", "poor"]:
            thresholds = stability_classes[level]
            if (jitter_px <= thresholds["jitter_max"] and
                flicker_std <= thresholds["flicker_std_max"] and
                frame_diff_var <= thresholds["frame_diff_var_max"]):
                return level
        
        return "poor"  # Fallback
    
    def calculate_temporal_quality_score(self, jitter_px: float, flicker_std: float, 
                                       frame_diff_var: float) -> float:
        """
        Calculate a normalized temporal quality score (0-1, higher is better).
        
        Args:
            jitter_px: Jitter in pixels per frame
            flicker_std: Standard deviation of meanY values
            frame_diff_var: Frame difference variance
            
        Returns:
            Temporal quality score between 0 and 1
        """
        # Normalize each metric to 0-1 scale (lower raw values = higher quality)
        jitter_thresholds = self.feature_thresholds["jitter_px"]
        flicker_thresholds = self.feature_thresholds["meanY_flicker_std"]
        frame_diff_thresholds = self.feature_thresholds["frame_diff_var"]
        
        # Inverse sigmoid normalization (lower is better)
        jitter_score = max(0, min(1, 1 - jitter_px / jitter_thresholds["critical"]))
        flicker_score = max(0, min(1, 1 - flicker_std / flicker_thresholds["critical"]))
        frame_diff_score = max(0, min(1, 1 - frame_diff_var / frame_diff_thresholds["critical"]))
        
        # Weighted average
        temporal_score = (
            0.4 * jitter_score +
            0.3 * flicker_score +
            0.3 * frame_diff_score
        )
        
        return temporal_score
    
    def get_analysis_recommendations(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get analysis recommendations based on feature values.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary with recommendations and alerts
        """
        recommendations = {
            "alerts": [],
            "suggestions": [],
            "stability_class": "unknown",
            "temporal_quality": 0.0
        }
        
        # Extract temporal features
        jitter = features.get("jitter_px", 0.0)
        flicker = features.get("meanY_flicker_std", 0.0)
        frame_diff = features.get("frame_diff_var", 0.0)
        
        # Classify individual metrics
        jitter_class = self.classify_jitter(jitter)
        flicker_class = self.classify_flicker(flicker)
        frame_diff_class = self.classify_frame_diff(frame_diff)
        
        # Overall stability
        stability_class = self.classify_overall_stability(jitter, flicker, frame_diff)
        recommendations["stability_class"] = stability_class
        
        # Calculate temporal quality score
        temporal_quality = self.calculate_temporal_quality_score(jitter, flicker, frame_diff)
        recommendations["temporal_quality"] = temporal_quality
        
        # Generate alerts and suggestions
        if jitter_class in ["high", "critical"]:
            recommendations["alerts"].append(f"High camera jitter detected ({jitter:.2f} px/frame)")
            recommendations["suggestions"].append("Consider stabilizing camera mount or reducing vibration")
        
        if flicker_class in ["high", "critical"]:
            recommendations["alerts"].append(f"Significant luminance flicker detected (std: {flicker:.2f})")
            recommendations["suggestions"].append("Check lighting conditions and camera exposure settings")
        
        if frame_diff_class in ["high", "critical"]:
            recommendations["alerts"].append(f"High frame-to-frame variation detected ({frame_diff:.0f})")
            recommendations["suggestions"].append("Verify camera settings and environmental conditions")
        
        if stability_class == "poor":
            recommendations["alerts"].append("Overall temporal stability is poor")
            recommendations["suggestions"].append("Camera requires immediate attention for stable operation")
        elif stability_class == "fair":
            recommendations["suggestions"].append("Consider optimizing camera settings for better stability")
        
        return recommendations
