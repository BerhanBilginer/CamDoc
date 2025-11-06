"""Configuration loader for quality thresholds and settings"""
import json
import os
from pathlib import Path


class ConfigLoader:
    """Loads and manages configuration for camera quality assessment"""
    
    def __init__(self, config_path=None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Default: config/quality_thresholds.json relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "quality_thresholds.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_thresholds(self, mode='day'):
        """
        Get quality thresholds for specified mode.
        
        Args:
            mode: 'day' or 'night'
        
        Returns:
            dict: Thresholds for the specified mode
        """
        return self.config['camera_quality_thresholds'].get(mode, {})
    
    def get_quality_classes(self):
        """Get quality classification thresholds"""
        return self.config['camera_quality_thresholds']['quality_classes']
    
    def get_health_policy(self):
        """Get health policy thresholds"""
        return self.config['camera_quality_thresholds']['health_policy']
    
    def classify_quality(self, score):
        """
        Classify quality score into categories.
        
        Args:
            score: Quality score (0-1)
        
        Returns:
            str: Quality class ('high', 'medium', 'low', or 'unusable')
        """
        classes = self.get_quality_classes()
        
        if score >= classes['high_quality_min']:
            return 'high'
        elif score >= classes['medium_quality_min']:
            return 'medium'
        elif score >= classes['low_quality_min']:
            return 'low'
        else:
            return 'unusable'
    
    def needs_maintenance(self, score):
        """
        Check if camera needs maintenance based on score.
        
        Args:
            score: Quality score (0-1)
        
        Returns:
            bool: True if maintenance needed
        """
        policy = self.get_health_policy()
        return score < policy['maintenance_threshold']
    
    def should_disable_analytics(self, score):
        """
        Check if analytics should be disabled based on score.
        
        Args:
            score: Quality score (0-1)
        
        Returns:
            bool: True if analytics should be disabled
        """
        policy = self.get_health_policy()
        return score < policy['disable_analytics_threshold']
    
    def get_all(self):
        """Get entire configuration"""
        return self.config
