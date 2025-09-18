"""
Explainability Module for Precision Agriculture Platform

This module provides:
- Grad-CAM for spatial-spectral attribution
- SHAP for feature importance analysis
- Attention visualization for temporal models
- Out-of-distribution detection
- Model interpretability tools
"""

from .grad_cam import GradCAM, SpectralGradCAM
from .shap_analysis import SHAPAnalyzer, SpectralSHAP
from .attention_viz import AttentionVisualizer
from .ood_detection import OutOfDistributionDetector
from .interpretability import ModelInterpreter

__all__ = [
    'GradCAM',
    'SpectralGradCAM', 
    'SHAPAnalyzer',
    'SpectralSHAP',
    'AttentionVisualizer',
    'OutOfDistributionDetector',
    'ModelInterpreter'
]