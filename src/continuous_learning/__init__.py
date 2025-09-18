"""
Continuous Learning Module for Precision Agriculture Platform

This module provides:
- Active learning for sample selection
- Model drift detection and monitoring
- Automated retraining pipelines
- User feedback integration
- Performance tracking and alerts
"""

from .active_learning import ActiveLearner, UncertaintySampler, DiversitySampler
from .drift_detection import DriftDetector, PerformanceMonitor
from .feedback_system import FeedbackCollector, AnnotationInterface
from .retraining import AutoRetrainer, ModelUpdater

__all__ = [
    'ActiveLearner',
    'UncertaintySampler',
    'DiversitySampler',
    'DriftDetector',
    'PerformanceMonitor',
    'FeedbackCollector',
    'AnnotationInterface',
    'AutoRetrainer',
    'ModelUpdater'
]