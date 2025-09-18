"""
Deployment Module for Precision Agriculture Platform

This module provides:
- FastAPI REST API for model serving
- Real-time processing pipeline
- Model optimization for edge devices
- Monitoring and logging
- Database integration
"""

from .api import app, create_app
from .model_server import ModelServer
from .edge_optimization import EdgeOptimizer
from .monitoring import MetricsCollector, HealthChecker

__all__ = [
    'app',
    'create_app',
    'ModelServer',
    'EdgeOptimizer', 
    'MetricsCollector',
    'HealthChecker'
]