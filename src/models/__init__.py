"""
AI Models Module for Precision Agriculture Platform

This module contains:
- CNN feature extractors for spatial and spectral analysis
- LSTM models with attention for temporal modeling
- Hybrid CNN-LSTM architectures
- Training utilities and optimization
- Model evaluation and validation
"""

from .cnn_models import CNNFeatureExtractor, SpectralCNN, SpatialCNN
from .lstm_models import TemporalAttentionLSTM, MultiScaleLSTM
from .hybrid_models import CropHealthModel, MultiModalModel
from .training import ModelTrainer, HyperparameterOptimizer
from .evaluation import ModelEvaluator

__all__ = [
    'CNNFeatureExtractor',
    'SpectralCNN', 
    'SpatialCNN',
    'TemporalAttentionLSTM',
    'MultiScaleLSTM',
    'CropHealthModel',
    'MultiModalModel',
    'ModelTrainer',
    'HyperparameterOptimizer',
    'ModelEvaluator'
]