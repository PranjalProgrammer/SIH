"""
Data Acquisition Module for Precision Agriculture Platform

This module handles:
- Dataset downloading from public repositories
- Hyperspectral and multispectral image processing
- IoT sensor data integration
- Radiometric and atmospheric corrections
- Feature extraction and preprocessing
"""

from .download_datasets import DatasetDownloader
from .preprocess_images import ImagePreprocessor
from .sensor_processing import SensorDataProcessor
from .feature_extraction import FeatureExtractor

__all__ = [
    'DatasetDownloader',
    'ImagePreprocessor', 
    'SensorDataProcessor',
    'FeatureExtractor'
]