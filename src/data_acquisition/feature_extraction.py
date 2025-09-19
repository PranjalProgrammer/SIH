"""
Feature Extraction Module
Extracts features from agricultural imagery and sensor data
"""

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts features from agricultural data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def extract_spectral_features(self, hyperspectral_data):
        """Extract spectral features from hyperspectral data"""
        features = {}
        
        # Mean spectral response
        features['mean_spectrum'] = np.mean(hyperspectral_data, axis=(0, 1))
        
        # Vegetation indices
        if hyperspectral_data.shape[2] >= 3:
            red = hyperspectral_data[:, :, 0]
            green = hyperspectral_data[:, :, 1] 
            nir = hyperspectral_data[:, :, 2] if hyperspectral_data.shape[2] > 3 else red
            
            # NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            features['ndvi'] = np.mean(ndvi)
            
            # GNDVI
            gndvi = (nir - green) / (nir + green + 1e-8)
            features['gndvi'] = np.mean(gndvi)
        
        logger.info("Spectral features extracted successfully")
        return features
    
    def extract_texture_features(self, image):
        """Extract texture features using GLCM"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Simple texture measures
        features = {
            'contrast': np.var(gray),
            'homogeneity': np.mean(gray),
            'energy': np.sum(gray**2),
            'correlation': np.corrcoef(gray.flatten(), gray.flatten())[0, 1] if gray.size > 1 else 0
        }
        
        return features
    
    def extract_morphological_features(self, binary_mask):
        """Extract morphological features from plant masks"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'area': 0, 'perimeter': 0, 'compactness': 0}
        
        # Find largest contour (main plant)
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = 4 * np.pi * area / (perimeter**2 + 1e-8)
        
        return {
            'area': area,
            'perimeter': perimeter, 
            'compactness': compactness
        }

def main():
    """Main feature extraction function"""
    extractor = FeatureExtractor()
    logger.info("Feature extractor initialized")

if __name__ == "__main__":
    main()