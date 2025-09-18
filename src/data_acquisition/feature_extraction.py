"""
Feature Extraction Module

Extracts vegetation indices, soil indices, and spectral features
from hyperspectral imagery for crop health analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts spectral features and vegetation indices from hyperspectral imagery."""
    
    def __init__(self, data_dir: str = "/workspace/data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard wavelength ranges for common bands (in nm)
        self.band_ranges = {
            'blue': (450, 520),
            'green': (520, 600),
            'red': (630, 690),
            'red_edge': (690, 730),
            'nir': (760, 900),
            'swir1': (1550, 1750),
            'swir2': (2080, 2350)
        }
        
        # Wavelength centers for hyperspectral analysis
        self.wavelengths = np.linspace(400, 2500, 200)  # Default 400-2500nm range
    
    def set_wavelengths(self, wavelengths: np.ndarray) -> None:
        """Set the wavelength array for the hyperspectral data."""
        self.wavelengths = wavelengths
        logger.info(f"Set wavelengths: {len(wavelengths)} bands from {wavelengths[0]:.1f} to {wavelengths[-1]:.1f} nm")
    
    def find_band_index(self, target_wavelength: float) -> int:
        """Find the closest band index for a target wavelength."""
        return np.argmin(np.abs(self.wavelengths - target_wavelength))
    
    def get_band_range_indices(self, start_wl: float, end_wl: float) -> Tuple[int, int]:
        """Get start and end indices for a wavelength range."""
        start_idx = self.find_band_index(start_wl)
        end_idx = self.find_band_index(end_wl)
        return start_idx, end_idx + 1
    
    def calculate_ndvi(self, img_array: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            img_array: Hyperspectral image array (H, W, Bands)
            
        Returns:
            NDVI array (H, W)
        """
        logger.info("Calculating NDVI...")
        
        # Find NIR and Red band indices
        nir_idx = self.find_band_index(850)  # NIR ~850nm
        red_idx = self.find_band_index(670)  # Red ~670nm
        
        nir_band = img_array[:, :, nir_idx]
        red_band = img_array[:, :, red_idx]
        
        # Calculate NDVI
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        
        logger.info("NDVI calculation completed")
        return ndvi
    
    def calculate_evi(self, img_array: np.ndarray) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index (EVI).
        
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        
        Args:
            img_array: Hyperspectral image array (H, W, Bands)
            
        Returns:
            EVI array (H, W)
        """
        logger.info("Calculating EVI...")
        
        # Find band indices
        nir_idx = self.find_band_index(850)
        red_idx = self.find_band_index(670)
        blue_idx = self.find_band_index(470)
        
        nir_band = img_array[:, :, nir_idx]
        red_band = img_array[:, :, red_idx]
        blue_band = img_array[:, :, blue_idx]
        
        # Calculate EVI
        evi = 2.5 * (nir_band - red_band) / (nir_band + 6*red_band - 7.5*blue_band + 1 + 1e-8)
        
        logger.info("EVI calculation completed")
        return evi
    
    def calculate_savi(self, img_array: np.ndarray, L: float = 0.5) -> np.ndarray:
        """
        Calculate Soil Adjusted Vegetation Index (SAVI).
        
        SAVI = (NIR - Red) * (1 + L) / (NIR + Red + L)
        
        Args:
            img_array: Hyperspectral image array
            L: Soil brightness correction factor (0.5 for intermediate vegetation)
            
        Returns:
            SAVI array (H, W)
        """
        logger.info("Calculating SAVI...")
        
        nir_idx = self.find_band_index(850)
        red_idx = self.find_band_index(670)
        
        nir_band = img_array[:, :, nir_idx]
        red_band = img_array[:, :, red_idx]
        
        savi = (nir_band - red_band) * (1 + L) / (nir_band + red_band + L + 1e-8)
        
        logger.info("SAVI calculation completed")
        return savi
    
    def calculate_msavi(self, img_array: np.ndarray) -> np.ndarray:
        """
        Calculate Modified Soil Adjusted Vegetation Index (MSAVI).
        
        MSAVI = 0.5 * (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red)))
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            MSAVI array (H, W)
        """
        logger.info("Calculating MSAVI...")
        
        nir_idx = self.find_band_index(850)
        red_idx = self.find_band_index(670)
        
        nir_band = img_array[:, :, nir_idx]
        red_band = img_array[:, :, red_idx]
        
        # MSAVI calculation
        term1 = 2 * nir_band + 1
        term2 = np.sqrt(term1**2 - 8 * (nir_band - red_band))
        msavi = 0.5 * (term1 - term2)
        
        logger.info("MSAVI calculation completed")
        return msavi
    
    def calculate_ndwi(self, img_array: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index (NDWI).
        
        NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            NDWI array (H, W)
        """
        logger.info("Calculating NDWI...")
        
        green_idx = self.find_band_index(560)
        nir_idx = self.find_band_index(850)
        
        green_band = img_array[:, :, green_idx]
        nir_band = img_array[:, :, nir_idx]
        
        ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-8)
        
        logger.info("NDWI calculation completed")
        return ndwi
    
    def calculate_red_edge_indices(self, img_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate various red-edge vegetation indices.
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            Dictionary of red-edge indices
        """
        logger.info("Calculating red-edge indices...")
        
        # Band indices for red-edge analysis
        red_idx = self.find_band_index(670)
        re1_idx = self.find_band_index(705)  # Red-edge 1
        re2_idx = self.find_band_index(740)  # Red-edge 2
        re3_idx = self.find_band_index(783)  # Red-edge 3
        nir_idx = self.find_band_index(850)
        
        red_band = img_array[:, :, red_idx]
        re1_band = img_array[:, :, re1_idx]
        re2_band = img_array[:, :, re2_idx]
        re3_band = img_array[:, :, re3_idx]
        nir_band = img_array[:, :, nir_idx]
        
        indices = {}
        
        # NDRE (Normalized Difference Red Edge)
        indices['ndre'] = (nir_band - re1_band) / (nir_band + re1_band + 1e-8)
        
        # MTCI (MERIS Terrestrial Chlorophyll Index)
        indices['mtci'] = (re2_band - re1_band) / (re1_band - red_band + 1e-8)
        
        # Red Edge Position (REP) - simplified
        indices['rep'] = 700 + 40 * ((red_band + re3_band)/2 - re1_band) / (re2_band - re1_band + 1e-8)
        
        logger.info("Red-edge indices calculation completed")
        return indices
    
    def calculate_chlorophyll_indices(self, img_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate chlorophyll-related vegetation indices.
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            Dictionary of chlorophyll indices
        """
        logger.info("Calculating chlorophyll indices...")
        
        # Band indices
        green_idx = self.find_band_index(560)
        red_idx = self.find_band_index(670)
        re_idx = self.find_band_index(705)
        nir_idx = self.find_band_index(800)
        
        green_band = img_array[:, :, green_idx]
        red_band = img_array[:, :, red_idx]
        re_band = img_array[:, :, re_idx]
        nir_band = img_array[:, :, nir_idx]
        
        indices = {}
        
        # Chlorophyll Index Green (CIG)
        indices['cig'] = (nir_band / green_band) - 1
        
        # Chlorophyll Index Red Edge (CIRE)
        indices['cire'] = (nir_band / re_band) - 1
        
        # Green NDVI
        indices['gndvi'] = (nir_band - green_band) / (nir_band + green_band + 1e-8)
        
        logger.info("Chlorophyll indices calculation completed")
        return indices
    
    def calculate_soil_indices(self, img_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate soil-related spectral indices.
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            Dictionary of soil indices
        """
        logger.info("Calculating soil indices...")
        
        # Band indices
        red_idx = self.find_band_index(670)
        nir_idx = self.find_band_index(850)
        swir1_idx = self.find_band_index(1650)
        swir2_idx = self.find_band_index(2200)
        
        red_band = img_array[:, :, red_idx]
        nir_band = img_array[:, :, nir_idx]
        swir1_band = img_array[:, :, swir1_idx]
        swir2_band = img_array[:, :, swir2_idx]
        
        indices = {}
        
        # Normalized Difference Soil Index (NDSI)
        indices['ndsi'] = (swir1_band - nir_band) / (swir1_band + nir_band + 1e-8)
        
        # Soil Brightness Index (SBI)
        indices['sbi'] = np.sqrt(red_band**2 + nir_band**2)
        
        # Clay Minerals Ratio (CMR)
        indices['cmr'] = swir1_band / swir2_band
        
        logger.info("Soil indices calculation completed")
        return indices
    
    def calculate_stress_indices(self, img_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate plant stress-related indices.
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            Dictionary of stress indices
        """
        logger.info("Calculating stress indices...")
        
        # Band indices
        green_idx = self.find_band_index(560)
        red_idx = self.find_band_index(670)
        nir_idx = self.find_band_index(850)
        swir1_idx = self.find_band_index(1650)
        
        green_band = img_array[:, :, green_idx]
        red_band = img_array[:, :, red_idx]
        nir_band = img_array[:, :, nir_idx]
        swir1_band = img_array[:, :, swir1_idx]
        
        indices = {}
        
        # Plant Senescence Reflectance Index (PSRI)
        indices['psri'] = (red_band - green_band) / nir_band
        
        # Moisture Stress Index (MSI)
        indices['msi'] = swir1_band / nir_band
        
        # Water Band Index (WBI)
        water_idx = self.find_band_index(970)  # Water absorption band
        ref_idx = self.find_band_index(900)    # Reference band
        
        if water_idx < img_array.shape[2] and ref_idx < img_array.shape[2]:
            water_band = img_array[:, :, water_idx]
            ref_band = img_array[:, :, ref_idx]
            indices['wbi'] = ref_band / water_band
        
        logger.info("Stress indices calculation completed")
        return indices
    
    def extract_spectral_statistics(self, img_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from spectral data.
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            Dictionary of spectral statistics
        """
        logger.info("Extracting spectral statistics...")
        
        height, width, bands = img_array.shape
        stats = {}
        
        # Calculate statistics across spectral dimension
        stats['spectral_mean'] = np.mean(img_array, axis=2)
        stats['spectral_std'] = np.std(img_array, axis=2)
        stats['spectral_min'] = np.min(img_array, axis=2)
        stats['spectral_max'] = np.max(img_array, axis=2)
        stats['spectral_range'] = stats['spectral_max'] - stats['spectral_min']
        stats['spectral_median'] = np.median(img_array, axis=2)
        
        # Spectral slope (linear fit across wavelengths)
        spectral_slopes = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                spectrum = img_array[i, j, :]
                slope, _ = np.polyfit(self.wavelengths, spectrum, 1)
                spectral_slopes[i, j] = slope
        
        stats['spectral_slope'] = spectral_slopes
        
        logger.info("Spectral statistics extraction completed")
        return stats
    
    def perform_pca_analysis(self, img_array: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, PCA, np.ndarray]:
        """
        Perform Principal Component Analysis on hyperspectral data.
        
        Args:
            img_array: Hyperspectral image array
            n_components: Number of principal components to extract
            
        Returns:
            PCA-transformed image, PCA object, and explained variance ratio
        """
        logger.info(f"Performing PCA analysis with {n_components} components...")
        
        height, width, bands = img_array.shape
        
        # Reshape for PCA
        reshaped_img = img_array.reshape(-1, bands)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(reshaped_img)
        
        # Reshape back
        pca_img = pca_result.reshape(height, width, n_components)
        
        explained_variance = pca.explained_variance_ratio_
        
        logger.info(f"PCA completed. Total explained variance: {explained_variance.sum():.3f}")
        return pca_img, pca, explained_variance
    
    def detect_anomalies(self, img_array: np.ndarray, method: str = 'isolation') -> np.ndarray:
        """
        Detect spectral anomalies in hyperspectral imagery.
        
        Args:
            img_array: Hyperspectral image array
            method: Anomaly detection method ('isolation', 'kmeans', 'mahalanobis')
            
        Returns:
            Anomaly map (H, W) with anomaly scores
        """
        logger.info(f"Detecting anomalies using {method} method...")
        
        height, width, bands = img_array.shape
        reshaped_img = img_array.reshape(-1, bands)
        
        if method == 'isolation':
            from sklearn.ensemble import IsolationForest
            detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = detector.fit_predict(reshaped_img)
            # Convert to anomaly scores (higher = more anomalous)
            anomaly_scores = -detector.score_samples(reshaped_img)
        
        elif method == 'kmeans':
            # Use distance to cluster centers as anomaly score
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(reshaped_img)
            
            # Calculate distance to nearest cluster center
            distances = []
            for point in reshaped_img:
                cluster_distances = [np.linalg.norm(point - center) 
                                   for center in kmeans.cluster_centers_]
                distances.append(min(cluster_distances))
            
            anomaly_scores = np.array(distances)
        
        elif method == 'mahalanobis':
            # Mahalanobis distance from mean
            mean = np.mean(reshaped_img, axis=0)
            cov = np.cov(reshaped_img.T)
            cov_inv = np.linalg.pinv(cov)  # Pseudo-inverse for numerical stability
            
            anomaly_scores = []
            for point in reshaped_img:
                diff = point - mean
                mahal_dist = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                anomaly_scores.append(mahal_dist)
            
            anomaly_scores = np.array(anomaly_scores)
        
        # Reshape back to image format
        anomaly_map = anomaly_scores.reshape(height, width)
        
        logger.info("Anomaly detection completed")
        return anomaly_map
    
    def extract_all_features(self, img_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all available features from hyperspectral imagery.
        
        Args:
            img_array: Hyperspectral image array
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info("Extracting all spectral features...")
        
        features = {}
        
        # Basic vegetation indices
        features['ndvi'] = self.calculate_ndvi(img_array)
        features['evi'] = self.calculate_evi(img_array)
        features['savi'] = self.calculate_savi(img_array)
        features['msavi'] = self.calculate_msavi(img_array)
        features['ndwi'] = self.calculate_ndwi(img_array)
        
        # Red-edge indices
        red_edge_indices = self.calculate_red_edge_indices(img_array)
        features.update(red_edge_indices)
        
        # Chlorophyll indices
        chlorophyll_indices = self.calculate_chlorophyll_indices(img_array)
        features.update(chlorophyll_indices)
        
        # Soil indices
        soil_indices = self.calculate_soil_indices(img_array)
        features.update(soil_indices)
        
        # Stress indices
        stress_indices = self.calculate_stress_indices(img_array)
        features.update(stress_indices)
        
        # Spectral statistics
        spectral_stats = self.extract_spectral_statistics(img_array)
        features.update(spectral_stats)
        
        # PCA features
        pca_img, _, _ = self.perform_pca_analysis(img_array, n_components=5)
        for i in range(pca_img.shape[2]):
            features[f'pca_{i+1}'] = pca_img[:, :, i]
        
        # Anomaly detection
        features['anomaly_score'] = self.detect_anomalies(img_array)
        
        logger.info(f"Feature extraction completed. Total features: {len(features)}")
        return features
    
    def create_feature_stack(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Stack all features into a single array.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Stacked feature array (H, W, N_features)
        """
        logger.info("Creating feature stack...")
        
        feature_list = []
        feature_names = []
        
        for name, feature_array in features.items():
            if feature_array.ndim == 2:  # 2D feature map
                feature_list.append(feature_array)
                feature_names.append(name)
            elif feature_array.ndim == 3:  # 3D feature stack
                for i in range(feature_array.shape[2]):
                    feature_list.append(feature_array[:, :, i])
                    feature_names.append(f"{name}_{i}")
        
        # Stack all features
        feature_stack = np.stack(feature_list, axis=2)
        
        logger.info(f"Feature stack created: {feature_stack.shape}")
        return feature_stack, feature_names


def main():
    """Main function to demonstrate feature extraction."""
    extractor = FeatureExtractor()
    
    # Load or create sample hyperspectral image
    from .preprocess_images import ImagePreprocessor
    
    preprocessor = ImagePreprocessor()
    sample_img = preprocessor.create_sample_hyperspectral_image(100, 100, 200)
    
    # Set wavelengths for the extractor
    wavelengths = np.linspace(400, 2500, 200)
    extractor.set_wavelengths(wavelengths)
    
    # Extract all features
    features = extractor.extract_all_features(sample_img)
    
    # Create feature stack
    feature_stack, feature_names = extractor.create_feature_stack(features)
    
    # Save results
    np.save(extractor.processed_dir / 'extracted_features.npy', feature_stack)
    
    # Save feature names
    with open(extractor.processed_dir / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    logger.info(f"Features saved: {feature_stack.shape}")
    logger.info(f"Feature names: {len(feature_names)}")
    
    # Print feature summary
    print("\nExtracted Features Summary:")
    for name, feature in features.items():
        if hasattr(feature, 'shape'):
            print(f"{name}: {feature.shape} - Mean: {np.mean(feature):.4f}, Std: {np.std(feature):.4f}")


if __name__ == "__main__":
    main()