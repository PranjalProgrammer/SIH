"""
Image Preprocessing Module

Handles radiometric correction, atmospheric correction, georeferencing,
and temporal alignment for hyperspectral and multispectral imagery.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import spectral.io.envi as envi
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
from scipy import ndimage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses hyperspectral and multispectral imagery for crop analysis."""
    
    def __init__(self, data_dir: str = "/workspace/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def radiometric_correction(self, 
                             img_array: np.ndarray,
                             dark_reference: Optional[np.ndarray] = None,
                             white_reference: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply radiometric correction to hyperspectral imagery.
        
        Args:
            img_array: Input hyperspectral image (H, W, Bands)
            dark_reference: Dark reference image
            white_reference: White reference image
            
        Returns:
            Radiometrically corrected image
        """
        logger.info("Applying radiometric correction...")
        
        # If no reference images provided, estimate from image statistics
        if dark_reference is None:
            dark_reference = np.percentile(img_array, 1, axis=(0, 1), keepdims=True)
        
        if white_reference is None:
            white_reference = np.percentile(img_array, 99, axis=(0, 1), keepdims=True)
        
        # Apply correction formula: (img - dark) / (white - dark)
        corrected_img = (img_array - dark_reference) / (white_reference - dark_reference + 1e-8)
        
        # Clip values to valid range [0, 1]
        corrected_img = np.clip(corrected_img, 0, 1)
        
        logger.info("Radiometric correction completed")
        return corrected_img
    
    def atmospheric_correction_dos(self, img_array: np.ndarray) -> np.ndarray:
        """
        Apply Dark Object Subtraction (DOS) atmospheric correction.
        
        Args:
            img_array: Input image array
            
        Returns:
            Atmospherically corrected image
        """
        logger.info("Applying atmospheric correction (DOS)...")
        
        corrected_img = np.zeros_like(img_array)
        
        for band in range(img_array.shape[2]):
            band_data = img_array[:, :, band]
            
            # Find dark object value (1st percentile)
            dark_value = np.percentile(band_data, 1)
            
            # Subtract dark object value
            corrected_band = band_data - dark_value
            corrected_band = np.clip(corrected_band, 0, None)
            
            corrected_img[:, :, band] = corrected_band
        
        logger.info("Atmospheric correction completed")
        return corrected_img
    
    def geometric_correction(self, 
                           img_array: np.ndarray,
                           src_crs: str = "EPSG:4326",
                           dst_crs: str = "EPSG:32633",
                           bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply geometric correction and reprojection.
        
        Args:
            img_array: Input image array
            src_crs: Source coordinate reference system
            dst_crs: Destination coordinate reference system
            bounds: Geographic bounds (left, bottom, right, top)
            
        Returns:
            Geometrically corrected image and metadata
        """
        logger.info("Applying geometric correction...")
        
        height, width, bands = img_array.shape
        
        # Default bounds if not provided
        if bounds is None:
            bounds = (-1.0, -1.0, 1.0, 1.0)  # Default geographic bounds
        
        # Create transform from bounds
        transform = from_bounds(*bounds, width, height)
        
        # Calculate default transform for reprojection
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, width, height, *bounds
        )
        
        # Prepare output array
        corrected_img = np.zeros((dst_height, dst_width, bands), dtype=img_array.dtype)
        
        # Reproject each band
        for band in range(bands):
            reproject(
                source=img_array[:, :, band],
                destination=corrected_img[:, :, band],
                src_transform=transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
        
        metadata = {
            'transform': dst_transform,
            'crs': dst_crs,
            'width': dst_width,
            'height': dst_height
        }
        
        logger.info("Geometric correction completed")
        return corrected_img, metadata
    
    def noise_reduction(self, img_array: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Apply noise reduction to hyperspectral imagery.
        
        Args:
            img_array: Input image array
            method: Noise reduction method ('gaussian', 'median', 'bilateral')
            
        Returns:
            Denoised image
        """
        logger.info(f"Applying {method} noise reduction...")
        
        denoised_img = np.zeros_like(img_array)
        
        for band in range(img_array.shape[2]):
            band_data = img_array[:, :, band]
            
            if method == 'gaussian':
                denoised_img[:, :, band] = ndimage.gaussian_filter(band_data, sigma=1.0)
            elif method == 'median':
                denoised_img[:, :, band] = ndimage.median_filter(band_data, size=3)
            elif method == 'bilateral':
                # Convert to uint8 for bilateral filter
                band_uint8 = (band_data * 255).astype(np.uint8)
                filtered = cv2.bilateralFilter(band_uint8, 9, 75, 75)
                denoised_img[:, :, band] = filtered.astype(np.float32) / 255.0
        
        logger.info("Noise reduction completed")
        return denoised_img
    
    def spectral_normalization(self, 
                             img_array: np.ndarray, 
                             method: str = 'minmax') -> Tuple[np.ndarray, object]:
        """
        Normalize spectral data across bands.
        
        Args:
            img_array: Input image array
            method: Normalization method ('minmax', 'standard', 'l2')
            
        Returns:
            Normalized image and scaler object
        """
        logger.info(f"Applying {method} spectral normalization...")
        
        original_shape = img_array.shape
        # Reshape to (pixels, bands) for normalization
        reshaped_img = img_array.reshape(-1, original_shape[2])
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'l2':
            from sklearn.preprocessing import normalize
            normalized_img = normalize(reshaped_img, norm='l2', axis=1)
            return normalized_img.reshape(original_shape), None
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        normalized_img = scaler.fit_transform(reshaped_img)
        
        # Reshape back to original shape
        normalized_img = normalized_img.reshape(original_shape)
        
        logger.info("Spectral normalization completed")
        return normalized_img, scaler
    
    def temporal_alignment(self, 
                         image_list: List[np.ndarray],
                         timestamps: List,
                         target_resolution: str = '1H') -> Tuple[List[np.ndarray], List]:
        """
        Align images temporally for time-series analysis.
        
        Args:
            image_list: List of image arrays
            timestamps: List of acquisition timestamps
            target_resolution: Target temporal resolution
            
        Returns:
            Aligned images and timestamps
        """
        logger.info("Performing temporal alignment...")
        
        import pandas as pd
        
        # Create DataFrame with timestamps
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'image_idx': range(len(image_list))
        })
        
        # Set timestamp as index and resample
        df.set_index('timestamp', inplace=True)
        df_resampled = df.resample(target_resolution).first()
        
        # Extract aligned images
        aligned_images = []
        aligned_timestamps = []
        
        for timestamp, row in df_resampled.iterrows():
            if not pd.isna(row['image_idx']):
                aligned_images.append(image_list[int(row['image_idx'])])
                aligned_timestamps.append(timestamp)
        
        logger.info(f"Temporal alignment completed: {len(aligned_images)} images")
        return aligned_images, aligned_timestamps
    
    def create_sample_hyperspectral_image(self, 
                                        height: int = 100, 
                                        width: int = 100, 
                                        bands: int = 200) -> np.ndarray:
        """Create a sample hyperspectral image for testing."""
        logger.info(f"Creating sample hyperspectral image ({height}x{width}x{bands})...")
        
        np.random.seed(42)
        
        # Create base vegetation patterns
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        
        # Simulate different crop areas
        crop1 = ((x - 0.3)**2 + (y - 0.3)**2) < 0.2**2
        crop2 = ((x - 0.7)**2 + (y - 0.7)**2) < 0.15**2
        soil = ~(crop1 | crop2)
        
        # Create spectral signatures
        img = np.zeros((height, width, bands))
        wavelengths = np.linspace(400, 2500, bands)  # 400-2500 nm
        
        for i in range(height):
            for j in range(width):
                if crop1[i, j]:
                    # Healthy vegetation signature
                    reflectance = self._vegetation_signature(wavelengths, health=0.8)
                elif crop2[i, j]:
                    # Stressed vegetation signature  
                    reflectance = self._vegetation_signature(wavelengths, health=0.4)
                else:
                    # Soil signature
                    reflectance = self._soil_signature(wavelengths)
                
                # Add noise
                noise = np.random.normal(0, 0.02, bands)
                img[i, j, :] = np.clip(reflectance + noise, 0, 1)
        
        logger.info("Sample hyperspectral image created")
        return img
    
    def _vegetation_signature(self, wavelengths: np.ndarray, health: float = 0.8) -> np.ndarray:
        """Generate realistic vegetation spectral signature."""
        reflectance = np.zeros_like(wavelengths)
        
        # Visible region (400-700 nm): low reflectance with chlorophyll absorption
        visible_mask = (wavelengths >= 400) & (wavelengths <= 700)
        reflectance[visible_mask] = 0.05 + 0.03 * health
        
        # Green peak around 550 nm
        green_mask = (wavelengths >= 520) & (wavelengths <= 580)
        reflectance[green_mask] += 0.05 * health
        
        # Red edge (700-750 nm): steep increase
        red_edge_mask = (wavelengths >= 700) & (wavelengths <= 750)
        red_edge_factor = (wavelengths[red_edge_mask] - 700) / 50
        reflectance[red_edge_mask] = 0.05 + 0.4 * health * red_edge_factor
        
        # NIR region (750-1300 nm): high reflectance
        nir_mask = (wavelengths >= 750) & (wavelengths <= 1300)
        reflectance[nir_mask] = 0.45 * health
        
        # SWIR region (1300-2500 nm): moderate reflectance with water absorption
        swir_mask = wavelengths >= 1300
        reflectance[swir_mask] = 0.2 * health
        
        # Water absorption bands
        water_bands = [(1400, 1500), (1900, 2000)]
        for start, end in water_bands:
            water_mask = (wavelengths >= start) & (wavelengths <= end)
            reflectance[water_mask] *= 0.3
        
        return reflectance
    
    def _soil_signature(self, wavelengths: np.ndarray) -> np.ndarray:
        """Generate realistic soil spectral signature."""
        # Soil has generally increasing reflectance with wavelength
        base_reflectance = 0.1 + 0.3 * (wavelengths - 400) / (2500 - 400)
        
        # Add some spectral features
        noise = 0.02 * np.sin(2 * np.pi * wavelengths / 200)
        
        return np.clip(base_reflectance + noise, 0, 1)
    
    def process_image_pipeline(self, 
                             img_array: np.ndarray,
                             apply_radiometric: bool = True,
                             apply_atmospheric: bool = True,
                             apply_noise_reduction: bool = True,
                             apply_normalization: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Complete image preprocessing pipeline.
        
        Args:
            img_array: Input image array
            apply_radiometric: Apply radiometric correction
            apply_atmospheric: Apply atmospheric correction
            apply_noise_reduction: Apply noise reduction
            apply_normalization: Apply spectral normalization
            
        Returns:
            Processed image and metadata
        """
        logger.info("Starting complete image preprocessing pipeline...")
        
        processed_img = img_array.copy()
        metadata = {}
        
        # Step 1: Radiometric correction
        if apply_radiometric:
            processed_img = self.radiometric_correction(processed_img)
            metadata['radiometric_correction'] = True
        
        # Step 2: Atmospheric correction
        if apply_atmospheric:
            processed_img = self.atmospheric_correction_dos(processed_img)
            metadata['atmospheric_correction'] = True
        
        # Step 3: Noise reduction
        if apply_noise_reduction:
            processed_img = self.noise_reduction(processed_img, method='gaussian')
            metadata['noise_reduction'] = 'gaussian'
        
        # Step 4: Spectral normalization
        if apply_normalization:
            processed_img, scaler = self.spectral_normalization(processed_img, method='minmax')
            metadata['normalization'] = 'minmax'
            metadata['scaler'] = scaler
        
        logger.info("Image preprocessing pipeline completed")
        return processed_img, metadata


def main():
    """Main function to demonstrate image preprocessing."""
    preprocessor = ImagePreprocessor()
    
    # Create sample hyperspectral image
    sample_img = preprocessor.create_sample_hyperspectral_image(100, 100, 200)
    
    # Apply preprocessing pipeline
    processed_img, metadata = preprocessor.process_image_pipeline(sample_img)
    
    # Save processed image
    output_path = preprocessor.processed_dir / 'sample_processed_hyperspectral.npy'
    np.save(output_path, processed_img)
    
    logger.info(f"Processed image saved to: {output_path}")
    logger.info(f"Original shape: {sample_img.shape}")
    logger.info(f"Processed shape: {processed_img.shape}")
    logger.info(f"Processing metadata: {metadata}")


if __name__ == "__main__":
    main()