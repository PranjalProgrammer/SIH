"""
Image Preprocessing Module
Handles preprocessing of hyperspectral and multispectral images
"""

import numpy as np
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for agricultural data"""
    
    def __init__(self):
        self.processed_count = 0
    
    def normalize_image(self, image):
        """Normalize image values to 0-1 range"""
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def resize_image(self, image, target_size=(224, 224)):
        """Resize image to target dimensions"""
        if len(image.shape) == 3:
            return cv2.resize(image, target_size)
        else:
            # Handle hyperspectral data
            resized_bands = []
            for band in range(image.shape[2]):
                resized_band = cv2.resize(image[:, :, band], target_size)
                resized_bands.append(resized_band)
            return np.stack(resized_bands, axis=2)
    
    def preprocess_batch(self, image_paths, output_dir):
        """Preprocess a batch of images"""
        logger.info(f"Preprocessing {len(image_paths)} images...")
        
        processed_images = []
        for path in image_paths:
            try:
                # Load image
                if path.endswith(('.tif', '.tiff')):
                    # Handle hyperspectral TIFF
                    image = np.array(Image.open(path))
                else:
                    image = cv2.imread(path)
                
                # Preprocess
                image = self.normalize_image(image)
                image = self.resize_image(image)
                
                processed_images.append(image)
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        
        logger.info(f"Successfully preprocessed {len(processed_images)} images")
        return processed_images

def main():
    """Main preprocessing function"""
    preprocessor = ImagePreprocessor()
    logger.info("Image preprocessor initialized")

if __name__ == "__main__":
    main()