"""
Dataset Download Module

Downloads and organizes public hyperspectral datasets and IoT sensor data
for the precision agriculture platform.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads and manages agricultural datasets from various sources."""
    
    def __init__(self, data_dir: str = "/workspace/data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and metadata
        self.datasets = {
            'proximal_hyperspectral': {
                'url': 'https://zenodo.org/record/4767453/files/proximal_hyperspectral_crop_dataset.zip',
                'description': 'Proximal Hyperspectral Crop Dataset',
                'type': 'hyperspectral'
            },
            'sentinel2_sample': {
                'url': 'https://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/33/T/UG/S2A_MSIL1C_20210615T100031_N0300_R122_T33TUG_20210615T120304.SAFE.zip',
                'description': 'Sentinel-2 Sample Data',
                'type': 'multispectral'
            },
            'iot_sensor_sample': {
                'url': 'https://raw.githubusercontent.com/agricultural-iot/sample-data/main/sensor_data.csv',
                'description': 'Sample IoT Sensor Data',
                'type': 'sensor'
            }
        }
    
    def download_file(self, url: str, filename: str, chunk_size: int = 8192) -> bool:
        """Download a file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = self.data_dir / filename
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {str(e)}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None) -> bool:
        """Extract zip or tar archives."""
        if extract_to is None:
            extract_to = archive_path.parent / archive_path.stem
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.bz2']:
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            logger.info(f"Extracted: {archive_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path.name}: {str(e)}")
            return False
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a specific dataset."""
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        url = dataset_info['url']
        filename = url.split('/')[-1]
        
        logger.info(f"Downloading {dataset_info['description']}...")
        
        if self.download_file(url, filename):
            file_path = self.data_dir / filename
            
            # Extract if it's an archive
            if filename.endswith(('.zip', '.tar', '.tar.gz', '.tar.bz2')):
                self.extract_archive(file_path)
            
            return True
        
        return False
    
    def download_all_datasets(self) -> Dict[str, bool]:
        """Download all available datasets."""
        results = {}
        
        for dataset_name in self.datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            results[dataset_name] = self.download_dataset(dataset_name)
        
        return results
    
    def create_sample_iot_data(self) -> None:
        """Create sample IoT sensor data if real data is not available."""
        logger.info("Creating sample IoT sensor data...")
        
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate synthetic sensor data
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(hours=x) for x in range(720)]  # 30 days, hourly
        
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'soil_moisture': np.random.normal(45, 10, len(dates)).clip(0, 100),
            'soil_temperature': np.random.normal(22, 5, len(dates)).clip(-10, 50),
            'air_temperature': np.random.normal(25, 8, len(dates)).clip(-20, 50),
            'humidity': np.random.normal(65, 15, len(dates)).clip(0, 100),
            'leaf_wetness': np.random.exponential(2, len(dates)).clip(0, 24),
            'light_intensity': np.random.gamma(2, 2, len(dates)) * 1000,
            'wind_speed': np.random.exponential(1.5, len(dates)).clip(0, 20),
            'rainfall': np.random.exponential(0.5, len(dates)).clip(0, 50)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.data_dir / 'sample_iot_sensor_data.csv', index=False)
        logger.info("Sample IoT data created successfully")
    
    def create_sample_hyperspectral_metadata(self) -> None:
        """Create sample hyperspectral image metadata."""
        logger.info("Creating sample hyperspectral metadata...")
        
        metadata = {
            'image_id': [f'IMG_{i:04d}' for i in range(100)],
            'crop_type': np.random.choice(['wheat', 'corn', 'soybean', 'rice'], 100),
            'growth_stage': np.random.choice(['seedling', 'vegetative', 'flowering', 'maturity'], 100),
            'health_status': np.random.choice(['healthy', 'stressed', 'diseased'], 100, p=[0.6, 0.3, 0.1]),
            'acquisition_date': pd.date_range('2023-01-01', periods=100, freq='3D'),
            'latitude': np.random.uniform(40.0, 45.0, 100),
            'longitude': np.random.uniform(-95.0, -85.0, 100),
            'altitude': np.random.uniform(200, 500, 100)
        }
        
        df = pd.DataFrame(metadata)
        df.to_csv(self.data_dir / 'hyperspectral_metadata.csv', index=False)
        logger.info("Sample hyperspectral metadata created successfully")


def main():
    """Main function to download all datasets."""
    downloader = DatasetDownloader()
    
    # Download datasets
    results = downloader.download_all_datasets()
    
    # Create sample data if downloads fail
    if not any(results.values()):
        logger.warning("Dataset downloads failed, creating sample data...")
        downloader.create_sample_iot_data()
        downloader.create_sample_hyperspectral_metadata()
    
    # Print results
    print("\nDownload Results:")
    for dataset, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{dataset}: {status}")


if __name__ == "__main__":
    main()