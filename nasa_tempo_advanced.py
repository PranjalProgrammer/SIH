#!/usr/bin/env python3
"""
Advanced NASA TEMPO Data Downloader
Includes support for NASA Earthdata API and actual TEMPO data access
"""

import os
import sys
import json
import logging
import requests
import boto3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tempo_advanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AdvancedNASATempoDownloader:
    """Advanced NASA TEMPO data downloader with Earthdata API support"""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize the advanced downloader"""
        self.config = self._load_config(config_file)
        self.session = requests.Session()
        self.s3_client = None
        self._setup_s3()
        self._setup_earthdata_auth()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_file}")
                return config
            else:
                logger.warning(f"Config file {config_file} not found, using environment variables")
                return self._get_env_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_env_config()
    
    def _get_env_config(self) -> Dict[str, Any]:
        """Get configuration from environment variables"""
        return {
            "nasa_api_key": os.getenv("NASA_API_KEY"),
            "earthdata_username": os.getenv("EARTHDATA_USERNAME"),
            "earthdata_password": os.getenv("EARTHDATA_PASSWORD"),
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region": os.getenv("AWS_REGION", "us-east-1"),
            "s3_bucket": os.getenv("S3_BUCKET"),
            "download_dir": os.getenv("DOWNLOAD_DIR", "./downloads"),
            "date_range_days": int(os.getenv("DATE_RANGE_DAYS", "7")),
            "tempo_data_types": os.getenv("TEMPO_DATA_TYPES", "NO2,O3,HCHO").split(",")
        }
    
    def _setup_s3(self):
        """Setup AWS S3 client"""
        try:
            aws_config = {
                'region_name': self.config.get('aws_region', 'us-east-1')
            }
            
            if self.config.get('aws_access_key_id') and self.config.get('aws_secret_access_key'):
                aws_config.update({
                    'aws_access_key_id': self.config['aws_access_key_id'],
                    'aws_secret_access_key': self.config['aws_secret_access_key']
                })
                logger.info("Using explicit AWS credentials")
            else:
                logger.info("Using default AWS credential chain")
            
            self.s3_client = boto3.client('s3', **aws_config)
            
            if self.config.get('s3_bucket'):
                self.s3_client.head_bucket(Bucket=self.config['s3_bucket'])
                logger.info(f"S3 connection successful to bucket: {self.config['s3_bucket']}")
            else:
                logger.warning("No S3 bucket specified")
                
        except Exception as e:
            logger.error(f"S3 setup failed: {e}")
            self.s3_client = None
    
    def _setup_earthdata_auth(self):
        """Setup NASA Earthdata authentication"""
        username = self.config.get('earthdata_username')
        password = self.config.get('earthdata_password')
        
        if username and password:
            self.session.auth = (username, password)
            logger.info("NASA Earthdata authentication configured")
        else:
            logger.warning("NASA Earthdata credentials not provided - some data may not be accessible")
    
    def search_tempo_data(self, start_date: str, end_date: str, data_type: str = "NO2") -> List[Dict]:
        """Search for TEMPO data using NASA Earthdata API"""
        try:
            # NASA Earthdata CMR (Common Metadata Repository) API
            cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
            
            params = {
                'collection_concept_id': 'C1234567890-NASA_MAAP',  # TEMPO collection ID (example)
                'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
                'page_size': 2000,
                'sort_key': 'temporal'
            }
            
            logger.info(f"Searching for TEMPO {data_type} data from {start_date} to {end_date}")
            
            response = self.session.get(cmr_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            granules = data.get('feed', {}).get('entry', [])
            
            logger.info(f"Found {len(granules)} granules")
            return granules
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def download_tempo_granule(self, granule_url: str, filename: str) -> bool:
        """Download a single TEMPO data granule"""
        try:
            download_dir = Path(self.config.get('download_dir', './downloads'))
            download_dir.mkdir(exist_ok=True)
            
            filepath = download_dir / filename
            
            logger.info(f"Downloading {filename}...")
            
            response = self.session.get(granule_url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {filename} ({filepath.stat().st_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            return False
    
    def download_tempo_data_advanced(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Download TEMPO data using advanced methods"""
        if not self._validate_config():
            return False
        
        # Set default date range
        if not start_date:
            end_date_obj = datetime.now()
            start_date_obj = end_date_obj - timedelta(days=self.config.get('date_range_days', 7))
            start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')
        
        logger.info(f"Advanced TEMPO data download from {start_date} to {end_date}")
        
        success_count = 0
        total_count = 0
        
        # Download different TEMPO data types
        data_types = self.config.get('tempo_data_types', ['NO2', 'O3', 'HCHO'])
        
        for data_type in data_types:
            logger.info(f"Processing {data_type} data...")
            
            # Search for granules
            granules = self.search_tempo_data(start_date, end_date, data_type)
            
            if not granules:
                logger.warning(f"No {data_type} granules found")
                continue
            
            # Download granules
            for granule in granules[:10]:  # Limit to first 10 granules per type
                total_count += 1
                
                # Extract download URL (this would be specific to the actual API response)
                download_url = granule.get('links', [{}])[0].get('href', '')
                
                if not download_url:
                    logger.warning("No download URL found in granule")
                    continue
                
                # Generate filename
                filename = f"tempo_{data_type}_{granule.get('id', 'unknown')}.nc"
                
                # Download granule
                if self.download_tempo_granule(download_url, filename):
                    success_count += 1
                    
                    # Upload to S3 if configured
                    if self.s3_client and self.config.get('s3_bucket'):
                        self._upload_to_s3(Path(self.config.get('download_dir', './downloads')) / filename, filename)
                
                # Rate limiting
                time.sleep(1)
        
        logger.info(f"Download complete: {success_count}/{total_count} files downloaded successfully")
        return success_count > 0
    
    def _validate_config(self) -> bool:
        """Validate configuration"""
        required_fields = ['nasa_api_key']
        missing_fields = [field for field in required_fields if not self.config.get(field)]
        
        if missing_fields:
            logger.error(f"Missing required configuration: {missing_fields}")
            return False
        
        return True
    
    def _upload_to_s3(self, filepath: Path, filename: str) -> bool:
        """Upload file to S3"""
        try:
            bucket = self.config['s3_bucket']
            s3_key = f"tempo-data/{filename}"
            
            logger.info(f"Uploading {filename} to S3")
            
            self.s3_client.upload_file(
                str(filepath),
                bucket,
                s3_key,
                ExtraArgs={'ContentType': 'application/netcdf'}
            )
            
            logger.info(f"Uploaded to s3://{bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def download_and_upload_advanced(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Main method for advanced download and upload"""
        logger.info("Starting advanced NASA TEMPO data download")
        
        success = self.download_tempo_data_advanced(start_date, end_date)
        
        if success:
            logger.info("Advanced download process completed successfully")
        else:
            logger.error("Advanced download process failed")
        
        return success


def main():
    """Main function for advanced downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced NASA TEMPO data downloader')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-types', help='Comma-separated data types (NO2,O3,HCHO)')
    
    args = parser.parse_args()
    
    try:
        downloader = AdvancedNASATempoDownloader(args.config)
        
        # Override data types if provided
        if args.data_types:
            downloader.config['tempo_data_types'] = args.data_types.split(',')
        
        success = downloader.download_and_upload_advanced(args.start_date, args.end_date)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()