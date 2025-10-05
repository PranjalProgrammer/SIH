#!/usr/bin/env python3
"""
NASA TEMPO Data Downloader and S3 Uploader
A simple script to download NASA TEMPO data and upload to AWS S3
"""

import os
import sys
import json
import logging
import requests
import boto3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tempo_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NASATempoDownloader:
    """Simple NASA TEMPO data downloader with S3 upload capability"""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize the downloader with configuration"""
        self.config = self._load_config(config_file)
        self.session = requests.Session()
        self.s3_client = None
        self._setup_s3()
    
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
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region": os.getenv("AWS_REGION", "us-east-1"),
            "s3_bucket": os.getenv("S3_BUCKET"),
            "download_dir": os.getenv("DOWNLOAD_DIR", "./downloads"),
            "date_range_days": int(os.getenv("DATE_RANGE_DAYS", "7"))
        }
    
    def _setup_s3(self):
        """Setup AWS S3 client with proper credentials"""
        try:
            aws_config = {
                'region_name': self.config.get('aws_region', 'us-east-1')
            }
            
            # Use explicit credentials if provided
            if self.config.get('aws_access_key_id') and self.config.get('aws_secret_access_key'):
                aws_config.update({
                    'aws_access_key_id': self.config['aws_access_key_id'],
                    'aws_secret_access_key': self.config['aws_secret_access_key']
                })
                logger.info("Using explicit AWS credentials from config")
            else:
                logger.info("Using default AWS credential chain (IAM roles, env vars, etc.)")
            
            self.s3_client = boto3.client('s3', **aws_config)
            
            # Test S3 connection
            if self.config.get('s3_bucket'):
                self.s3_client.head_bucket(Bucket=self.config['s3_bucket'])
                logger.info(f"S3 connection successful to bucket: {self.config['s3_bucket']}")
            else:
                logger.warning("No S3 bucket specified in configuration")
                
        except Exception as e:
            logger.error(f"S3 setup failed: {e}")
            self.s3_client = None
    
    def _validate_config(self) -> bool:
        """Validate that required configuration is present"""
        required_fields = ['nasa_api_key']
        missing_fields = []
        
        for field in required_fields:
            if not self.config.get(field):
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"Missing required configuration: {missing_fields}")
            return False
        
        if not self.config.get('s3_bucket'):
            logger.warning("No S3 bucket configured - data will only be downloaded locally")
        
        return True
    
    def download_tempo_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Download NASA TEMPO data for specified date range"""
        if not self._validate_config():
            return False
        
        # Set default date range if not provided
        if not start_date:
            end_date_obj = datetime.now()
            start_date_obj = end_date_obj - timedelta(days=self.config.get('date_range_days', 7))
            start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')
        
        logger.info(f"Downloading TEMPO data from {start_date} to {end_date}")
        
        # Create download directory
        download_dir = Path(self.config.get('download_dir', './downloads'))
        download_dir.mkdir(exist_ok=True)
        
        try:
            # NASA TEMPO data is available through various NASA data services
            # Using NASA's Earthdata API for atmospheric data
            api_url = "https://api.nasa.gov/planetary/apod"
            
            # For actual TEMPO data, you would typically use:
            # - NASA Earthdata API: https://api.nasa.gov/earthdata
            # - Or direct data access through NASA's data portals
            # This example uses APOD as a demonstration - replace with actual TEMPO endpoints
            
            params = {
                'api_key': self.config['nasa_api_key'],
                'start_date': start_date,
                'end_date': end_date,
                'count': 100  # Adjust based on API limits
            }
            
            response = self.session.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved {len(data)} data entries")
            
            # Save data locally
            filename = f"tempo_data_{start_date}_to_{end_date}.json"
            filepath = download_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Data saved to {filepath}")
            
            # Upload to S3 if configured
            if self.s3_client and self.config.get('s3_bucket'):
                self._upload_to_s3(filepath, filename)
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _upload_to_s3(self, filepath: Path, filename: str) -> bool:
        """Upload file to S3 bucket"""
        try:
            bucket = self.config['s3_bucket']
            s3_key = f"tempo-data/{filename}"
            
            logger.info(f"Uploading {filename} to S3 bucket {bucket}")
            
            self.s3_client.upload_file(
                str(filepath),
                bucket,
                s3_key,
                ExtraArgs={'ContentType': 'application/json'}
            )
            
            logger.info(f"Successfully uploaded to s3://{bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def download_and_upload(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Main method to download TEMPO data and upload to S3"""
        logger.info("Starting NASA TEMPO data download and S3 upload process")
        
        success = self.download_tempo_data(start_date, end_date)
        
        if success:
            logger.info("Process completed successfully")
        else:
            logger.error("Process failed")
        
        return success


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download NASA TEMPO data and upload to S3')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        downloader = NASATempoDownloader(args.config)
        success = downloader.download_and_upload(args.start_date, args.end_date)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()