#!/usr/bin/env python3
"""
Example usage of NASA TEMPO Data Downloader
"""

import os
import sys
from nasa_tempo_downloader import NASATempoDownloader


def example_with_env_vars():
    """Example using environment variables"""
    print("Example: Using environment variables")
    print("=" * 40)
    
    # Set example environment variables (replace with real values)
    os.environ['NASA_API_KEY'] = 'DEMO_KEY'  # Replace with real NASA API key
    os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'  # Replace with real AWS key
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'  # Replace with real AWS secret
    os.environ['S3_BUCKET'] = 'your-bucket-name'  # Replace with real bucket name
    
    try:
        # Initialize downloader (will use environment variables)
        downloader = NASATempoDownloader()
        
        # Download data for last 3 days
        success = downloader.download_and_upload()
        
        if success:
            print("✓ Data downloaded and uploaded successfully!")
        else:
            print("✗ Download/upload failed")
            
    except Exception as e:
        print(f"Error: {e}")


def example_with_config_file():
    """Example using config file"""
    print("\nExample: Using config file")
    print("=" * 40)
    
    try:
        # Initialize downloader with config file
        downloader = NASATempoDownloader("config.json")
        
        # Download data for specific date range
        success = downloader.download_and_upload(
            start_date="2024-01-01",
            end_date="2024-01-03"
        )
        
        if success:
            print("✓ Data downloaded and uploaded successfully!")
        else:
            print("✗ Download/upload failed")
            
    except Exception as e:
        print(f"Error: {e}")


def example_local_download_only():
    """Example downloading only (no S3 upload)"""
    print("\nExample: Local download only")
    print("=" * 40)
    
    try:
        # Initialize downloader without S3 configuration
        downloader = NASATempoDownloader()
        
        # Remove S3 bucket from config to skip upload
        if 's3_bucket' in downloader.config:
            del downloader.config['s3_bucket']
        
        # Download data locally only
        success = downloader.download_tempo_data()
        
        if success:
            print("✓ Data downloaded successfully!")
            print("Check the 'downloads' directory for your files")
        else:
            print("✗ Download failed")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run examples"""
    print("NASA TEMPO Data Downloader - Usage Examples")
    print("=" * 50)
    
    print("\nNote: These examples use placeholder values.")
    print("Replace with your actual API keys and credentials before running.")
    
    # Uncomment the example you want to run:
    
    # example_with_env_vars()
    # example_with_config_file()
    example_local_download_only()
    
    print("\n" + "=" * 50)
    print("To use with real data:")
    print("1. Get NASA API key from: https://api.nasa.gov/")
    print("2. Configure AWS credentials")
    print("3. Update config.json or set environment variables")
    print("4. Run: python3 nasa_tempo_downloader.py")


if __name__ == "__main__":
    main()