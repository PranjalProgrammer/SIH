#!/usr/bin/env python3
"""
Setup script for NASA TEMPO Data Downloader
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def setup_config():
    """Setup configuration file with user input"""
    config_file = Path("config.json")
    
    if config_file.exists():
        print("Configuration file already exists.")
        response = input("Do you want to update it? (y/n): ").lower().strip()
        if response != 'y':
            return True
    
    print("\nSetting up configuration...")
    print("You can get a NASA API key from: https://api.nasa.gov/")
    print("AWS credentials can be found in your AWS Console or use IAM roles")
    
    config = {}
    
    # NASA API Key
    nasa_key = input("Enter your NASA API key (or press Enter to skip): ").strip()
    if nasa_key:
        config["nasa_api_key"] = nasa_key
    
    # AWS Credentials
    aws_access_key = input("Enter AWS Access Key ID (or press Enter to skip): ").strip()
    if aws_access_key:
        config["aws_access_key_id"] = aws_access_key
    
    aws_secret_key = input("Enter AWS Secret Access Key (or press Enter to skip): ").strip()
    if aws_secret_key:
        config["aws_secret_access_key"] = aws_secret_key
    
    # AWS Region
    aws_region = input("Enter AWS Region (default: us-east-1): ").strip()
    config["aws_region"] = aws_region if aws_region else "us-east-1"
    
    # S3 Bucket
    s3_bucket = input("Enter S3 bucket name (or press Enter to skip): ").strip()
    if s3_bucket:
        config["s3_bucket"] = s3_bucket
    
    # Download directory
    download_dir = input("Enter download directory (default: ./downloads): ").strip()
    config["download_dir"] = download_dir if download_dir else "./downloads"
    
    # Date range
    date_range = input("Enter default date range in days (default: 7): ").strip()
    try:
        config["date_range_days"] = int(date_range) if date_range else 7
    except ValueError:
        config["date_range_days"] = 7
    
    # Save configuration
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {config_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to save configuration: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    try:
        Path("downloads").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        print("✓ Directories created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create directories: {e}")
        return False


def test_installation():
    """Test the installation"""
    print("Testing installation...")
    try:
        import requests
        import boto3
        print("✓ All required packages are available")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        return False


def main():
    """Main setup function"""
    print("NASA TEMPO Data Downloader Setup")
    print("=" * 40)
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Creating directories", create_directories),
        ("Setting up configuration", setup_config),
        ("Testing installation", test_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"Setup failed at: {step_name}")
            sys.exit(1)
    
    print("\n" + "=" * 40)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update config.json with your actual API keys and credentials")
    print("2. Run: python nasa_tempo_downloader.py --help")
    print("3. Run: python nasa_tempo_downloader.py")


if __name__ == "__main__":
    main()