#!/usr/bin/env python3
"""
Demo script showing how to use the NASA TEMPO downloader
"""

import os
import sys
from pathlib import Path


def show_setup_instructions():
    """Display setup instructions"""
    print("🚀 NASA TEMPO Data Downloader - Setup Instructions")
    print("=" * 60)
    
    print("\n📋 STEP 1: Get NASA API Key")
    print("-" * 30)
    print("1. Visit: https://api.nasa.gov/")
    print("2. Click 'Get Started'")
    print("3. Fill out the form to get your free API key")
    print("4. Copy your API key")
    
    print("\n📋 STEP 2: Configure AWS Credentials (Optional)")
    print("-" * 40)
    print("Option A - Environment Variables:")
    print("  export AWS_ACCESS_KEY_ID='your_access_key'")
    print("  export AWS_SECRET_ACCESS_KEY='your_secret_key'")
    print("  export S3_BUCKET='your-bucket-name'")
    print("\nOption B - AWS CLI:")
    print("  aws configure")
    print("\nOption C - IAM Roles (for EC2/ECS)")
    print("  No additional setup needed")
    
    print("\n📋 STEP 3: Update Configuration")
    print("-" * 30)
    print("Edit config.json and replace placeholder values:")
    print("  - nasa_api_key: Your NASA API key")
    print("  - aws_access_key_id: Your AWS access key")
    print("  - aws_secret_access_key: Your AWS secret key")
    print("  - s3_bucket: Your S3 bucket name")
    
    print("\n📋 STEP 4: Run the Downloader")
    print("-" * 30)
    print("Basic usage:")
    print("  python3 nasa_tempo_downloader.py")
    print("\nWith date range:")
    print("  python3 nasa_tempo_downloader.py --start-date 2024-01-01 --end-date 2024-01-07")
    print("\nAdvanced version:")
    print("  python3 nasa_tempo_advanced.py")


def show_file_structure():
    """Show the project file structure"""
    print("\n📁 Project Structure")
    print("=" * 20)
    
    files = [
        ("nasa_tempo_downloader.py", "Main downloader script (simple version)"),
        ("nasa_tempo_advanced.py", "Advanced downloader with Earthdata API"),
        ("config.json", "Configuration file with API keys"),
        ("requirements.txt", "Python dependencies"),
        ("setup.py", "Automated setup script"),
        ("test_setup.py", "Test script to verify setup"),
        ("example_usage.py", "Usage examples"),
        ("README.md", "Complete documentation"),
        ("downloads/", "Directory for downloaded data (created automatically)"),
        ("*.log", "Log files (created automatically)")
    ]
    
    for filename, description in files:
        print(f"  {filename:<25} - {description}")


def show_quick_test():
    """Show how to run a quick test"""
    print("\n🧪 Quick Test")
    print("=" * 12)
    print("Test your setup:")
    print("  python3 test_setup.py")
    print("\nRun example (with placeholder data):")
    print("  python3 example_usage.py")


def show_troubleshooting():
    """Show troubleshooting tips"""
    print("\n🔧 Troubleshooting")
    print("=" * 18)
    
    issues = [
        ("403 Forbidden", "Check your NASA API key"),
        ("AWS credentials error", "Verify AWS credentials and permissions"),
        ("No data found", "Check date range and data availability"),
        ("Network timeout", "Check internet connection and firewall"),
        ("S3 upload failed", "Verify S3 bucket exists and is accessible")
    ]
    
    for issue, solution in issues:
        print(f"  {issue:<20} → {solution}")


def main():
    """Main demo function"""
    show_setup_instructions()
    show_file_structure()
    show_quick_test()
    show_troubleshooting()
    
    print("\n" + "=" * 60)
    print("🎉 You're ready to download NASA TEMPO data!")
    print("Start with: python3 nasa_tempo_downloader.py --help")


if __name__ == "__main__":
    main()