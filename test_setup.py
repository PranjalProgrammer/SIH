#!/usr/bin/env python3
"""
Test script to verify NASA TEMPO downloader setup
"""

import os
import sys
import json
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import requests
        import boto3
        import json
        import logging
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration file"""
    print("Testing configuration...")
    config_file = Path("config.json")
    
    if not config_file.exists():
        print("✗ config.json not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        required_fields = ['nasa_api_key', 'aws_region']
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            print(f"⚠ Missing fields in config: {missing_fields}")
            print("  You can set these via environment variables or update config.json")
        
        print("✓ Configuration file is valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_aws_credentials():
    """Test AWS credentials"""
    print("Testing AWS credentials...")
    try:
        import boto3
        
        # Try to create S3 client
        s3_client = boto3.client('s3')
        
        # Test with a simple operation
        s3_client.list_buckets()
        print("✓ AWS credentials are working")
        return True
        
    except Exception as e:
        print(f"⚠ AWS credentials issue: {e}")
        print("  This is normal if you haven't configured AWS credentials yet")
        return False


def test_nasa_api():
    """Test NASA API key"""
    print("Testing NASA API...")
    try:
        import requests
        
        # Load config
        with open("config.json", 'r') as f:
            config = json.load(f)
        
        api_key = config.get('nasa_api_key') or os.getenv('NASA_API_KEY')
        
        if not api_key:
            print("⚠ No NASA API key found")
            return False
        
        # Test API call
        response = requests.get(
            "https://api.nasa.gov/planetary/apod",
            params={'api_key': api_key},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✓ NASA API key is working")
            return True
        else:
            print(f"✗ NASA API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠ NASA API test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("NASA TEMPO Downloader - Setup Test")
    print("=" * 40)
    
    tests = [
        ("Package imports", test_imports),
        ("Configuration", test_config),
        ("AWS credentials", test_aws_credentials),
        ("NASA API", test_nasa_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to use the downloader.")
    else:
        print("⚠ Some tests failed. Check the configuration and try again.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)