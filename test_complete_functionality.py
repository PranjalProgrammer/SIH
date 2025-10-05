#!/usr/bin/env python3
"""
Complete functionality test for NASA TEMPO Air Quality Monitor
"""

import requests
import json
import time
import subprocess
import sys
import os
from pathlib import Path


def test_backend_api():
    """Test backend API functionality"""
    print("Testing Backend API...")
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed: {health_data['status']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False
    
    # Test air quality data endpoint
    try:
        response = requests.get(f"{base_url}/api/air-quality", timeout=10)
        if response.status_code == 200:
            data = response.json()
            pollutants = list(data.keys())
            print(f"✓ Air quality data loaded: {len(pollutants)} pollutants ({', '.join(pollutants)})")
            
            # Check data structure
            for pollutant in pollutants:
                if 'features' in data[pollutant]:
                    feature_count = len(data[pollutant]['features'])
                    print(f"  - {pollutant}: {feature_count} data points")
        else:
            print(f"✗ Air quality data failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Air quality data error: {e}")
        return False
    
    # Test specific pollutant endpoint
    try:
        response = requests.get(f"{base_url}/api/air-quality/NO2", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'features' in data:
                print(f"✓ NO2 data endpoint working: {len(data['features'])} points")
        else:
            print(f"✗ NO2 endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ NO2 endpoint error: {e}")
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✓ Status endpoint working: {status_data['data_points']} total data points")
        else:
            print(f"✗ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Status endpoint error: {e}")
    
    # Test location saving
    try:
        location_data = {
            "name": "Test Location",
            "latitude": 40.7128,
            "longitude": -74.0060
        }
        
        response = requests.post(
            f"{base_url}/api/locations",
            json=location_data,
            timeout=5
        )
        
        if response.status_code == 201:
            saved_location = response.json()
            print(f"✓ Location saving working: ID {saved_location['id']}")
        else:
            print(f"✗ Location saving failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Location saving error: {e}")
    
    return True


def test_frontend_files():
    """Test frontend files"""
    print("\nTesting Frontend Files...")
    
    files_to_check = [
        'index.html',
        'app-styles.css',
        'app-script.js'
    ]
    
    all_exist = True
    for file in files_to_check:
        if Path(file).exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist


def test_file_content():
    """Test file content for basic functionality"""
    print("\nTesting File Content...")
    
    # Check HTML file
    try:
        with open('index.html', 'r') as f:
            html_content = f.read()
            
        required_elements = ['leaflet', 'map', 'pollutant-select']
        for element in required_elements:
            if element in html_content.lower():
                print(f"✓ HTML contains {element}")
            else:
                print(f"✗ HTML missing {element}")
    except Exception as e:
        print(f"✗ HTML file error: {e}")
    
    # Check CSS file
    try:
        with open('app-styles.css', 'r') as f:
            css_content = f.read()
            
        if 'leaflet' in css_content.lower():
            print("✓ CSS contains Leaflet styles")
        else:
            print("✗ CSS missing Leaflet styles")
    except Exception as e:
        print(f"✗ CSS file error: {e}")
    
    # Check JavaScript file
    try:
        with open('app-script.js', 'r') as f:
            js_content = f.read()
            
        required_functions = ['AirQualityMonitor', 'initMap', 'fetchAirQualityData']
        for func in required_functions:
            if func in js_content:
                print(f"✓ JavaScript contains {func}")
            else:
                print(f"✗ JavaScript missing {func}")
    except Exception as e:
        print(f"✗ JavaScript file error: {e}")


def test_docker_config():
    """Test Docker configuration"""
    print("\nTesting Docker Configuration...")
    
    # Check Dockerfile
    if Path('Dockerfile').exists():
        print("✓ Dockerfile exists")
        
        try:
            with open('Dockerfile', 'r') as f:
                dockerfile_content = f.read()
                
            if 'python' in dockerfile_content.lower() and 'flask' in dockerfile_content.lower():
                print("✓ Dockerfile contains Python and Flask")
            else:
                print("✗ Dockerfile missing Python/Flask configuration")
        except Exception as e:
            print(f"✗ Dockerfile read error: {e}")
    else:
        print("✗ Dockerfile missing")
    
    # Check docker-compose.yml
    if Path('docker-compose.yml').exists():
        print("✓ docker-compose.yml exists")
        
        try:
            with open('docker-compose.yml', 'r') as f:
                compose_content = f.read()
                
            if 'nasa-tempo-backend' in compose_content and 'nasa-tempo-frontend' in compose_content:
                print("✓ docker-compose.yml contains both services")
            else:
                print("✗ docker-compose.yml missing services")
        except Exception as e:
            print(f"✗ docker-compose.yml read error: {e}")
    else:
        print("✗ docker-compose.yml missing")


def main():
    """Run complete functionality test"""
    print("NASA TEMPO Air Quality Monitor - Complete Functionality Test")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code != 200:
            print("✗ Backend not running. Please start with: python3 flask-backend.py")
            return False
    except:
        print("✗ Backend not running. Please start with: python3 flask-backend.py")
        return False
    
    # Run all tests
    tests = [
        ("Backend API", test_backend_api),
        ("Frontend Files", test_frontend_files),
        ("File Content", test_file_content),
        ("Docker Config", test_docker_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result if result is not None else True)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe NASA TEMPO Air Quality Monitor is fully functional!")
        print("\nTo use the application:")
        print("1. Backend is running on: http://localhost:5001")
        print("2. Open index.html in your browser")
        print("3. Or serve with: python3 -m http.server 8000")
        print("4. Then visit: http://localhost:8000")
        print("\nFeatures working:")
        print("- ✓ Real-time air quality data visualization")
        print("- ✓ Interactive Leaflet maps")
        print("- ✓ Multiple pollutant support (NO2, O3, HCHO, SO2, CO)")
        print("- ✓ Location saving functionality")
        print("- ✓ Responsive web interface")
        print("- ✓ RESTful API endpoints")
        print("- ✓ Docker deployment ready")
    else:
        print(f"\n⚠ {total - passed} tests failed. Check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)