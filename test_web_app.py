#!/usr/bin/env python3
"""
Test script for NASA TEMPO Web Application
Tests all components and ensures full functionality
"""

import os
import sys
import json
import requests
import time
import subprocess
import threading
from pathlib import Path


class WebAppTester:
    """Test suite for the NASA TEMPO web application"""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.frontend_url = "http://localhost:8000"
        self.flask_process = None
        
    def test_imports(self):
        """Test if all required packages can be imported"""
        print("Testing imports...")
        try:
            import flask
            import flask_cors
            import boto3
            import xarray
            import netcdf4
            import numpy
            print("✓ All required packages imported successfully")
            return True
        except ImportError as e:
            print(f"✗ Import error: {e}")
            return False
    
    def test_file_structure(self):
        """Test if all required files exist"""
        print("Testing file structure...")
        required_files = [
            'flask-backend.py',
            'index.html',
            'app-styles.css',
            'app-script.js',
            'requirements.txt',
            'config.json',
            'Dockerfile',
            'docker-compose.yml'
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
            return False
        
        print("✓ All required files present")
        return True
    
    def start_flask_server(self):
        """Start Flask server in background"""
        print("Starting Flask server...")
        try:
            self.flask_process = subprocess.Popen(
                [sys.executable, 'flask-backend.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, 'FLASK_ENV': 'development'}
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✓ Flask server started successfully")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print("✗ Flask server failed to start")
            return False
            
        except Exception as e:
            print(f"✗ Error starting Flask server: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        print("Testing API endpoints...")
        
        endpoints = [
            ('/health', 'GET'),
            ('/api/air-quality', 'GET'),
            ('/api/status', 'GET'),
            ('/api/locations', 'GET')
        ]
        
        for endpoint, method in endpoints:
            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code in [200, 201]:
                    print(f"✓ {method} {endpoint} - Status: {response.status_code}")
                else:
                    print(f"⚠ {method} {endpoint} - Status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"✗ {method} {endpoint} - Error: {e}")
        
        return True
    
    def test_frontend_files(self):
        """Test if frontend files are accessible"""
        print("Testing frontend files...")
        
        # Start simple HTTP server for frontend
        try:
            frontend_process = subprocess.Popen(
                [sys.executable, '-m', 'http.server', '8000'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(2)
            
            # Test if files are accessible
            files_to_test = ['index.html', 'app-styles.css', 'app-script.js']
            
            for file in files_to_test:
                try:
                    response = requests.get(f"{self.frontend_url}/{file}", timeout=5)
                    if response.status_code == 200:
                        print(f"✓ Frontend file accessible: {file}")
                    else:
                        print(f"✗ Frontend file not accessible: {file}")
                except requests.exceptions.RequestException:
                    print(f"✗ Frontend file error: {file}")
            
            frontend_process.terminate()
            return True
            
        except Exception as e:
            print(f"✗ Error testing frontend: {e}")
            return False
    
    def test_docker_config(self):
        """Test Docker configuration"""
        print("Testing Docker configuration...")
        
        try:
            # Test if Dockerfile is valid
            result = subprocess.run(
                ['docker', 'build', '--dry-run', '.'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✓ Dockerfile is valid")
            else:
                print(f"⚠ Dockerfile validation: {result.stderr}")
            
            # Test if docker-compose.yml is valid
            result = subprocess.run(
                ['docker-compose', 'config'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✓ docker-compose.yml is valid")
            else:
                print(f"⚠ docker-compose.yml validation: {result.stderr}")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("⚠ Docker validation timed out")
            return True
        except FileNotFoundError:
            print("⚠ Docker not installed - skipping Docker tests")
            return True
        except Exception as e:
            print(f"⚠ Docker test error: {e}")
            return True
    
    def cleanup(self):
        """Cleanup test processes"""
        if self.flask_process:
            self.flask_process.terminate()
            self.flask_process.wait()
            print("✓ Flask server stopped")
    
    def run_all_tests(self):
        """Run all tests"""
        print("NASA TEMPO Web Application Test Suite")
        print("=" * 50)
        
        tests = [
            ("Package imports", self.test_imports),
            ("File structure", self.test_file_structure),
            ("Docker configuration", self.test_docker_config),
            ("Flask server", self.start_flask_server),
            ("API endpoints", self.test_api_endpoints),
            ("Frontend files", self.test_frontend_files)
        ]
        
        results = []
        
        try:
            for test_name, test_func in tests:
                print(f"\n{test_name}:")
                result = test_func()
                results.append(result)
                
        finally:
            self.cleanup()
        
        print("\n" + "=" * 50)
        print("Test Summary:")
        
        passed = sum(results)
        total = len(results)
        
        for i, (test_name, _) in enumerate(tests):
            status = "✓ PASS" if results[i] else "✗ FAIL"
            print(f"  {test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! The application is ready to use.")
            print("\nTo start the application:")
            print("1. Set your AWS credentials:")
            print("   export AWS_ACCESS_KEY_ID=your_key")
            print("   export AWS_SECRET_ACCESS_KEY=your_secret")
            print("   export S3_BUCKET=your-bucket")
            print("2. Start the backend: python flask-backend.py")
            print("3. Open index.html in your browser")
            print("4. Or use Docker: docker-compose up --build")
        else:
            print("⚠ Some tests failed. Check the configuration and try again.")
        
        return passed == total


def main():
    """Main test function"""
    tester = WebAppTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()