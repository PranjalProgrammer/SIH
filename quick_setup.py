#!/usr/bin/env python3
"""
Quick Setup Script for NASA TEMPO Air Quality Monitor
Automatically installs dependencies and tests the setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please install Python 3.9 or higher")
        return False


def install_system_dependencies():
    """Install system dependencies based on OS"""
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Detected Linux system")
        commands = [
            ("sudo apt-get update", "Updating package list"),
            ("sudo apt-get install -y libnetcdf-dev libhdf5-dev pkg-config gcc g++", "Installing NetCDF dependencies")
        ]
    elif system == "darwin":
        print("🍎 Detected macOS system")
        commands = [
            ("brew install netcdf hdf5 pkg-config", "Installing NetCDF dependencies via Homebrew")
        ]
    elif system == "windows":
        print("🪟 Detected Windows system")
        print("⚠️  Windows users: Please install NetCDF4 manually from https://www.lfd.uci.edu/~gohlke/pythonlibs/")
        return True
    else:
        print(f"❓ Unknown system: {system}")
        return True
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def install_python_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    success = run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")
    
    if not success:
        print("⚠️  Trying alternative installation method...")
        # Try installing packages individually
        packages = [
            "Flask==2.3.3",
            "Flask-CORS==4.0.0", 
            "boto3==1.28.57",
            "numpy==1.24.3",
            "pandas==2.0.3",
            "requests==2.31.0",
            "gunicorn==21.2.0"
        ]
        
        for package in packages:
            if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
                print(f"⚠️  Failed to install {package}, continuing...")
        
        # Try NetCDF packages separately
        netcdf_packages = ["xarray==2023.7.0", "netCDF4==1.6.4"]
        for package in netcdf_packages:
            if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
                print(f"⚠️  Failed to install {package} - app will use mock data")
    
    return True


def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    packages_to_test = [
        ("flask", "Flask"),
        ("flask_cors", "Flask-CORS"),
        ("boto3", "boto3"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("requests", "requests")
    ]
    
    optional_packages = [
        ("xarray", "xarray"),
        ("netCDF4", "netCDF4")
    ]
    
    success = True
    for module, name in packages_to_test:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            success = False
    
    # Test optional packages
    for module, name in optional_packages:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"⚠️  {name} not available - will use mock data")
    
    return success


def test_flask_app():
    """Test if Flask app can start"""
    print("🚀 Testing Flask application...")
    
    try:
        # Import the Flask app
        sys.path.insert(0, '.')
        from flask_backend import app
        
        # Test if app can be created
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Flask app test successful")
                return True
            else:
                print(f"❌ Flask app test failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False


def create_startup_script():
    """Create a startup script"""
    print("📝 Creating startup script...")
    
    startup_script = """#!/bin/bash
# NASA TEMPO Air Quality Monitor Startup Script

echo "🚀 Starting NASA TEMPO Air Quality Monitor..."

# Check if backend is already running
if pgrep -f "flask-backend.py" > /dev/null; then
    echo "⚠️  Backend is already running"
else
    echo "🔄 Starting Flask backend..."
    python flask-backend.py &
    sleep 3
fi

# Check if backend is responding
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ Backend is running on http://localhost:5000"
else
    echo "❌ Backend failed to start"
    exit 1
fi

echo "🌐 Starting frontend server..."
echo "📱 Open your browser and go to: http://localhost:8000"
echo "🔗 Or simply open index.html in your browser"

# Start frontend server
python -m http.server 8000
"""
    
    try:
        with open('start_app.sh', 'w') as f:
            f.write(startup_script)
        
        # Make it executable
        os.chmod('start_app.sh', 0o755)
        print("✅ Startup script created: start_app.sh")
        return True
    except Exception as e:
        print(f"❌ Failed to create startup script: {e}")
        return False


def main():
    """Main setup function"""
    print("🌟 NASA TEMPO Air Quality Monitor - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Install system dependencies
    print("\n📦 Installing system dependencies...")
    install_system_dependencies()
    
    # Install Python dependencies
    print("\n🐍 Installing Python dependencies...")
    install_python_dependencies()
    
    # Test imports
    print("\n🧪 Testing package imports...")
    if not test_imports():
        print("\n⚠️  Some packages failed to import, but continuing...")
    
    # Test Flask app
    print("\n🚀 Testing Flask application...")
    test_flask_app()
    
    # Create startup script
    print("\n📝 Creating startup script...")
    create_startup_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run: ./start_app.sh")
    print("2. Or manually:")
    print("   - Start backend: python flask-backend.py")
    print("   - Open index.html in your browser")
    print("   - Or serve frontend: python -m http.server 8000")
    print("\n🌐 Access the application:")
    print("   - Frontend: http://localhost:8000")
    print("   - Backend API: http://localhost:5000")
    print("\n📚 For detailed setup instructions, see: SETUP_GUIDE.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)