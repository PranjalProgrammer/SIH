#!/bin/bash

# NASA TEMPO Data Downloader Setup Script

echo "🚀 NASA TEMPO Data Downloader Setup"
echo "===================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install requirements
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p downloads logs

# Make scripts executable
chmod +x nasa_tempo_downloader.py
chmod +x nasa_tempo_advanced.py
chmod +x setup.py
chmod +x test_setup.py
chmod +x example_usage.py
chmod +x run_demo.py

echo "✅ Scripts made executable"

# Run setup test
echo "🧪 Running setup test..."
python3 test_setup.py

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Get NASA API key from: https://api.nasa.gov/"
echo "2. Update config.json with your credentials"
echo "3. Run: python3 nasa_tempo_downloader.py"
echo ""
echo "For help: python3 run_demo.py"