# NASA TEMPO Air Quality Monitor - Complete Setup Guide

This guide will help you set up and run the NASA TEMPO Air Quality Monitor project step by step.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Modern web browser
- AWS Account (optional, for real data)

## Step 1: Install Dependencies

### Install Python packages:
```bash
pip install -r requirements.txt
```

If you encounter issues with NetCDF4 installation:

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libnetcdf-dev libhdf5-dev pkg-config
pip install -r requirements.txt
```

**On macOS:**
```bash
brew install netcdf hdf5 pkg-config
pip install -r requirements.txt
```

**On Windows:**
```bash
# Download pre-compiled wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# Or use conda:
conda install netcdf4
pip install -r requirements.txt
```

## Step 2: Configure Environment Variables (Optional)

For real NASA TEMPO data, set these environment variables:

```bash
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET="your-s3-bucket-name"
```

**Note:** If you don't set these, the application will use mock data for demonstration.

## Step 3: Test the Setup

Run the setup test to verify everything is working:

```bash
python test_setup.py
```

## Step 4: Start the Application

### Option A: Manual Start (Recommended for Development)

1. **Start the Flask backend:**
   ```bash
   python flask-backend.py
   ```
   The backend will start on http://localhost:5000

2. **Open the frontend:**
   - Simply open `index.html` in your web browser
   - Or serve it with a simple HTTP server:
     ```bash
     python -m http.server 8000
     ```
   - Then visit: http://localhost:8000

### Option B: Docker (Recommended for Production)

1. **Build and start with Docker Compose:**
   ```bash
   docker-compose up --build
   ```
   - Backend: http://localhost:5000
   - Frontend: http://localhost:80

## Step 5: Verify Everything Works

1. **Check backend health:**
   ```bash
   curl http://localhost:5000/health
   ```

2. **Check air quality data:**
   ```bash
   curl http://localhost:5000/api/air-quality
   ```

3. **Open the web interface:**
   - Visit http://localhost:8000 (or http://localhost:80 with Docker)
   - You should see an interactive map with air quality data

## Troubleshooting

### Common Issues and Solutions

#### 1. "ModuleNotFoundError: No module named 'netcdf4'"
**Solution:**
```bash
# Install system dependencies first
sudo apt-get install libnetcdf-dev libhdf5-dev pkg-config  # Ubuntu/Debian
# or
brew install netcdf hdf5 pkg-config  # macOS

# Then reinstall
pip install netCDF4==1.6.4
```

#### 2. "Port already in use"
**Solution:**
```bash
# Kill existing processes
pkill -f flask-backend.py
# or use a different port
PORT=5001 python flask-backend.py
```

#### 3. "AWS credentials not found"
**Solution:**
- This is normal! The app will use mock data
- To use real data, set AWS environment variables (see Step 2)

#### 4. "CORS errors in browser"
**Solution:**
- Make sure Flask-CORS is installed: `pip install Flask-CORS==4.0.0`
- Check that the backend is running on the correct port

#### 5. "Map not loading"
**Solution:**
- Check browser console for JavaScript errors
- Ensure you have internet connection (for Leaflet map tiles)
- Try refreshing the page

### Testing Individual Components

#### Test Backend Only:
```bash
python test_complete_functionality.py
```

#### Test Data Downloader:
```bash
python nasa_tempo_downloader.py --help
```

#### Test Advanced Downloader:
```bash
python nasa_tempo_advanced.py --help
```

## Project Structure

```
├── flask-backend.py          # Main Flask application
├── index.html               # Frontend web page
├── app-styles.css           # CSS styles
├── app-script.js            # JavaScript functionality
├── nasa_tempo_downloader.py # Standalone data downloader
├── nasa_tempo_advanced.py   # Advanced downloader
├── config.json              # Configuration file
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker orchestration
├── test_*.py               # Test scripts
└── README.md               # Documentation
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/air-quality` - All air quality data
- `GET /api/air-quality/{pollutant}` - Specific pollutant data
- `GET /api/air-quality/bounds` - Data within geographical bounds
- `POST /api/locations` - Save user location
- `GET /api/locations` - Get saved locations
- `GET /api/status` - Application status

## Features

- ✅ Real-time air quality data visualization
- ✅ Interactive Leaflet maps
- ✅ Multiple pollutant support (NO2, O3, HCHO, SO2, CO)
- ✅ Location saving functionality
- ✅ Responsive web interface
- ✅ RESTful API endpoints
- ✅ Docker deployment ready
- ✅ Mock data for demonstration

## Getting Help

If you're still having issues:

1. **Check the logs:**
   ```bash
   tail -f flask_backend.log
   ```

2. **Run diagnostics:**
   ```bash
   python test_complete_functionality.py
   ```

3. **Verify Python version:**
   ```bash
   python --version  # Should be 3.9+
   ```

4. **Check installed packages:**
   ```bash
   pip list | grep -E "(flask|boto3|xarray|netcdf4)"
   ```

## Next Steps

Once everything is running:

1. **Explore the web interface** - Click on the map to see air quality data
2. **Try different pollutants** - Use the dropdown to switch between NO2, O3, etc.
3. **Save locations** - Click on the map to save interesting locations
4. **Check the API** - Use curl or Postman to explore the REST API
5. **Deploy to production** - Use Docker for production deployment

---

**Need more help?** Check the troubleshooting section above or run the test scripts to diagnose issues.