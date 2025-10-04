#!/usr/bin/env python3
"""
Flask Backend for NASA TEMPO Air Quality Data Integration
Processes NetCDF files from AWS S3 and provides REST API
"""

import os
import sys
import json
import logging
import boto3
import numpy as np
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("Warning: xarray not available. Using mock data for demonstration.")
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time
from io import BytesIO

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask_backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global data cache
data_cache = {
    'air_quality_data': {},
    'last_update': None,
    'locations': [],
    'cache_lock': threading.Lock()
}

class NASATempoProcessor:
    """Processes NASA TEMPO data from S3"""
    
    def __init__(self):
        """Initialize the processor with AWS S3 client"""
        self.bucket_name = os.getenv('S3_BUCKET', 'nasa-tempo-air-quality')
        self.pollutants = ['NO2', 'O3', 'HCHO', 'SO2', 'CO']
        self.s3_client = self._setup_s3()
        
    def _setup_s3(self):
        """Setup AWS S3 client"""
        try:
            aws_config = {
                'region_name': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            }
            
            # Use explicit credentials if provided
            if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
                aws_config.update({
                    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
                })
                logger.info("Using explicit AWS credentials")
            else:
                logger.info("Using default AWS credential chain")
            
            s3_client = boto3.client('s3', **aws_config)
            
            # Test connection
            s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 connection successful to bucket: {self.bucket_name}")
            return s3_client
            
        except Exception as e:
            logger.error(f"S3 setup failed: {e}")
            return None
    
    def fetch_tempo_data(self) -> Dict[str, Any]:
        """Fetch NASA TEMPO data from S3"""
        if not self.s3_client:
            logger.warning("S3 client not available - using mock data for demonstration")
            return self._generate_mock_data("mock_tempo_data.nc")
        
        try:
            # List objects in S3 bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='tempo-data/',
                MaxKeys=100
            )
            
            if 'Contents' not in response:
                logger.warning("No data found in S3 bucket")
                return {}
            
            # Process the most recent files
            latest_files = sorted(
                response['Contents'], 
                key=lambda x: x['LastModified'], 
                reverse=True
            )[:10]  # Get latest 10 files
            
            processed_data = {}
            
            for file_obj in latest_files:
                file_key = file_obj['Key']
                logger.info(f"Processing file: {file_key}")
                
                try:
                    # Download file from S3
                    file_response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=file_key
                    )
                    
                    # Process NetCDF data
                    file_data = file_response['Body'].read()
                    processed_file_data = self._process_netcdf_data(file_data, file_key)
                    
                    if processed_file_data:
                        processed_data.update(processed_file_data)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_key}: {e}")
                    continue
            
            logger.info(f"Processed {len(processed_data)} data points")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching TEMPO data: {e}")
            return {}
    
    def _process_netcdf_data(self, file_data: bytes, file_key: str) -> Dict[str, Any]:
        """Process NetCDF data and convert to GeoJSON format"""
        try:
            if not HAS_XARRAY:
                # Return mock data for demonstration
                return self._generate_mock_data(file_key)
            
            # Create a BytesIO object for xarray
            file_buffer = BytesIO(file_data)
            
            # Open NetCDF file with xarray
            with xr.open_dataset(file_buffer) as dataset:
                # Extract relevant variables
                processed_data = {}
                
                # Get coordinates
                if 'lat' in dataset.variables and 'lon' in dataset.variables:
                    lats = dataset['lat'].values
                    lons = dataset['lon'].values
                else:
                    logger.warning(f"No lat/lon coordinates found in {file_key}")
                    return {}
                
                # Process each pollutant
                for pollutant in self.pollutants:
                    if pollutant in dataset.variables:
                        values = dataset[pollutant].values
                        
                        # Handle multi-dimensional data
                        if values.ndim > 2:
                            values = values[0]  # Take first time step
                        
                        # Create GeoJSON features
                        features = []
                        for i in range(min(len(lats), values.shape[0])):
                            for j in range(min(len(lons), values.shape[1])):
                                value = float(values[i, j])
                                
                                # Skip invalid values
                                if np.isnan(value) or value < 0:
                                    continue
                                
                                feature = {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Point",
                                        "coordinates": [float(lons[j]), float(lats[i])]
                                    },
                                    "properties": {
                                        "pollutant": pollutant,
                                        "value": value,
                                        "timestamp": datetime.now().isoformat(),
                                        "file": file_key
                                    }
                                }
                                features.append(feature)
                        
                        if features:
                            processed_data[pollutant] = {
                                "type": "FeatureCollection",
                                "features": features
                            }
                
                return processed_data
                
        except Exception as e:
            logger.error(f"Error processing NetCDF data from {file_key}: {e}")
            return {}
    
    def _generate_mock_data(self, file_key: str) -> Dict[str, Any]:
        """Generate mock air quality data for demonstration"""
        logger.info(f"Generating mock data for {file_key}")
        
        # Generate mock data points across North America
        mock_data = {}
        
        for pollutant in self.pollutants:
            features = []
            
            # Generate random data points
            for i in range(50):  # 50 data points per pollutant
                # Random coordinates within North America
                lat = np.random.uniform(25, 50)  # 25°N to 50°N
                lon = np.random.uniform(-125, -65)  # 125°W to 65°W
                
                # Generate realistic air quality values
                if pollutant == 'NO2':
                    value = np.random.uniform(10, 100)  # ppb
                elif pollutant == 'O3':
                    value = np.random.uniform(20, 120)  # ppb
                elif pollutant == 'HCHO':
                    value = np.random.uniform(5, 50)   # ppb
                elif pollutant == 'SO2':
                    value = np.random.uniform(2, 30)   # ppb
                elif pollutant == 'CO':
                    value = np.random.uniform(100, 1000)  # ppb
                else:
                    value = np.random.uniform(10, 100)
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "pollutant": pollutant,
                        "value": value,
                        "timestamp": datetime.now().isoformat(),
                        "file": file_key
                    }
                }
                features.append(feature)
            
            if features:
                mock_data[pollutant] = {
                    "type": "FeatureCollection",
                    "features": features
                }
        
        return mock_data
    
    def get_air_quality_bounds(self, lat_min: float, lat_max: float, 
                              lon_min: float, lon_max: float) -> Dict[str, Any]:
        """Get air quality data within geographical bounds"""
        with data_cache['cache_lock']:
            all_data = data_cache['air_quality_data']
        
        filtered_data = {}
        
        for pollutant, geojson_data in all_data.items():
            if 'features' not in geojson_data:
                continue
                
            filtered_features = []
            for feature in geojson_data['features']:
                coords = feature['geometry']['coordinates']
                lon, lat = coords[0], coords[1]
                
                if (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    filtered_features.append(feature)
            
            if filtered_features:
                filtered_data[pollutant] = {
                    "type": "FeatureCollection",
                    "features": filtered_features
                }
        
        return filtered_data

# Initialize processor
tempo_processor = NASATempoProcessor()

def update_data_cache():
    """Background task to update data cache"""
    global data_cache
    
    while True:
        try:
            logger.info("Updating air quality data cache...")
            
            # Fetch new data
            new_data = tempo_processor.fetch_tempo_data()
            
            # Update cache
            with data_cache['cache_lock']:
                data_cache['air_quality_data'] = new_data
                data_cache['last_update'] = datetime.now()
            
            logger.info("Data cache updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating data cache: {e}")
        
        # Wait for next update (1 hour)
        time.sleep(3600)

# Start background update thread
update_thread = threading.Thread(target=update_data_cache, daemon=True)
update_thread.start()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'last_data_update': data_cache['last_update'].isoformat() if data_cache['last_update'] else None
    })

@app.route('/api/air-quality', methods=['GET'])
def get_air_quality():
    """Get all air quality data"""
    with data_cache['cache_lock']:
        return jsonify(data_cache['air_quality_data'])

@app.route('/api/air-quality/<pollutant>', methods=['GET'])
def get_pollutant_data(pollutant):
    """Get data for specific pollutant"""
    pollutant = pollutant.upper()
    
    with data_cache['cache_lock']:
        if pollutant in data_cache['air_quality_data']:
            return jsonify(data_cache['air_quality_data'][pollutant])
        else:
            return jsonify({'error': f'No data found for pollutant: {pollutant}'}), 404

@app.route('/api/air-quality/bounds', methods=['GET'])
def get_air_quality_bounds():
    """Get air quality data within geographical bounds"""
    try:
        lat_min = float(request.args.get('lat_min', -90))
        lat_max = float(request.args.get('lat_max', 90))
        lon_min = float(request.args.get('lon_min', -180))
        lon_max = float(request.args.get('lon_max', 180))
        
        filtered_data = tempo_processor.get_air_quality_bounds(
            lat_min, lat_max, lon_min, lon_max
        )
        
        return jsonify(filtered_data)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid bounds parameters: {e}'}), 400

@app.route('/api/locations', methods=['POST'])
def save_location():
    """Save user location"""
    try:
        data = request.get_json()
        
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Missing required fields: latitude, longitude'}), 400
        
        location = {
            'id': len(data_cache['locations']) + 1,
            'name': data.get('name', 'Unnamed Location'),
            'latitude': float(data['latitude']),
            'longitude': float(data['longitude']),
            'timestamp': datetime.now().isoformat()
        }
        
        with data_cache['cache_lock']:
            data_cache['locations'].append(location)
        
        return jsonify(location), 201
        
    except Exception as e:
        return jsonify({'error': f'Error saving location: {e}'}), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get saved user locations"""
    with data_cache['cache_lock']:
        return jsonify(data_cache['locations'])

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get application status"""
    with data_cache['cache_lock']:
        return jsonify({
            'status': 'running',
            'data_points': sum(len(data.get('features', [])) for data in data_cache['air_quality_data'].values()),
            'pollutants': list(data_cache['air_quality_data'].keys()),
            'last_update': data_cache['last_update'].isoformat() if data_cache['last_update'] else None,
            'locations_count': len(data_cache['locations'])
        })

# Serve static files
@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    # Initial data fetch
    logger.info("Starting NASA TEMPO Flask backend...")
    logger.info("Fetching initial data...")
    
    initial_data = tempo_processor.fetch_tempo_data()
    with data_cache['cache_lock']:
        data_cache['air_quality_data'] = initial_data
        data_cache['last_update'] = datetime.now()
    
    logger.info("Initial data fetch completed")
    
    # Start Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)