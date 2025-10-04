#!/usr/bin/env python3
"""
NASA TEMPO Air Quality Data Backend
Fetches and processes air quality data from NASA TEMPO AWS S3 bucket
Converts NetCDF files to GeoJSON and serves via REST API
"""

import os
import json
import logging
import boto3
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from typing import Dict, List, Optional, Tuple
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class TEMPODataProcessor:
    """Handles NASA TEMPO air quality data processing"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'nasa-tempo-air-quality'
        self.data_cache = {}
        self.last_update = None
        
    def fetch_latest_data(self) -> Dict:
        """Fetch the latest TEMPO data from S3 bucket"""
        try:
            # List objects in the bucket to find the latest file
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='tempo-air-quality/',
                MaxKeys=100
            )
            
            if 'Contents' not in response:
                logger.warning("No data files found in S3 bucket")
                return {}
            
            # Get the most recent file
            latest_file = max(response['Contents'], key=lambda x: x['LastModified'])
            file_key = latest_file['Key']
            
            logger.info(f"Fetching latest data from: {file_key}")
            
            # Download the NetCDF file
            temp_file = f"/tmp/tempo_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
            self.s3_client.download_file(self.bucket_name, file_key, temp_file)
            
            return self.process_netcdf_file(temp_file)
            
        except Exception as e:
            logger.error(f"Error fetching data from S3: {str(e)}")
            return {}
    
    def process_netcdf_file(self, file_path: str) -> Dict:
        """Process NetCDF file and convert to GeoJSON format"""
        try:
            # Open NetCDF file
            ds = xr.open_dataset(file_path)
            
            # Extract air quality parameters
            # TEMPO typically provides NO2, O3, HCHO, and other pollutants
            pollutants = {}
            
            # Process each pollutant variable
            for var_name in ds.data_vars:
                if var_name in ['NO2', 'O3', 'HCHO', 'SO2', 'CO']:
                    data = ds[var_name]
                    
                    # Convert to GeoJSON format
                    geojson_data = self.convert_to_geojson(data, var_name)
                    pollutants[var_name] = geojson_data
            
            # Clean up temporary file
            os.remove(file_path)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'pollutants': pollutants,
                'metadata': {
                    'source': 'NASA TEMPO',
                    'resolution': '0.1° x 0.1°',
                    'coverage': 'North America'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing NetCDF file: {str(e)}")
            return {}
    
    def convert_to_geojson(self, data_array: xr.DataArray, pollutant_name: str) -> Dict:
        """Convert xarray DataArray to GeoJSON format"""
        try:
            # Get coordinates
            lons = data_array.longitude.values
            lats = data_array.latitude.values
            
            # Get data values (handle NaN values)
            values = data_array.values
            values = np.where(np.isnan(values), None, values)
            
            # Create GeoJSON features
            features = []
            
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    value = values[i, j]
                    if value is not None:
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [float(lon), float(lat)]
                            },
                            "properties": {
                                "pollutant": pollutant_name,
                                "value": float(value),
                                "unit": self.get_pollutant_unit(pollutant_name),
                                "aqi_category": self.calculate_aqi_category(pollutant_name, value)
                            }
                        }
                        features.append(feature)
            
            return {
                "type": "FeatureCollection",
                "features": features,
                "properties": {
                    "pollutant": pollutant_name,
                    "total_points": len(features),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error converting to GeoJSON: {str(e)}")
            return {"type": "FeatureCollection", "features": []}
    
    def get_pollutant_unit(self, pollutant: str) -> str:
        """Get the unit for a specific pollutant"""
        units = {
            'NO2': 'ppb',
            'O3': 'ppb',
            'HCHO': 'ppb',
            'SO2': 'ppb',
            'CO': 'ppm'
        }
        return units.get(pollutant, 'unknown')
    
    def calculate_aqi_category(self, pollutant: str, value: float) -> str:
        """Calculate AQI category based on pollutant value"""
        # Simplified AQI calculation (would need actual EPA standards)
        if pollutant == 'NO2':
            if value < 53: return 'Good'
            elif value < 100: return 'Moderate'
            elif value < 360: return 'Unhealthy for Sensitive Groups'
            elif value < 649: return 'Unhealthy'
            else: return 'Very Unhealthy'
        elif pollutant == 'O3':
            if value < 54: return 'Good'
            elif value < 70: return 'Moderate'
            elif value < 85: return 'Unhealthy for Sensitive Groups'
            elif value < 105: return 'Unhealthy'
            else: return 'Very Unhealthy'
        else:
            return 'Unknown'

# Initialize data processor
data_processor = TEMPODataProcessor()

# Background task for data updates
def update_data_periodically():
    """Update data every hour in background"""
    while True:
        try:
            logger.info("Updating air quality data...")
            new_data = data_processor.fetch_latest_data()
            if new_data:
                data_processor.data_cache = new_data
                data_processor.last_update = datetime.now()
                logger.info("Data updated successfully")
            else:
                logger.warning("Failed to update data")
        except Exception as e:
            logger.error(f"Error in background update: {str(e)}")
        
        # Wait for 1 hour (3600 seconds)
        time.sleep(3600)

# Start background thread
update_thread = threading.Thread(target=update_data_periodically, daemon=True)
update_thread.start()

# REST API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'last_update': data_processor.last_update.isoformat() if data_processor.last_update else None
    })

@app.route('/api/air-quality', methods=['GET'])
def get_air_quality_data():
    """Get current air quality data"""
    try:
        # Check if we have cached data
        if not data_processor.data_cache:
            # Fetch fresh data
            data_processor.data_cache = data_processor.fetch_latest_data()
            data_processor.last_update = datetime.now()
        
        return jsonify(data_processor.data_cache)
        
    except Exception as e:
        logger.error(f"Error getting air quality data: {str(e)}")
        return jsonify({'error': 'Failed to fetch air quality data'}), 500

@app.route('/api/air-quality/<pollutant>', methods=['GET'])
def get_pollutant_data(pollutant: str):
    """Get data for a specific pollutant"""
    try:
        if not data_processor.data_cache:
            data_processor.data_cache = data_processor.fetch_latest_data()
            data_processor.last_update = datetime.now()
        
        pollutant_data = data_processor.data_cache.get('pollutants', {}).get(pollutant.upper())
        
        if pollutant_data:
            return jsonify(pollutant_data)
        else:
            return jsonify({'error': f'Pollutant {pollutant} not found'}), 404
            
    except Exception as e:
        logger.error(f"Error getting pollutant data: {str(e)}")
        return jsonify({'error': 'Failed to fetch pollutant data'}), 500

@app.route('/api/air-quality/bounds', methods=['GET'])
def get_data_bounds():
    """Get geographical bounds of available data"""
    try:
        lat_min = request.args.get('lat_min', type=float)
        lat_max = request.args.get('lat_max', type=float)
        lon_min = request.args.get('lon_min', type=float)
        lon_max = request.args.get('lon_max', type=float)
        
        if not all([lat_min, lat_max, lon_min, lon_max]):
            return jsonify({'error': 'All bounds parameters required'}), 400
        
        # Filter data by bounds
        filtered_data = {}
        
        for pollutant, geojson_data in data_processor.data_cache.get('pollutants', {}).items():
            filtered_features = []
            for feature in geojson_data.get('features', []):
                coords = feature['geometry']['coordinates']
                lon, lat = coords[0], coords[1]
                
                if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                    filtered_features.append(feature)
            
            filtered_data[pollutant] = {
                **geojson_data,
                'features': filtered_features
            }
        
        return jsonify({
            'bounds': {
                'lat_min': lat_min,
                'lat_max': lat_max,
                'lon_min': lon_min,
                'lon_max': lon_max
            },
            'pollutants': filtered_data
        })
        
    except Exception as e:
        logger.error(f"Error filtering data by bounds: {str(e)}")
        return jsonify({'error': 'Failed to filter data'}), 500

@app.route('/api/locations', methods=['POST'])
def save_user_location():
    """Save user's pinned location"""
    try:
        data = request.get_json()
        
        required_fields = ['latitude', 'longitude', 'name']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # In a real application, you'd save this to a database
        # For now, we'll just return success
        location_data = {
            'id': f"loc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'name': data['name'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'message': 'Location saved successfully',
            'location': location_data
        })
        
    except Exception as e:
        logger.error(f"Error saving location: {str(e)}")
        return jsonify({'error': 'Failed to save location'}), 500

@app.route('/api/locations', methods=['GET'])
def get_user_locations():
    """Get saved user locations"""
    # In a real application, you'd fetch from database
    # For demo purposes, return empty list
    return jsonify({'locations': []})

if __name__ == '__main__':
    # Initial data fetch
    logger.info("Starting NASA TEMPO Air Quality Backend...")
    logger.info("Fetching initial data...")
    data_processor.data_cache = data_processor.fetch_latest_data()
    data_processor.last_update = datetime.now()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)