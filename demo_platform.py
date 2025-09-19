#!/usr/bin/env python3
"""
Precision Agriculture Platform - Main Demo Application
Comprehensive AI-powered platform for crop health monitoring
"""

import os
import sys
import logging
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='dashboard/templates', static_folder='dashboard/static')
CORS(app)

# Configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'precision-agriculture-key'),
    MODEL_PATH=os.environ.get('MODEL_PATH', '/app/models'),
    DATA_PATH=os.environ.get('DATA_PATH', '/app/data'),
    UPLOAD_FOLDER='/app/uploads'
)

# Global variables for platform status
platform_status = {
    'status': 'initializing',
    'services': {},
    'last_update': datetime.now().isoformat()
}

def initialize_platform():
    """Initialize the precision agriculture platform"""
    global platform_status
    
    logger.info("Initializing Precision Agriculture Platform...")
    
    try:
        # Create necessary directories
        os.makedirs('/app/logs', exist_ok=True)
        os.makedirs('/app/data', exist_ok=True)
        os.makedirs('/app/models', exist_ok=True)
        os.makedirs('/app/uploads', exist_ok=True)
        
        # Initialize services
        services_status = {}
        
        # Data Acquisition Service
        try:
            from data_acquisition import download_datasets, preprocess_images
            services_status['data_acquisition'] = 'active'
            logger.info("Data acquisition service initialized")
        except ImportError as e:
            services_status['data_acquisition'] = 'error'
            logger.warning(f"Data acquisition service failed: {e}")
        
        # Model Service
        try:
            from models import cnn_models, training
            services_status['models'] = 'active'
            logger.info("Model service initialized")
        except ImportError as e:
            services_status['models'] = 'error'
            logger.warning(f"Model service failed: {e}")
        
        # Explainability Service
        try:
            from explainability import grad_cam, interpretability
            services_status['explainability'] = 'active'
            logger.info("Explainability service initialized")
        except ImportError as e:
            services_status['explainability'] = 'error'
            logger.warning(f"Explainability service failed: {e}")
        
        # Deployment Service
        try:
            from deployment import api, model_server
            services_status['deployment'] = 'active'
            logger.info("Deployment service initialized")
        except ImportError as e:
            services_status['deployment'] = 'error'
            logger.warning(f"Deployment service failed: {e}")
        
        # Continuous Learning Service
        try:
            from continuous_learning import active_learning, drift_detection
            services_status['continuous_learning'] = 'active'
            logger.info("Continuous learning service initialized")
        except ImportError as e:
            services_status['continuous_learning'] = 'error'
            logger.warning(f"Continuous learning service failed: {e}")
        
        platform_status.update({
            'status': 'running',
            'services': services_status,
            'last_update': datetime.now().isoformat()
        })
        
        logger.info("Platform initialization completed successfully")
        
    except Exception as e:
        platform_status.update({
            'status': 'error',
            'error': str(e),
            'last_update': datetime.now().isoformat()
        })
        logger.error(f"Platform initialization failed: {e}")

@app.route('/')
def index():
    """Main dashboard page"""
    return jsonify({
        'message': 'Precision Agriculture Platform',
        'status': platform_status['status'],
        'services': platform_status['services'],
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'platform_status': platform_status['status'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status')
def get_status():
    """Get platform status"""
    return jsonify(platform_status)

@app.route('/api/models')
def list_models():
    """List available models"""
    models_dir = app.config['MODEL_PATH']
    models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.pth', '.pkl', '.joblib', '.h5')):
                models.append({
                    'name': file,
                    'path': os.path.join(models_dir, file),
                    'size': os.path.getsize(os.path.join(models_dir, file)),
                    'modified': datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(models_dir, file))
                    ).isoformat()
                })
    
    return jsonify({'models': models})

@app.route('/api/data')
def list_datasets():
    """List available datasets"""
    data_dir = app.config['DATA_PATH']
    datasets = []
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                datasets.append({
                    'name': item,
                    'path': item_path,
                    'files': len(os.listdir(item_path)) if os.path.isdir(item_path) else 0,
                    'modified': datetime.fromtimestamp(
                        os.path.getmtime(item_path)
                    ).isoformat()
                })
    
    return jsonify({'datasets': datasets})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # This is a placeholder for the actual prediction logic
        data = request.get_json()
        
        # Simulate prediction
        prediction = {
            'crop_health': 'healthy',
            'confidence': 0.95,
            'recommendations': [
                'Continue current irrigation schedule',
                'Monitor for pest activity in sector 3'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """File upload endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'path': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_background_tasks():
    """Run background monitoring tasks"""
    while True:
        try:
            # Update platform status
            platform_status['last_update'] = datetime.now().isoformat()
            
            # Here you would add actual monitoring logic
            logger.info("Background monitoring task executed")
            
            time.sleep(60)  # Run every minute
            
        except Exception as e:
            logger.error(f"Background task error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    # Initialize platform
    initialize_platform()
    
    # Start background tasks
    background_thread = threading.Thread(target=run_background_tasks, daemon=True)
    background_thread.start()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)