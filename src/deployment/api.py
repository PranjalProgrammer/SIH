"""
REST API for Precision Agriculture Platform
Flask-based API for model inference and data management
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import os
from datetime import datetime
import logging

# Import platform modules
import sys
sys.path.append('/app/src')

logger = logging.getLogger(__name__)

class AgriculturalAPI:
    """Main API class for agricultural platform"""
    
    def __init__(self, model_path='/app/models', data_path='/app/data'):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.model_path = model_path
        self.data_path = data_path
        self.models = {}
        self.class_names = ['Healthy', 'Disease', 'Pest', 'Nutrient Deficiency', 'Water Stress']
        
        self._setup_routes()
        self._load_models()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'models_loaded': len(self.models)
            })
        
        @self.app.route('/api/predict/image', methods=['POST'])
        def predict_image():
            """Predict crop health from image"""
            try:
                # Get image from request
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                image = Image.open(image_file.stream).convert('RGB')
                
                # Preprocess image
                image_tensor = self._preprocess_image(image)
                
                # Get model
                model_name = request.form.get('model', 'default')
                model = self.models.get(model_name)
                
                if model is None:
                    return jsonify({'error': f'Model {model_name} not found'}), 404
                
                # Predict
                prediction = self._predict_image(model, image_tensor)
                
                # Add metadata
                prediction['timestamp'] = datetime.now().isoformat()
                prediction['model_used'] = model_name
                
                return jsonify(prediction)
                
            except Exception as e:
                logger.error(f"Image prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/predict/sensor', methods=['POST'])
        def predict_sensor():
            """Predict from sensor data"""
            try:
                data = request.get_json()
                
                if 'sensor_data' not in data:
                    return jsonify({'error': 'No sensor data provided'}), 400
                
                sensor_data = np.array(data['sensor_data'])
                model_name = data.get('model', 'sensor_default')
                
                model = self.models.get(model_name)
                if model is None:
                    return jsonify({'error': f'Model {model_name} not found'}), 404
                
                prediction = self._predict_sensor(model, sensor_data)
                prediction['timestamp'] = datetime.now().isoformat()
                prediction['model_used'] = model_name
                
                return jsonify(prediction)
                
            except Exception as e:
                logger.error(f"Sensor prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/predict/multimodal', methods=['POST'])
        def predict_multimodal():
            """Predict using multiple data modalities"""
            try:
                # Handle multipart form data
                image_file = request.files.get('image')
                sensor_data = request.form.get('sensor_data')
                weather_data = request.form.get('weather_data')
                
                if not image_file:
                    return jsonify({'error': 'Image required for multimodal prediction'}), 400
                
                # Process image
                image = Image.open(image_file.stream).convert('RGB')
                image_tensor = self._preprocess_image(image)
                
                # Process sensor data
                sensor_tensor = None
                if sensor_data:
                    sensor_array = json.loads(sensor_data)
                    sensor_tensor = torch.FloatTensor(sensor_array).unsqueeze(0)
                
                # Process weather data
                weather_tensor = None
                if weather_data:
                    weather_array = json.loads(weather_data)
                    weather_tensor = torch.FloatTensor(weather_array).unsqueeze(0)
                
                model_name = request.form.get('model', 'multimodal_default')
                model = self.models.get(model_name)
                
                if model is None:
                    return jsonify({'error': f'Model {model_name} not found'}), 404
                
                prediction = self._predict_multimodal(model, image_tensor, sensor_tensor, weather_tensor)
                prediction['timestamp'] = datetime.now().isoformat()
                prediction['model_used'] = model_name
                
                return jsonify(prediction)
                
            except Exception as e:
                logger.error(f"Multimodal prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/explain', methods=['POST'])
        def explain_prediction():
            """Get model explanation for prediction"""
            try:
                # Similar to predict but with explanations
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                image = Image.open(image_file.stream).convert('RGB')
                image_tensor = self._preprocess_image(image)
                
                model_name = request.form.get('model', 'default')
                model = self.models.get(model_name)
                
                if model is None:
                    return jsonify({'error': f'Model {model_name} not found'}), 404
                
                # Get prediction and explanation
                prediction = self._predict_image(model, image_tensor)
                explanation = self._explain_prediction(model, image_tensor, prediction)
                
                result = {
                    'prediction': prediction,
                    'explanation': explanation,
                    'timestamp': datetime.now().isoformat(),
                    'model_used': model_name
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Explanation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/models', methods=['GET'])
        def list_models():
            """List available models"""
            model_info = []
            for name, model in self.models.items():
                info = {
                    'name': name,
                    'type': type(model).__name__,
                    'parameters': sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0
                }
                model_info.append(info)
            
            return jsonify({'models': model_info})
        
        @self.app.route('/api/datasets', methods=['GET'])
        def list_datasets():
            """List available datasets"""
            datasets = []
            if os.path.exists(self.data_path):
                for item in os.listdir(self.data_path):
                    item_path = os.path.join(self.data_path, item)
                    if os.path.isdir(item_path):
                        datasets.append({
                            'name': item,
                            'path': item_path,
                            'files': len(os.listdir(item_path)),
                            'modified': datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                        })
            
            return jsonify({'datasets': datasets})
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_data():
            """Upload new data"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                data_type = request.form.get('type', 'image')
                
                # Create upload directory
                upload_dir = os.path.join(self.data_path, 'uploads', data_type)
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save file
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                
                return jsonify({
                    'message': 'File uploaded successfully',
                    'filename': filename,
                    'path': file_path,
                    'type': data_type
                })
                
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _load_models(self):
        """Load available models"""
        try:
            # This is a placeholder - in practice you'd load actual trained models
            logger.info("Loading models...")
            
            # For demo purposes, we'll create dummy model entries
            self.models = {
                'default': None,  # Placeholder for default model
                'cnn_model': None,  # Placeholder for CNN model
                'hybrid_model': None,  # Placeholder for hybrid model
                'sensor_default': None,  # Placeholder for sensor model
                'multimodal_default': None  # Placeholder for multimodal model
            }
            
            logger.info(f"Loaded {len(self.models)} model placeholders")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to standard size
        image = image.resize((224, 224))
        
        # Convert to tensor
        image_array = np.array(image) / 255.0
        image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def _predict_image(self, model, image_tensor):
        """Make prediction from image"""
        # Placeholder prediction logic
        # In practice, this would use the actual model
        
        prediction = {
            'predicted_class': np.random.randint(0, 5),
            'confidence': np.random.uniform(0.7, 0.95),
            'probabilities': np.random.dirichlet(np.ones(5)).tolist(),
            'class_names': self.class_names
        }
        
        prediction['predicted_label'] = self.class_names[prediction['predicted_class']]
        
        return prediction
    
    def _predict_sensor(self, model, sensor_data):
        """Make prediction from sensor data"""
        # Placeholder sensor prediction
        prediction = {
            'risk_scores': {
                'drought': np.random.uniform(0, 1),
                'flood': np.random.uniform(0, 1),
                'pest': np.random.uniform(0, 1),
                'disease': np.random.uniform(0, 1)
            },
            'overall_health': np.random.uniform(0.6, 0.9),
            'recommendations': [
                "Monitor soil moisture levels",
                "Check for pest activity",
                "Maintain current irrigation schedule"
            ]
        }
        
        return prediction
    
    def _predict_multimodal(self, model, image_tensor, sensor_tensor, weather_tensor):
        """Make multimodal prediction"""
        # Placeholder multimodal prediction
        prediction = {
            'predicted_class': np.random.randint(0, 5),
            'confidence': np.random.uniform(0.8, 0.95),
            'modality_weights': {
                'image': 0.6,
                'sensor': 0.3,
                'weather': 0.1
            },
            'risk_assessment': {
                'immediate': np.random.uniform(0, 0.3),
                'short_term': np.random.uniform(0, 0.5),
                'long_term': np.random.uniform(0, 0.7)
            }
        }
        
        prediction['predicted_label'] = self.class_names[prediction['predicted_class']]
        
        return prediction
    
    def _explain_prediction(self, model, image_tensor, prediction):
        """Generate explanation for prediction"""
        # Placeholder explanation
        explanation = {
            'important_regions': 'Upper left quadrant shows disease symptoms',
            'key_features': ['leaf discoloration', 'texture changes', 'spot patterns'],
            'confidence_factors': {
                'visual_symptoms': 0.8,
                'pattern_recognition': 0.7,
                'historical_data': 0.6
            },
            'alternative_diagnoses': [
                {'class': 'Nutrient Deficiency', 'probability': 0.15},
                {'class': 'Water Stress', 'probability': 0.10}
            ]
        }
        
        return explanation
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def create_api():
    """Create and return API instance"""
    return AgriculturalAPI()

def main():
    """Run API server"""
    api = create_api()
    api.run()

if __name__ == "__main__":
    main()