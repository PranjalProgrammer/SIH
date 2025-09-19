"""
Dashboard Application for Precision Agriculture Platform
Web-based interface for monitoring and controlling the agricultural AI system
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)

class AgriculturalDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.app.config.update(
            SECRET_KEY='precision-agriculture-dashboard',
            TEMPLATES_AUTO_RELOAD=True
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup dashboard routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('index.html')
        
        @self.app.route('/dashboard')
        def dashboard():
            """Main dashboard view"""
            # Get system status
            system_status = self._get_system_status()
            
            # Get recent predictions
            recent_predictions = self._get_recent_predictions()
            
            # Get alerts
            alerts = self._get_alerts()
            
            return render_template('dashboard.html',
                                 system_status=system_status,
                                 recent_predictions=recent_predictions,
                                 alerts=alerts)
        
        @self.app.route('/models')
        def models():
            """Model management page"""
            model_info = self._get_model_info()
            return render_template('models.html', models=model_info)
        
        @self.app.route('/data')
        def data():
            """Data management page"""
            datasets = self._get_datasets()
            return render_template('data.html', datasets=datasets)
        
        @self.app.route('/monitoring')
        def monitoring():
            """System monitoring page"""
            metrics = self._get_monitoring_metrics()
            return render_template('monitoring.html', metrics=metrics)
        
        @self.app.route('/predictions')
        def predictions():
            """Predictions history page"""
            predictions = self._get_prediction_history()
            return render_template('predictions.html', predictions=predictions)
        
        @self.app.route('/api/predict', methods=['POST'])
        def api_predict():
            """API endpoint for predictions"""
            try:
                # Handle file upload
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                
                # For demo purposes, return mock prediction
                prediction = {
                    'predicted_class': 'Healthy',
                    'confidence': 0.92,
                    'probabilities': {
                        'Healthy': 0.92,
                        'Disease': 0.03,
                        'Pest': 0.02,
                        'Nutrient Deficiency': 0.02,
                        'Water Stress': 0.01
                    },
                    'recommendations': [
                        'Crop appears healthy',
                        'Continue current care routine',
                        'Monitor for any changes'
                    ],
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(prediction)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system_status')
        def api_system_status():
            """API endpoint for system status"""
            return jsonify(self._get_system_status())
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """API endpoint for alerts"""
            return jsonify(self._get_alerts())
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics"""
            return jsonify(self._get_monitoring_metrics())
    
    def _get_system_status(self):
        """Get current system status"""
        return {
            'status': 'running',
            'uptime': '2 days, 14 hours',
            'cpu_usage': 45.2,
            'memory_usage': 68.7,
            'disk_usage': 23.4,
            'gpu_usage': 78.9,
            'models_loaded': 5,
            'active_connections': 12,
            'last_update': datetime.now().isoformat()
        }
    
    def _get_recent_predictions(self):
        """Get recent predictions"""
        return [
            {
                'id': 1,
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                'type': 'Image Analysis',
                'result': 'Healthy',
                'confidence': 0.94
            },
            {
                'id': 2,
                'timestamp': (datetime.now() - timedelta(minutes=12)).isoformat(),
                'type': 'Sensor Data',
                'result': 'Water Stress',
                'confidence': 0.87
            },
            {
                'id': 3,
                'timestamp': (datetime.now() - timedelta(minutes=18)).isoformat(),
                'type': 'Multimodal',
                'result': 'Nutrient Deficiency',
                'confidence': 0.91
            }
        ]
    
    def _get_alerts(self):
        """Get current alerts"""
        return [
            {
                'id': 1,
                'type': 'warning',
                'message': 'Model accuracy dropped below 90% in sector 3',
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'severity': 'medium'
            },
            {
                'id': 2,
                'type': 'info',
                'message': 'New training data available for processing',
                'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(),
                'severity': 'low'
            }
        ]
    
    def _get_model_info(self):
        """Get model information"""
        return [
            {
                'name': 'CNN Crop Health Model',
                'type': 'CNN',
                'status': 'active',
                'accuracy': 0.94,
                'last_trained': (datetime.now() - timedelta(days=3)).isoformat(),
                'predictions_today': 1247
            },
            {
                'name': 'LSTM Sensor Model',
                'type': 'LSTM',
                'status': 'active',
                'accuracy': 0.89,
                'last_trained': (datetime.now() - timedelta(days=1)).isoformat(),
                'predictions_today': 892
            },
            {
                'name': 'Hybrid Multimodal Model',
                'type': 'Hybrid',
                'status': 'active',
                'accuracy': 0.96,
                'last_trained': (datetime.now() - timedelta(days=2)).isoformat(),
                'predictions_today': 634
            }
        ]
    
    def _get_datasets(self):
        """Get dataset information"""
        return [
            {
                'name': 'Hyperspectral Images',
                'type': 'Image',
                'size': '15.2 GB',
                'samples': 45000,
                'last_updated': (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                'name': 'IoT Sensor Data',
                'type': 'Time Series',
                'size': '8.7 GB',
                'samples': 120000,
                'last_updated': (datetime.now() - timedelta(hours=6)).isoformat()
            },
            {
                'name': 'Ground Truth Labels',
                'type': 'Labels',
                'size': '124 MB',
                'samples': 45000,
                'last_updated': (datetime.now() - timedelta(days=2)).isoformat()
            }
        ]
    
    def _get_monitoring_metrics(self):
        """Get monitoring metrics"""
        return {
            'performance': {
                'requests_per_minute': 45,
                'average_response_time': 0.23,
                'error_rate': 0.02,
                'cache_hit_rate': 0.87
            },
            'model_metrics': {
                'accuracy': 0.94,
                'precision': 0.93,
                'recall': 0.95,
                'f1_score': 0.94
            },
            'system_health': {
                'cpu_usage': 45.2,
                'memory_usage': 68.7,
                'disk_usage': 23.4,
                'network_io': 12.3
            }
        }
    
    def _get_prediction_history(self):
        """Get prediction history"""
        history = []
        for i in range(20):
            history.append({
                'id': i + 1,
                'timestamp': (datetime.now() - timedelta(hours=i * 2)).isoformat(),
                'input_type': ['Image', 'Sensor', 'Multimodal'][i % 3],
                'prediction': ['Healthy', 'Disease', 'Pest', 'Nutrient Deficiency', 'Water Stress'][i % 5],
                'confidence': round(0.8 + (i % 20) * 0.01, 2),
                'processing_time': round(0.1 + (i % 10) * 0.02, 3)
            })
        return history
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Create templates directory and basic templates
def create_templates():
    """Create basic HTML templates"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Precision Agriculture Platform{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar {
            min-height: 100vh;
            background: #2c3e50;
        }
        .nav-link {
            color: #ecf0f1;
        }
        .nav-link:hover {
            color: #3498db;
        }
        .card-metric {
            border-left: 4px solid #3498db;
        }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <nav class="col-md-2 d-none d-md-block sidebar">
                <div class="position-sticky pt-3">
                    <h5 class="text-white mb-4">
                        <i class="fas fa-seedling me-2"></i>AgriAI
                    </h5>
                    <ul class="nav flex-column">
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('dashboard') }}">
                                <i class="fas fa-tachometer-alt me-2"></i>Dashboard
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('models') }}">
                                <i class="fas fa-brain me-2"></i>Models
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('data') }}">
                                <i class="fas fa-database me-2"></i>Data
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('predictions') }}">
                                <i class="fas fa-chart-line me-2"></i>Predictions
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('monitoring') }}">
                                <i class="fas fa-chart-bar me-2"></i>Monitoring
                            </a>
                        </li>
                    </ul>
                </div>
            </nav>
            
            <main class="col-md-10 ms-sm-auto px-md-4">
                {% block content %}{% endblock %}
            </main>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    with open(os.path.join(templates_dir, 'base.html'), 'w') as f:
        f.write(base_template)
    
    # Dashboard template
    dashboard_template = '''{% extends "base.html" %}

{% block title %}Dashboard - Precision Agriculture Platform{% endblock %}

{% block content %}
<div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
    <h1 class="h2">Dashboard</h1>
    <div class="btn-toolbar mb-2 mb-md-0">
        <div class="btn-group me-2">
            <button type="button" class="btn btn-sm btn-outline-secondary">
                <i class="fas fa-sync-alt me-1"></i>Refresh
            </button>
        </div>
    </div>
</div>

<!-- System Status Cards -->
<div class="row mb-4">
    <div class="col-md-3">
        <div class="card card-metric">
            <div class="card-body">
                <h5 class="card-title">System Status</h5>
                <h3 class="status-healthy">{{ system_status.status|title }}</h3>
                <small class="text-muted">Uptime: {{ system_status.uptime }}</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card card-metric">
            <div class="card-body">
                <h5 class="card-title">Models Loaded</h5>
                <h3>{{ system_status.models_loaded }}</h3>
                <small class="text-muted">Active models</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card card-metric">
            <div class="card-body">
                <h5 class="card-title">CPU Usage</h5>
                <h3>{{ "%.1f"|format(system_status.cpu_usage) }}%</h3>
                <small class="text-muted">Current load</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card card-metric">
            <div class="card-body">
                <h5 class="card-title">Memory Usage</h5>
                <h3>{{ "%.1f"|format(system_status.memory_usage) }}%</h3>
                <small class="text-muted">RAM utilization</small>
            </div>
        </div>
    </div>
</div>

<!-- Recent Predictions and Alerts -->
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Recent Predictions</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Type</th>
                                <th>Result</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for prediction in recent_predictions %}
                            <tr>
                                <td>{{ prediction.timestamp[:19]|replace('T', ' ') }}</td>
                                <td>{{ prediction.type }}</td>
                                <td>
                                    <span class="badge bg-{% if prediction.result == 'Healthy' %}success{% else %}warning{% endif %}">
                                        {{ prediction.result }}
                                    </span>
                                </td>
                                <td>{{ "%.1f"|format(prediction.confidence * 100) }}%</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Alerts</h5>
            </div>
            <div class="card-body">
                {% for alert in alerts %}
                <div class="alert alert-{% if alert.severity == 'high' %}danger{% elif alert.severity == 'medium' %}warning{% else %}info{% endif %} alert-dismissible fade show" role="alert">
                    <strong>{{ alert.type|title }}:</strong> {{ alert.message }}
                    <small class="d-block text-muted">{{ alert.timestamp[:19]|replace('T', ' ') }}</small>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
        f.write(dashboard_template)
    
    # Simple index template
    index_template = '''{% extends "base.html" %}

{% block content %}
<div class="container mt-5">
    <div class="row justify-content-center">
        <div class="col-md-8 text-center">
            <h1 class="display-4 mb-4">
                <i class="fas fa-seedling text-success me-3"></i>
                Precision Agriculture Platform
            </h1>
            <p class="lead mb-4">
                AI-powered platform for accurate, timely monitoring of crop health, 
                soil condition, and pest risks using multispectral imaging and sensor data.
            </p>
            <a href="{{ url_for('dashboard') }}" class="btn btn-primary btn-lg">
                <i class="fas fa-tachometer-alt me-2"></i>Open Dashboard
            </a>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_template)

def main():
    """Run the dashboard"""
    create_templates()
    
    dashboard = AgriculturalDashboard()
    dashboard.run(debug=True)

if __name__ == "__main__":
    main()