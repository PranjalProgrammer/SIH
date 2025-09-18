# Precision Agriculture Platform - Deployment Guide

## 🌾 Platform Overview

The Precision Agriculture Platform is a comprehensive AI-powered system for crop health monitoring, combining hyperspectral imagery analysis with IoT sensor data to provide accurate, timely insights for agricultural decision-making.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRECISION AGRICULTURE PLATFORM           │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Data Acquisition & Preprocessing                 │
│  ├── Hyperspectral image processing                        │
│  ├── IoT sensor data integration                           │
│  ├── Radiometric & atmospheric correction                  │
│  └── Feature extraction (NDVI, EVI, SAVI, etc.)          │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: AI Model Design & Training                       │
│  ├── CNN-LSTM hybrid architecture                          │
│  ├── Spatial-spectral feature extraction                   │
│  ├── Temporal attention mechanisms                         │
│  └── Advanced training pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Explainability & Novelty Detection              │
│  ├── Grad-CAM spatial attribution                          │
│  ├── SHAP spectral importance                              │
│  ├── Out-of-distribution detection                         │
│  └── Model interpretability suite                          │
├─────────────────────────────────────────────────────────────┤
│  Phase 4: Deployment & Dashboard                           │
│  ├── FastAPI REST API                                      │
│  ├── Interactive web dashboard                             │
│  ├── Docker containerization                               │
│  └── Kubernetes orchestration                              │
├─────────────────────────────────────────────────────────────┤
│  Phase 5: Continuous Learning & Feedback                   │
│  ├── Active learning pipeline                              │
│  ├── Drift detection & monitoring                          │
│  ├── Automated retraining                                  │
│  └── User feedback integration                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start Deployment

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- NVIDIA GPU (optional, for acceleration)
- 8GB+ RAM recommended

### 1. Environment Setup

```bash
# Clone the repository (or use existing workspace)
cd /workspace

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download and prepare sample data
python src/data_acquisition/download_datasets.py

# Run preprocessing pipeline
python src/data_acquisition/preprocess_images.py
python src/data_acquisition/sensor_processing.py
python src/data_acquisition/feature_extraction.py
```

### 3. Model Training (Optional)

```bash
# Train the CNN-LSTM model
python src/models/train.py --config configs/model_config.yaml

# Run hyperparameter optimization
python src/models/training.py --optimize
```

### 4. API Deployment

```bash
# Start the FastAPI server
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose -f docker/docker-compose.yml up -d
```

### 5. Dashboard Launch

```bash
# Install Streamlit dashboard dependencies
pip install streamlit plotly requests

# Launch dashboard
streamlit run dashboard/app.py --server.port 8501
```

## 🐳 Docker Deployment

### Single Container Deployment

```bash
# Build the image
docker build -f docker/Dockerfile -t precision-agriculture .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models precision-agriculture
```

### Full Stack Deployment

```bash
# Deploy complete stack with Docker Compose
cd docker
docker-compose up -d

# Services will be available at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:3000
# - Database: localhost:5432
# - Redis: localhost:6379
```

## ☸️ Kubernetes Deployment

### Basic Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n precision-agriculture

# Access services
kubectl port-forward svc/api-service 8000:8000
```

### Helm Chart Deployment

```bash
# Install using Helm
helm install precision-agriculture ./helm-chart/

# Upgrade deployment
helm upgrade precision-agriculture ./helm-chart/
```

## 🔧 Configuration

### API Configuration

Edit `configs/deployment_config.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

model_serving:
  model_path: "/workspace/models/best_model.pth"
  batch_size: 32
  
database:
  host: "localhost"
  port: 5432
  name: "precision_agriculture"
```

### Model Configuration

Edit `configs/model_config.yaml`:

```yaml
model:
  type: "crop_health"
  cnn:
    hidden_dim: 256
    dropout: 0.3
  lstm:
    hidden_dim: 128
    attention_type: "additive"

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
```

## 🌐 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Make Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": [[[...]]],
    "sensor_data": [[...]],
    "include_interpretation": true
  }'
```

### Get Model Information

```bash
curl http://localhost:8000/model/info
```

## 📊 Dashboard Features

Access the dashboard at `http://localhost:8501`

### Available Pages:

1. **Dashboard Overview**
   - Real-time field health maps
   - Key performance metrics
   - Environmental trends

2. **Real-time Monitoring**
   - Live sensor data streams
   - Immediate alert notifications
   - Current field conditions

3. **Crop Health Analysis**
   - NDVI trend analysis
   - Disease detection results
   - Stress pattern identification

4. **Prediction Tool**
   - Upload hyperspectral images
   - Input sensor data
   - Get AI predictions with explanations

5. **System Status**
   - API health monitoring
   - Performance metrics
   - Model information

## 🔍 Monitoring & Observability

### Performance Metrics

- **Prediction Latency**: Average response time
- **Throughput**: Requests per second
- **Accuracy**: Model performance metrics
- **Resource Usage**: CPU, memory, GPU utilization

### Logging

Logs are available in:
- Application logs: `/workspace/logs/`
- Container logs: `docker logs <container_name>`
- Kubernetes logs: `kubectl logs -f <pod_name>`

### Alerts

Configure alerts for:
- Model performance degradation
- System resource thresholds
- Data drift detection
- API error rates

## 🔄 Continuous Learning

### Active Learning Pipeline

```bash
# Run active learning selection
python src/continuous_learning/active_learning.py \
  --strategy combined \
  --n_samples 100
```

### Drift Detection

```bash
# Monitor for data drift
python src/continuous_learning/drift_detection.py \
  --method ks_test \
  --window_size 1000
```

### Automated Retraining

Configure automatic retraining triggers:
- Performance drop > 5%
- Data drift detected
- New labeled data threshold reached

## 📈 Scaling Considerations

### Horizontal Scaling

- **API**: Scale replicas based on load
- **Database**: Use read replicas for query scaling
- **Cache**: Implement Redis clustering
- **Storage**: Use distributed storage for large datasets

### Performance Optimization

- **Model Optimization**: Use TensorRT for GPU inference
- **Caching**: Implement multi-level caching
- **Load Balancing**: Use NGINX or cloud load balancers
- **Database**: Optimize queries and add indexes

## 🔐 Security

### Authentication & Authorization

- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- OAuth2 integration

### Data Security

- Encryption at rest and in transit
- Secure API endpoints
- Input validation and sanitization
- Regular security audits

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model file exists
   ls -la /workspace/models/
   
   # Verify model compatibility
   python -c "import torch; print(torch.load('models/best_model.pth').keys())"
   ```

2. **API Connection Issues**
   ```bash
   # Check API status
   curl http://localhost:8000/health
   
   # Check logs
   docker logs precision-agriculture-api
   ```

3. **Dashboard Not Loading**
   ```bash
   # Check Streamlit process
   ps aux | grep streamlit
   
   # Restart dashboard
   streamlit run dashboard/app.py --server.port 8501
   ```

4. **GPU Not Available**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check PyTorch GPU support
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Performance Issues

1. **Slow Predictions**
   - Check GPU utilization
   - Optimize batch size
   - Enable model quantization
   - Use TensorRT optimization

2. **High Memory Usage**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training
   - Optimize data loading

## 📚 Additional Resources

### Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Model Architecture](src/models/README.md) - Detailed model information
- [Data Processing](src/data_acquisition/README.md) - Data pipeline docs

### Support

- GitHub Issues: Report bugs and feature requests
- Documentation: Comprehensive guides and tutorials
- Community: Join discussions and share experiences

## 🎯 Production Checklist

Before deploying to production:

- [ ] Model performance validation
- [ ] Security audit completion
- [ ] Load testing performed
- [ ] Monitoring systems configured
- [ ] Backup and recovery tested
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Incident response plan ready

## 📝 License

MIT License - see LICENSE file for details.

---

**🌾 Ready for Agricultural Innovation!**

This platform empowers farmers, researchers, and agricultural organizations with cutting-edge AI technology for sustainable and productive farming practices.