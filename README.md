# Precision Agriculture Platform

A comprehensive AI-powered platform for accurate, timely monitoring of crop health, soil condition, and pest risks using multispectral/hyperspectral imaging and environmental sensor data.

## 🌟 Features

- **Multi-modal Data Processing**: Hyperspectral imagery + IoT sensor data integration
- **Advanced AI Models**: CNN-LSTM hybrid architecture with attention mechanisms
- **Explainable AI**: Grad-CAM and SHAP for model interpretability
- **Real-time Monitoring**: Continuous health monitoring and alert systems
- **Scalable Deployment**: Docker containerization for edge and cloud deployment
- **Continuous Learning**: Active learning and model adaptation pipeline
- **Web Dashboard**: Interactive web interface for monitoring and control

## 🚀 Quick Start with Docker

### Prerequisites

- Docker (20.10+)
- Docker Compose (1.29+)
- At least 4GB RAM
- 10GB free disk space

### 1. Clone and Build

```bash
git clone <repository-url>
cd precision-agriculture-platform

# Build the platform
chmod +x build.sh deploy.sh
./build.sh
```

### 2. Deploy

```bash
# Deploy in production mode
./deploy.sh

# Or deploy in development mode (with logs)
./deploy.sh --dev

# Deploy with custom scaling
./deploy.sh --scale 3
```

### 3. Access the Platform

- **Dashboard**: http://localhost:80
- **API**: http://localhost:80/api/
- **Health Check**: http://localhost:80/health

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── data_acquisition/       # Data collection and preprocessing
│   ├── models/                 # AI model architectures
│   ├── explainability/         # Model interpretability
│   ├── deployment/             # Deployment infrastructure
│   └── continuous_learning/    # Feedback and adaptation
├── dashboard/                  # Web dashboard
├── configs/                    # Configuration files
├── docker/                     # Docker configurations
├── nginx/                      # Nginx configuration
├── mqtt/                       # MQTT broker configuration
├── database/                   # Database initialization
├── data/                       # Data storage
├── models/                     # Trained models
├── logs/                       # Application logs
├── Dockerfile                  # Main Docker image
├── docker-compose.yml          # Service orchestration
├── build.sh                    # Build script
├── deploy.sh                   # Deployment script
└── test_core_functionality.py  # Test suite
```

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

Configure AI models, training parameters, and evaluation metrics:

```yaml
models:
  cnn_crop_health:
    type: "CNN"
    architecture: "CropHealthCNN"
    input_channels: 3
    num_classes: 5
```

### Deployment Configuration (`configs/deployment_config.yaml`)

Configure deployment settings, databases, and monitoring:

```yaml
api:
  host: "0.0.0.0"
  port: 5000
  workers: 4

database:
  postgres:
    host: "postgres"
    port: 5432
```

## 🎯 Usage Examples

### 1. Image Prediction via API

```bash
curl -X POST \
  http://localhost:80/api/predict/image \
  -F "image=@crop_image.jpg" \
  -F "model=cnn_model"
```

### 2. Sensor Data Prediction

```bash
curl -X POST \
  http://localhost:80/api/predict/sensor \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [25.5, 60.2, 7.1, 45.0, 12.3, 89.1, 0.5],
    "model": "sensor_default"
  }'
```

### 3. Multimodal Prediction

```bash
curl -X POST \
  http://localhost:80/api/predict/multimodal \
  -F "image=@crop_image.jpg" \
  -F "sensor_data=[25.5, 60.2, 7.1, 45.0, 12.3, 89.1, 0.5]" \
  -F "weather_data=[22.1, 65.0, 1013.2, 5.2, 0.0]"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_core_functionality.py
```

## 📊 Monitoring

### View Service Status

```bash
docker-compose ps
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f precision-agriculture-app
```

### Monitor Resources

```bash
./monitor.sh  # Created after deployment
```

## 🔄 Management Commands

### Stop Services

```bash
docker-compose down
```

### Restart Services

```bash
docker-compose restart
```

### Scale Services

```bash
docker-compose up -d --scale precision-agriculture-app=3
```

### Update and Rebuild

```bash
./build.sh --no-pull
./deploy.sh --build
```

## 🏗️ Development

### Development Mode

```bash
# Deploy in development mode with live logs
./deploy.sh --dev
```

### Add New Models

1. Implement model in `src/models/`
2. Update `configs/model_config.yaml`
3. Register in model server
4. Rebuild and deploy

### Custom Configuration

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'
services:
  precision-agriculture-app:
    environment:
      - DEBUG=true
    volumes:
      - ./custom_data:/app/custom_data
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in `docker-compose.yml`
2. **Memory issues**: Increase Docker memory limit
3. **Permission errors**: Check file permissions with `chmod -R 755`

### Debug Mode

```bash
# Enable debug logging
export DEBUG=1
./deploy.sh --dev
```

### Check Service Health

```bash
curl http://localhost:80/health
curl http://localhost:80/api/status
```

## 📈 Performance Optimization

### For Production

1. **Enable caching**: Redis is included by default
2. **Scale services**: Use `--scale` parameter
3. **Monitor resources**: Use built-in monitoring
4. **Optimize models**: Use model quantization

### Hardware Recommendations

- **Minimum**: 4 CPU cores, 8GB RAM, 20GB storage
- **Recommended**: 8 CPU cores, 16GB RAM, 50GB storage, GPU support
- **Production**: 16+ CPU cores, 32GB+ RAM, SSD storage, multiple GPUs

## 🔒 Security

### Production Deployment

1. **Change default passwords** in `configs/deployment_config.yaml`
2. **Enable authentication** in API configuration
3. **Use HTTPS** with proper SSL certificates
4. **Configure firewall** rules
5. **Regular security updates**

### Environment Variables

Set sensitive data via environment variables:

```bash
export POSTGRES_PASSWORD=your_secure_password
export REDIS_PASSWORD=your_redis_password
./deploy.sh
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test: `python test_core_functionality.py`
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check inline code documentation
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Email**: [Your contact email]

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Flask team for the web framework
- Docker team for containerization technology
- Open source community for various libraries used

---

**Made with ❤️ for sustainable agriculture**