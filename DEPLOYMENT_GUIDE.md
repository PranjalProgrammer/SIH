# Precision Agriculture Platform - Deployment Guide

This guide provides comprehensive instructions for deploying the Precision Agriculture Platform as an independent Docker containerized solution.

## 🎯 Overview

The platform is designed to be completely self-contained and independent of your host machine. All components run in Docker containers with proper orchestration, networking, and data persistence.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB free space
- OS: Linux, macOS, or Windows with WSL2

**Recommended Requirements:**
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 50+ GB SSD
- GPU: NVIDIA GPU with CUDA support (optional)

### Software Requirements

1. **Docker Engine** (20.10+)
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Add user to docker group
   sudo usermod -aG docker $USER
   ```

2. **Docker Compose** (1.29+)
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Git** (for cloning)
   ```bash
   sudo apt-get install git  # Ubuntu/Debian
   brew install git          # macOS
   ```

## 🚀 Quick Deployment

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd precision-agriculture-platform
```

### Step 2: Build the Platform

```bash
# Make scripts executable
chmod +x build.sh deploy.sh

# Build the Docker images and setup
./build.sh
```

The build script will:
- ✅ Check Docker installation
- ✅ Validate configuration files
- ✅ Create necessary directories
- ✅ Build the main application Docker image
- ✅ Pull required service images (PostgreSQL, Redis, etc.)
- ✅ Initialize sample data

### Step 3: Deploy the Platform

```bash
# Deploy in production mode (detached)
./deploy.sh

# OR deploy in development mode (with live logs)
./deploy.sh --dev
```

### Step 4: Verify Deployment

```bash
# Check service status
docker-compose ps

# Check health
curl http://localhost:80/health

# View logs
docker-compose logs -f
```

## 🌐 Access Points

After successful deployment:

- **Web Dashboard**: http://localhost:80
- **REST API**: http://localhost:80/api/
- **Health Check**: http://localhost:80/health
- **API Documentation**: http://localhost:80/api/docs (if enabled)

## 🏗️ Architecture Overview

### Services Included

1. **Main Application** (`precision-agriculture-app`)
   - Flask API server
   - Web dashboard
   - AI model inference
   - Port: 5000 (internal), 80 (external via nginx)

2. **Database Services**
   - **PostgreSQL**: Metadata and results storage
   - **InfluxDB**: Time-series sensor data
   - **Redis**: Caching and session management

3. **Message Queue**
   - **MQTT Broker** (Mosquitto): IoT sensor communication

4. **Reverse Proxy**
   - **Nginx**: Load balancing and SSL termination

5. **Development Tools** (optional)
   - **Jupyter Lab**: For experimentation and analysis

### Network Architecture

```
Internet/Local Network
         ↓
    Nginx (Port 80/443)
         ↓
┌─────────────────────────┐
│  Main Application       │
│  - API Server (5000)    │
│  - Dashboard (8080)     │
│  - Model Server         │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Supporting Services    │
│  - PostgreSQL (5432)    │
│  - Redis (6379)         │
│  - InfluxDB (8086)      │
│  - MQTT (1883/9001)     │
└─────────────────────────┘
```

## 🔧 Configuration

### Environment Configuration

Create a `.env` file for environment-specific settings:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_password
POSTGRES_USER=agri_user
POSTGRES_DB=precision_agriculture

# Redis Configuration
REDIS_PASSWORD=your_redis_password

# InfluxDB Configuration
INFLUXDB_ADMIN_PASSWORD=admin_password
INFLUXDB_USER_PASSWORD=user_password

# Application Configuration
FLASK_ENV=production
DEBUG=false
SECRET_KEY=your_secret_key

# Model Configuration
MODEL_PATH=/app/models
DATA_PATH=/app/data
```

### Custom Docker Compose Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'

services:
  precision-agriculture-app:
    environment:
      - DEBUG=true
    volumes:
      - ./custom_models:/app/custom_models
    ports:
      - "5001:5000"  # Additional port mapping
  
  postgres:
    ports:
      - "5433:5432"  # Expose PostgreSQL externally
```

## 📊 Monitoring and Management

### Service Management

```bash
# View service status
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# Restart specific service
docker-compose restart precision-agriculture-app

# Scale services
docker-compose up -d --scale precision-agriculture-app=3

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ Data loss!)
docker-compose down -v
```

### Resource Monitoring

```bash
# View resource usage
docker stats

# View system resource usage
./monitor.sh  # Created after deployment
```

### Health Checks

```bash
# Application health
curl http://localhost:80/health

# Database connectivity
curl http://localhost:80/api/status

# Service-specific health checks
docker-compose exec postgres pg_isready
docker-compose exec redis redis-cli ping
```

## 🔒 Security Configuration

### Production Security Checklist

1. **Change Default Passwords**
   ```bash
   # Update .env file with secure passwords
   POSTGRES_PASSWORD=complex_secure_password_123!
   REDIS_PASSWORD=another_secure_password_456!
   ```

2. **Enable SSL/TLS**
   ```bash
   # Add SSL certificates to nginx/ssl/
   mkdir -p nginx/ssl
   cp your-cert.pem nginx/ssl/
   cp your-key.pem nginx/ssl/
   ```

3. **Configure Firewall**
   ```bash
   # Only expose necessary ports
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw deny 5432/tcp  # Block direct database access
   ```

4. **Enable Authentication**
   - Update `configs/deployment_config.yaml`
   - Set `authentication.enabled: true`
   - Configure JWT settings

### Network Security

```yaml
# In docker-compose.yml, create isolated networks
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access
```

## 🔄 Data Management

### Data Persistence

All data is persisted in Docker volumes:

- `postgres_data`: Database data
- `redis_data`: Cache data
- `influxdb_data`: Time-series data
- `./data`: Application data (mounted)
- `./models`: AI models (mounted)
- `./logs`: Application logs (mounted)

### Backup Strategy

```bash
# Automated backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup databases
docker-compose exec -T postgres pg_dump -U agri_user precision_agriculture > "$BACKUP_DIR/postgres.sql"
docker-compose exec -T redis redis-cli --rdb - > "$BACKUP_DIR/redis.rdb"

# Backup application data
cp -r data/ "$BACKUP_DIR/"
cp -r models/ "$BACKUP_DIR/"
cp -r configs/ "$BACKUP_DIR/"

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

### Data Recovery

```bash
# Restore from backup
BACKUP_DIR="backups/20231201_120000"

# Restore PostgreSQL
cat "$BACKUP_DIR/postgres.sql" | docker-compose exec -T postgres psql -U agri_user precision_agriculture

# Restore application data
cp -r "$BACKUP_DIR/data/" ./
cp -r "$BACKUP_DIR/models/" ./
```

## 🚀 Scaling and Performance

### Horizontal Scaling

```bash
# Scale main application
docker-compose up -d --scale precision-agriculture-app=5

# Use external load balancer for production
# Update nginx configuration for upstream servers
```

### Performance Optimization

1. **Enable Caching**
   - Redis is enabled by default
   - Configure TTL in `configs/deployment_config.yaml`

2. **Database Optimization**
   ```sql
   -- PostgreSQL tuning
   ALTER SYSTEM SET shared_buffers = '256MB';
   ALTER SYSTEM SET effective_cache_size = '1GB';
   ```

3. **Model Optimization**
   - Use model quantization
   - Enable batch processing
   - Configure model pools

### Resource Limits

```yaml
# In docker-compose.yml
services:
  precision-agriculture-app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :80
   
   # Change ports in docker-compose.yml
   ports:
     - "8080:80"  # Use port 8080 instead
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Memory
   
   # Linux: No limit by default
   ```

3. **Permission Errors**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   chmod -R 755 data/ models/ logs/
   ```

4. **Database Connection Issues**
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec postgres psql -U agri_user -d precision_agriculture
   ```

### Debug Mode

```bash
# Enable debug mode
export DEBUG=1
./deploy.sh --dev

# View detailed logs
docker-compose logs -f --tail=100
```

### Service Recovery

```bash
# Restart failed services
docker-compose restart

# Recreate containers
docker-compose up -d --force-recreate

# Complete reset (⚠️ Data loss!)
docker-compose down -v
docker system prune -a
./build.sh
./deploy.sh
```

## 📈 Performance Benchmarks

### Expected Performance

- **API Response Time**: < 200ms (95th percentile)
- **Model Inference**: < 100ms per image
- **Throughput**: 100+ requests/minute
- **Memory Usage**: 2-4GB per instance
- **CPU Usage**: 20-40% under normal load

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API endpoint
ab -n 1000 -c 10 http://localhost:80/health

# Test prediction endpoint
ab -n 100 -c 5 -p image.jpg -T multipart/form-data http://localhost:80/api/predict/image
```

## 🔄 Updates and Maintenance

### Regular Updates

```bash
# Update platform
git pull origin main
./build.sh
./deploy.sh --build

# Update only specific service
docker-compose pull postgres
docker-compose up -d postgres
```

### Maintenance Tasks

1. **Weekly**
   - Check logs for errors
   - Monitor resource usage
   - Backup data

2. **Monthly**
   - Update Docker images
   - Clean up old logs
   - Review security settings

3. **Quarterly**
   - Performance review
   - Security audit
   - Capacity planning

### Log Management

```bash
# Configure log rotation
echo '{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}' | sudo tee /etc/docker/daemon.json

sudo systemctl restart docker
```

## 📞 Support and Resources

### Getting Help

1. **Check Logs**: Always start with `docker-compose logs`
2. **Health Checks**: Use `/health` and `/api/status` endpoints
3. **Documentation**: Refer to inline code documentation
4. **Community**: GitHub issues and discussions

### Useful Commands Reference

```bash
# Quick status check
docker-compose ps && curl -s http://localhost:80/health

# View all logs
docker-compose logs --tail=50 -f

# Execute command in container
docker-compose exec precision-agriculture-app bash

# Database access
docker-compose exec postgres psql -U agri_user precision_agriculture

# Redis access
docker-compose exec redis redis-cli

# Clean restart
docker-compose down && docker-compose up -d
```

---

## 🎉 Conclusion

Your Precision Agriculture Platform is now deployed as a fully independent, containerized solution! The platform includes:

✅ **Complete AI Pipeline**: Data processing, model inference, and results
✅ **Web Dashboard**: User-friendly interface for monitoring
✅ **REST API**: Programmatic access for integrations
✅ **Database Stack**: PostgreSQL, InfluxDB, and Redis
✅ **Monitoring**: Health checks and logging
✅ **Scalability**: Easy horizontal scaling
✅ **Security**: Production-ready security features

The platform is designed to run independently on any Docker-capable machine without requiring any dependencies on your host system.

Happy farming! 🌱🚜