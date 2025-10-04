# Configuration Files for NASA TEMPO Air Quality Project

## Environment Variables

### Backend (.env file)
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
PORT=5000

# Data Configuration
TEMPO_S3_BUCKET=nasa-tempo-air-quality
UPDATE_INTERVAL=3600  # seconds (1 hour)
```

### Frontend (.env file)
```bash
# API Configuration
REACT_APP_API_URL=http://localhost:5000/api

# For production deployment
# REACT_APP_API_URL=https://your-backend-domain.com/api
```

## Docker Configuration

### Backend Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask-backend.py .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "flask-backend:app"]
```

### Frontend Dockerfile
```dockerfile
FROM nginx:alpine

COPY index.html /usr/share/nginx/html/
COPY app-styles.css /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose (docker-compose.yml)
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
    volumes:
      - ./flask-backend.py:/app/flask-backend.py

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

## Deployment Scripts

### Deploy to Heroku (deploy-heroku.sh)
```bash
#!/bin/bash

# Install Heroku CLI first: https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create Heroku app
heroku create nasa-tempo-air-quality

# Set environment variables
heroku config:set AWS_ACCESS_KEY_ID=your_aws_access_key
heroku config:set AWS_SECRET_ACCESS_KEY=your_aws_secret_key
heroku config:set AWS_DEFAULT_REGION=us-east-1
heroku config:set FLASK_ENV=production

# Deploy
git add .
git commit -m "Deploy NASA TEMPO Air Quality app"
git push heroku main

# Open the app
heroku open
```

### Deploy to Railway (deploy-railway.sh)
```bash
#!/bin/bash

# Install Railway CLI first: npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Set environment variables
railway variables set AWS_ACCESS_KEY_ID=your_aws_access_key
railway variables set AWS_SECRET_ACCESS_KEY=your_aws_secret_key
railway variables set AWS_DEFAULT_REGION=us-east-1

# Deploy
railway up

# Get deployment URL
railway domain
```

## Cron Job for Data Updates

### Local Cron Job (/etc/cron.d/tempo-update)
```bash
# Update NASA TEMPO data every hour
0 * * * * root curl -X POST http://localhost:5000/api/update-data
```

### Cloud Cron Job (using GitHub Actions)
```yaml
# .github/workflows/update-data.yml
name: Update Air Quality Data

on:
  schedule:
    - cron: '0 * * * *'  # Every hour
  workflow_dispatch:

jobs:
  update-data:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger data update
        run: |
          curl -X POST ${{ secrets.BACKEND_URL }}/api/update-data
```

## Monitoring Configuration

### Health Check Endpoint
The backend includes a health check endpoint at `/api/health` that returns:
- Application status
- Last data update timestamp
- System health metrics

### Logging Configuration
```python
# Add to flask-backend.py for production logging
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/tempo-app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('NASA TEMPO Air Quality app startup')
```

## Security Configuration

### CORS Configuration
```python
# In flask-backend.py
from flask_cors import CORS

CORS(app, origins=[
    "http://localhost:3000",
    "https://your-frontend-domain.com"
])
```

### Rate Limiting
```python
# Add to requirements.txt: Flask-Limiter==3.3.1
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Apply to endpoints
@app.route('/api/air-quality')
@limiter.limit("10 per minute")
def get_air_quality_data():
    # ... existing code
```

## Database Configuration (Optional)

### SQLite Database for User Locations
```python
# Add to requirements.txt: SQLAlchemy==2.0.19
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tempo_locations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class UserLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.String(50), nullable=True)  # For user authentication

# Initialize database
with app.app_context():
    db.create_all()
```