#!/bin/bash

# NASA TEMPO Air Quality Project Deployment Script
# This script helps deploy the project to various cloud platforms

set -e

echo "🚀 NASA TEMPO Air Quality Project Deployment Script"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "📦 Deploying to Heroku..."
    
    if ! command_exists heroku; then
        echo "❌ Heroku CLI not found. Please install it first:"
        echo "   macOS: brew install heroku/brew/heroku"
        echo "   Ubuntu: curl https://cli-assets.heroku.com/install-ubuntu.sh | sh"
        exit 1
    fi
    
    # Login to Heroku
    heroku login
    
    # Create app
    read -p "Enter your Heroku app name (or press Enter for auto-generated): " app_name
    if [ -z "$app_name" ]; then
        heroku create nasa-tempo-air-quality-$(date +%s)
    else
        heroku create "$app_name"
    fi
    
    # Set environment variables
    echo "🔧 Setting up environment variables..."
    read -p "Enter your AWS Access Key ID: " aws_key
    read -p "Enter your AWS Secret Access Key: " aws_secret
    read -p "Enter your AWS Region (default: us-east-1): " aws_region
    
    aws_region=${aws_region:-us-east-1}
    
    heroku config:set AWS_ACCESS_KEY_ID="$aws_key"
    heroku config:set AWS_SECRET_ACCESS_KEY="$aws_secret"
    heroku config:set AWS_DEFAULT_REGION="$aws_region"
    heroku config:set FLASK_ENV=production
    
    # Create Procfile
    echo "web: gunicorn flask-backend:app" > Procfile
    
    # Deploy
    git add .
    git commit -m "Deploy NASA TEMPO Air Quality app to Heroku"
    git push heroku main
    
    echo "✅ Deployment successful!"
    echo "🌐 Your app is available at: $(heroku apps:info --json | jq -r '.app.web_url')"
}

# Function to deploy to Railway
deploy_railway() {
    echo "🚂 Deploying to Railway..."
    
    if ! command_exists railway; then
        echo "❌ Railway CLI not found. Please install it first:"
        echo "   npm install -g @railway/cli"
        exit 1
    fi
    
    # Login to Railway
    railway login
    
    # Initialize project
    railway init
    
    # Set environment variables
    echo "🔧 Setting up environment variables..."
    read -p "Enter your AWS Access Key ID: " aws_key
    read -p "Enter your AWS Secret Access Key: " aws_secret
    read -p "Enter your AWS Region (default: us-east-1): " aws_region
    
    aws_region=${aws_region:-us-east-1}
    
    railway variables set AWS_ACCESS_KEY_ID="$aws_key"
    railway variables set AWS_SECRET_ACCESS_KEY="$aws_secret"
    railway variables set AWS_DEFAULT_REGION="$aws_region"
    railway variables set FLASK_ENV=production
    
    # Deploy
    railway up
    
    echo "✅ Deployment successful!"
    echo "🌐 Your app is available at: $(railway domain)"
}

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    
    if ! command_exists docker; then
        echo "❌ Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f Dockerfile ]; then
        cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask-backend.py .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "flask-backend:app"]
EOF
    fi
    
    # Build and run
    echo "🔨 Building Docker image..."
    docker build -t nasa-tempo-air-quality .
    
    echo "🚀 Starting container..."
    docker run -d -p 5000:5000 \
        -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
        -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
        -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
        --name nasa-tempo-app \
        nasa-tempo-air-quality
    
    echo "✅ Docker deployment successful!"
    echo "🌐 Your app is available at: http://localhost:5000"
}

# Function to setup local development
setup_local() {
    echo "🛠️  Setting up local development environment..."
    
    # Check Python version
    if ! command_exists python3; then
        echo "❌ Python 3 not found. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Create virtual environment
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    
    # Set up environment variables
    echo "🔧 Setting up environment variables..."
    if [ ! -f .env ]; then
        cat > .env << EOF
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
EOF
        echo "📝 Created .env file. Please update it with your AWS credentials."
    fi
    
    echo "✅ Local setup complete!"
    echo "🚀 To start the backend: python flask-backend.py"
    echo "🌐 To start the frontend: python -m http.server 8000"
}

# Function to test the application
test_app() {
    echo "🧪 Testing the application..."
    
    # Start backend in background
    echo "🚀 Starting backend..."
    python flask-backend.py &
    BACKEND_PID=$!
    
    # Wait for backend to start
    sleep 5
    
    # Test health endpoint
    echo "🔍 Testing health endpoint..."
    if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "✅ Backend health check passed"
    else
        echo "❌ Backend health check failed"
        kill $BACKEND_PID
        exit 1
    fi
    
    # Test air quality endpoint
    echo "🔍 Testing air quality endpoint..."
    if curl -f http://localhost:5000/api/air-quality > /dev/null 2>&1; then
        echo "✅ Air quality endpoint working"
    else
        echo "⚠️  Air quality endpoint returned no data (this is normal if no AWS credentials are set)"
    fi
    
    # Clean up
    kill $BACKEND_PID
    
    echo "✅ Application tests completed!"
}

# Main menu
echo "Please select a deployment option:"
echo "1) Setup local development environment"
echo "2) Deploy to Heroku"
echo "3) Deploy to Railway"
echo "4) Deploy with Docker"
echo "5) Test application"
echo "6) Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        setup_local
        ;;
    2)
        deploy_heroku
        ;;
    3)
        deploy_railway
        ;;
    4)
        deploy_docker
        ;;
    5)
        test_app
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac