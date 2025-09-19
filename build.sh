#!/bin/bash

# Build Script for Precision Agriculture Platform
# This script builds and prepares the entire Docker containerized platform

set -e  # Exit on any error

echo "🌱 Building Precision Agriculture Platform..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker Compose is installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/{hyperspectral,multispectral,sensor_data,ground_truth,uploads}
    mkdir -p models
    mkdir -p logs
    mkdir -p uploads
    mkdir -p dashboard/templates
    mkdir -p dashboard/static
    
    print_success "Directories created"
}

# Set proper permissions
set_permissions() {
    print_status "Setting proper permissions..."
    
    chmod +x build.sh
    chmod +x deploy.sh
    chmod +x scripts/*.sh 2>/dev/null || true
    
    # Set permissions for data directories
    chmod -R 755 data/ 2>/dev/null || true
    chmod -R 755 models/ 2>/dev/null || true
    chmod -R 755 logs/ 2>/dev/null || true
    chmod -R 755 uploads/ 2>/dev/null || true
    
    print_success "Permissions set"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    # Build the main application image
    docker build -t precision-agriculture-platform:latest .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Pull required images
pull_images() {
    print_status "Pulling required Docker images..."
    
    docker-compose pull
    
    if [ $? -eq 0 ]; then
        print_success "Required images pulled successfully"
    else
        print_warning "Some images may not have been pulled, but continuing..."
    fi
}

# Validate configuration files
validate_configs() {
    print_status "Validating configuration files..."
    
    # Check if required config files exist
    if [ ! -f "configs/model_config.yaml" ]; then
        print_error "Model configuration file not found"
        exit 1
    fi
    
    if [ ! -f "configs/deployment_config.yaml" ]; then
        print_error "Deployment configuration file not found"
        exit 1
    fi
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "Docker Compose file not found"
        exit 1
    fi
    
    print_success "Configuration files validated"
}

# Initialize sample data
initialize_data() {
    print_status "Initializing sample data..."
    
    # Create sample data files
    echo "Sample hyperspectral data placeholder" > data/hyperspectral/sample.txt
    echo "Sample multispectral data placeholder" > data/multispectral/sample.txt
    echo "Sample sensor data placeholder" > data/sensor_data/sample.txt
    echo "Sample ground truth data placeholder" > data/ground_truth/sample.txt
    
    print_success "Sample data initialized"
}

# Test Docker Compose configuration
test_compose() {
    print_status "Testing Docker Compose configuration..."
    
    docker-compose config > /dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Docker Compose configuration is valid"
    else
        print_error "Docker Compose configuration has errors"
        exit 1
    fi
}

# Main build process
main() {
    echo "Starting build process..."
    echo "Time: $(date)"
    echo ""
    
    check_docker
    check_docker_compose
    validate_configs
    create_directories
    set_permissions
    initialize_data
    test_compose
    pull_images
    build_image
    
    echo ""
    echo "=================================================="
    print_success "Build completed successfully! 🎉"
    echo ""
    echo "Next steps:"
    echo "1. Run './deploy.sh' to start the platform"
    echo "2. Access the dashboard at http://localhost:80"
    echo "3. API will be available at http://localhost:80/api/"
    echo ""
    echo "For more information, check the README.md file"
    echo "=================================================="
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-pull)
            SKIP_PULL=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-pull    Skip pulling Docker images"
            echo "  --dev        Build in development mode"
            echo "  -h, --help   Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main