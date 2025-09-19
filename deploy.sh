#!/bin/bash

# Deployment Script for Precision Agriculture Platform
# This script deploys the containerized platform using Docker Compose

set -e  # Exit on any error

echo "🚀 Deploying Precision Agriculture Platform..."
echo "=============================================="

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

# Default values
ENVIRONMENT="production"
DETACHED=true
BUILD=false
SCALE_REPLICAS=1

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if build script has been run
    if ! docker images | grep -q "precision-agriculture-platform"; then
        print_warning "Docker image not found. Running build first..."
        ./build.sh
    fi
    
    # Check if Docker Compose file exists
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        exit 1
    fi
    
    print_success "Prerequisites checked"
}

# Stop existing containers
stop_existing() {
    print_status "Stopping existing containers..."
    
    docker-compose down --remove-orphans 2>/dev/null || true
    
    print_success "Existing containers stopped"
}

# Start services
start_services() {
    print_status "Starting services..."
    
    local compose_args=""
    
    if [ "$BUILD" = true ]; then
        compose_args="$compose_args --build"
    fi
    
    if [ "$DETACHED" = true ]; then
        compose_args="$compose_args -d"
    fi
    
    # Start the services
    docker-compose up $compose_args
    
    if [ $? -eq 0 ]; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Scale services if needed
scale_services() {
    if [ "$SCALE_REPLICAS" -gt 1 ]; then
        print_status "Scaling services to $SCALE_REPLICAS replicas..."
        
        docker-compose up -d --scale precision-agriculture-app=$SCALE_REPLICAS
        
        print_success "Services scaled successfully"
    fi
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        print_status "Health check attempt $attempt/$max_attempts"
        
        # Check main application health
        if curl -f http://localhost:5000/health &>/dev/null; then
            print_success "Main application is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Services failed to become healthy within timeout"
            print_status "Showing service logs..."
            docker-compose logs --tail=50
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Show deployment status
show_status() {
    print_status "Deployment status:"
    echo ""
    
    # Show running containers
    docker-compose ps
    echo ""
    
    # Show service URLs
    echo "🌐 Service URLs:"
    echo "   Dashboard: http://localhost:80"
    echo "   API:       http://localhost:80/api/"
    echo "   Health:    http://localhost:80/health"
    echo ""
    
    # Show logs command
    echo "📋 Useful commands:"
    echo "   View logs:    docker-compose logs -f"
    echo "   Stop:         docker-compose down"
    echo "   Restart:      docker-compose restart"
    echo "   Status:       docker-compose ps"
    echo ""
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Create monitoring scripts
    cat > monitor.sh << 'EOF'
#!/bin/bash
# Simple monitoring script

echo "=== Precision Agriculture Platform Status ==="
echo "Time: $(date)"
echo ""

echo "Container Status:"
docker-compose ps

echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo ""
echo "Health Checks:"
curl -s http://localhost:5000/health | python3 -m json.tool 2>/dev/null || echo "Health check failed"
EOF
    
    chmod +x monitor.sh
    
    print_success "Monitoring setup complete"
}

# Backup current state
backup_state() {
    print_status "Creating backup of current state..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup data directories
    cp -r data/ "$backup_dir/" 2>/dev/null || true
    cp -r models/ "$backup_dir/" 2>/dev/null || true
    cp -r logs/ "$backup_dir/" 2>/dev/null || true
    
    # Backup configurations
    cp -r configs/ "$backup_dir/" 2>/dev/null || true
    cp docker-compose.yml "$backup_dir/" 2>/dev/null || true
    
    print_success "Backup created at $backup_dir"
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    echo "Environment: $ENVIRONMENT"
    echo "Time: $(date)"
    echo ""
    
    check_prerequisites
    
    if [ "$ENVIRONMENT" = "production" ]; then
        backup_state
    fi
    
    stop_existing
    start_services
    scale_services
    
    if [ "$DETACHED" = true ]; then
        wait_for_services
        setup_monitoring
        show_status
        
        echo "=============================================="
        print_success "Deployment completed successfully! 🎉"
        echo ""
        print_status "Platform is running and ready to use"
        echo "=============================================="
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --no-detach)
            DETACHED=false
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --scale)
            SCALE_REPLICAS="$2"
            shift 2
            ;;
        --dev)
            ENVIRONMENT="development"
            DETACHED=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV        Set environment (development|staging|production)"
            echo "  --no-detach      Run in foreground (show logs)"
            echo "  --build          Rebuild images before starting"
            echo "  --scale N        Scale to N replicas"
            echo "  --dev            Development mode (equivalent to --env development --no-detach)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Deploy in production mode"
            echo "  $0 --dev             # Deploy in development mode"
            echo "  $0 --build           # Rebuild and deploy"
            echo "  $0 --scale 3         # Deploy with 3 replicas"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main