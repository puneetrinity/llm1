#!/bin/bash
# build_and_run.sh - Build and run the Docker container

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo -e "${BLUE}"
echo "🐳 LLM Proxy - Docker Build & Run"
echo "=================================="
echo -e "${NC}"

# Configuration
IMAGE_NAME="llm-proxy"
CONTAINER_NAME="llm-proxy-container"
PORT="8001"

# Parse command line arguments
BUILD_ONLY=false
RUN_ONLY=false
PRODUCTION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        --production)
            PRODUCTION=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build-only    Build image only, don't run"
            echo "  --run-only      Run existing image only"
            echo "  --production    Build for production (no debug)"
            echo "  --port PORT     Specify port (default: 8001)"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

print_success "Docker is available"

# Build the image (unless --run-only)
if [ "$RUN_ONLY" = false ]; then
    print_info "Building Docker image..."
    
    # Build arguments
    BUILD_ARGS=""
    if [ "$PRODUCTION" = true ]; then
        BUILD_ARGS="--build-arg VITE_DEBUG=false --build-arg VITE_AUTO_AUTHENTICATE=false"
        print_info "Building for production"
    else
        BUILD_ARGS="--build-arg VITE_DEBUG=true --build-arg VITE_AUTO_AUTHENTICATE=true"
        print_info "Building for development"
    fi
    
    # Build command
    docker build \
        $BUILD_ARGS \
        --build-arg VITE_BACKEND_URL=http://localhost:$PORT \
        --build-arg VITE_API_KEY=sk-default \
        -t $IMAGE_NAME:latest \
        -f Dockerfile \
        .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Docker build failed"
        exit 1
    fi
fi

# Exit if build-only
if [ "$BUILD_ONLY" = true ]; then
    print_info "Build complete. Use --run-only to run the container."
    exit 0
fi

# Stop and remove existing container
if docker ps -a | grep -q $CONTAINER_NAME; then
    print_info "Stopping existing container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

# Run the container
print_info "Starting Docker container..."

docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8001 \
    -e HOST=0.0.0.0 \
    -e PORT=8001 \
    -e DEBUG=$([ "$PRODUCTION" = true ] && echo "false" || echo "true") \
    -e LOG_LEVEL=INFO \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    $IMAGE_NAME:latest

if [ $? -eq 0 ]; then
    print_success "Container started successfully"
    
    # Wait a moment for startup
    print_info "Waiting for application to start..."
    sleep 5
    
    # Check if container is running
    if docker ps | grep -q $CONTAINER_NAME; then
        print_success "Container is running"
        
        echo ""
        echo "🎯 Access Points:"
        echo "  • Main Application: http://localhost:$PORT"
        echo "  • Health Check: http://localhost:$PORT/health"
        echo "  • API Documentation: http://localhost:$PORT/docs"
        echo ""
        echo "📋 Container Management:"
        echo "  • View logs: docker logs -f $CONTAINER_NAME"
        echo "  • Stop container: docker stop $CONTAINER_NAME"
        echo "  • Remove container: docker rm $CONTAINER_NAME"
        echo ""
        
        # Test health endpoint
        print_info "Testing health endpoint..."
        sleep 2
        if curl -s -f "http://localhost:$PORT/health" >/dev/null; then
            print_success "Health check passed"
        else
            print_warning "Health check failed - container may still be starting"
        fi
        
    else
        print_error "Container failed to start"
        print_info "Checking logs..."
        docker logs $CONTAINER_NAME
        exit 1
    fi
else
    print_error "Failed to start container"
    exit 1
fi
