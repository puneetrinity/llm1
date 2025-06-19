#!/bin/bash
# scripts/setup.sh - Enhanced setup script for development and containers

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo -e "${BLUE}🚀 Setting up Enhanced LLM Proxy development environment${NC}"
echo "============================================================="

# Detect environment
ENVIRONMENT="local"
if [ -n "$RUNPOD_POD_ID" ]; then
    ENVIRONMENT="runpod"
    print_info "Detected RunPod environment (Pod: $RUNPOD_POD_ID)"
elif [ -f /.dockerenv ]; then
    ENVIRONMENT="docker"
    print_info "Detected Docker container environment"
elif [ -n "$CONTAINER" ]; then
    ENVIRONMENT="container"
    print_info "Detected container environment"
else
    print_info "Detected local development environment"
fi

# Create necessary directories
print_info "Creating project directories..."
mkdir -p data/{cache,logs,models}
mkdir -p tests
mkdir -p frontend/{src,public,build}
mkdir -p static/dashboard

# Create placeholder files for data directories
touch data/cache/.gitkeep
touch data/logs/.gitkeep
touch data/models/.gitkeep

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: docker-compose up --build"
echo "3. Test: curl http://localhost:8000/health"

if [ "$ENVIRONMENT" = "runpod" ] && [ -n "$RUNPOD_POD_ID" ]; then
    echo ""
    echo -e "${BLUE}RunPod Access:${NC}"
    echo "   • Public URL: https://${RUNPOD_POD_ID}-8001.proxy.runpod.net"
    echo "   • Dashboard: https://${RUNPOD_POD_ID}-8001.proxy.runpod.net/app"
fi

echo ""
print_success "Your LLM Proxy is ready to go! 🎉"