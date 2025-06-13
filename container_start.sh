#!/bin/bash
# container_start.sh - Container-Optimized Startup Script
# This script works in Docker, RunPod, or any fresh container environment

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

echo -e "${BLUE}"
echo "🐳 CONTAINER STARTUP - CONSOLIDATED LLM PROXY"
echo "============================================="
echo "Version: 3.0.0-container"
echo -e "${NC}"

# Detect container environment
print_info "Detecting environment..."
if [ -f /.dockerenv ]; then
    CONTAINER_TYPE="Docker"
    print_success "Running in Docker container"
elif [ -n "$RUNPOD_POD_ID" ]; then
    CONTAINER_TYPE="RunPod"
    print_success "Running in RunPod container (ID: $RUNPOD_POD_ID)"
elif [ -n "$JUPYTER_SERVER_ROOT" ]; then
    CONTAINER_TYPE="Jupyter"
    print_success "Running in Jupyter environment"
else
    CONTAINER_TYPE="Generic"
    print_info "Running in generic container/VM environment"
fi

# Set container-optimized environment variables
print_info "Setting container environment..."
export HOST=0.0.0.0
export PORT=${PORT:-8001}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export PYTHONUNBUFFERED=1

# RunPod-specific configuration
if [ "$CONTAINER_TYPE" = "RunPod" ]; then
    export PUBLIC_URL="https://${RUNPOD_POD_ID}-8001.proxy.runpod.net"
    print_success "RunPod public URL: $PUBLIC_URL"
fi

print_success "Environment configured for $CONTAINER_TYPE"

# Check system dependencies
print_info "Checking system dependencies..."

# Check Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python available: $PYTHON_VERSION"
else
    print_error "Python3 not found. Installing..."
    apt-get update >/dev/null 2>&1
    apt-get install -y python3 python3-pip >/dev/null 2>&1
    print_success "Python3 installed"
fi

# Check Node.js
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node --version)
    print_success "Node.js available: $NODE_VERSION"
else
    print_info "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - >/dev/null 2>&1
    apt-get install -y nodejs >/dev/null 2>&1
    print_success "Node.js installed"
fi

# Install Python dependencies
print_info "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt >/dev/null 2>&1
    print_success "Python dependencies installed"
else
    print_warning "No requirements.txt found, installing basic dependencies..."
    pip3 install fastapi uvicorn pydantic-settings python-multipart >/dev/null 2>&1
    print_success "Basic dependencies installed"
fi

# Setup environment file
print_info "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.template" ]; then
        cp .env.template .env
        print_success "Created .env from template"
    else
        cat > .env << 'EOF'
DEBUG=false
HOST=0.0.0.0
PORT=8001
LOG_LEVEL=INFO
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_AUTH=false
ENABLE_DASHBOARD=true
EOF
        print_success "Created basic .env file"
    fi
fi

# Add container-specific settings to .env
if [ "$CONTAINER_TYPE" = "RunPod" ] && [ -n "$PUBLIC_URL" ]; then
    echo "PUBLIC_URL=$PUBLIC_URL" >> .env
    echo "CONTAINER_TYPE=RunPod" >> .env
fi

# Build frontend if it exists
if [ -d "frontend" ]; then
    print_info "Building React frontend..."
    cd frontend
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        print_info "Installing Node.js dependencies..."
        npm install --silent >/dev/null 2>&1
        print_success "Node.js dependencies installed"
    fi
    
    # Build the app
    print_info "Building React app..."
    npm run build >/dev/null 2>&1
    
    if [ -d "build" ] && [ -f "build/index.html" ]; then
        print_success "React app built successfully"
        BUILD_SIZE=$(du -sh build 2>/dev/null | cut -f1 || echo "Unknown")
        print_info "Build size: $BUILD_SIZE"
    else
        print_error "React build failed"
        exit 1
    fi
    
    cd ..
else
    print_warning "No frontend directory found - dashboard will be disabled"
fi

# Check for main application file
print_info "Locating main application file..."
if [ -f "main_master.py" ]; then
    MAIN_FILE="main_master.py"
    print_success "Using main_master.py"
elif [ -f "main.py" ]; then
    MAIN_FILE="main.py"
    print_success "Using main.py"
else
    print_error "No main application file found"
    exit 1
fi

# Start Ollama if needed (container-specific)
if command -v ollama >/dev/null 2>&1; then
    print_info "Starting Ollama service..."
    ollama serve >/dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    print_success "Ollama service started (PID: $OLLAMA_PID)"
else
    print_warning "Ollama not found - LLM features will be limited"
fi

# Final setup
print_info "Final container setup..."

# Create logs directory
mkdir -p logs

# Set proper permissions
chmod +x *.sh 2>/dev/null || true

print_success "Container setup completed"

# Display startup information
echo ""
echo -e "${GREEN}🎉 READY TO START!${NC}"
echo "=================="
echo "🔧 Environment: $CONTAINER_TYPE"
echo "🐍 Python: $(python3 --version)"
echo "📦 Node.js: $(node --version 2>/dev/null || echo 'Not available')"
echo "📁 Working directory: $(pwd)"
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"

if [ -n "$PUBLIC_URL" ]; then
    echo "🌍 Public URL: $PUBLIC_URL"
fi

echo ""
echo "📍 Access Points (after startup):"
echo "  • Main API: http://localhost:$PORT"
echo "  • Health: http://localhost:$PORT/health"
echo "  • Dashboard: http://localhost:$PORT/app"
echo "  • API Docs: http://localhost:$PORT/docs"

if [ -n "$PUBLIC_URL" ]; then
    echo ""
    echo "🌐 Public Access (RunPod):"
    echo "  • Dashboard: $PUBLIC_URL/app"
    echo "  • API Docs: $PUBLIC_URL/docs"
fi

echo ""
echo "🚀 Starting the application..."
echo "Press Ctrl+C to stop"
echo ""

# Start the application
exec python3 "$MAIN_FILE"
