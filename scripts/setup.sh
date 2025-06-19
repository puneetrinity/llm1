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

print_success "Project directories created"

# Environment file setup
print_info "Setting up environment configuration..."
if [ ! -f .env ]; then
    if [ -f .env.template ]; then
        print_info "Creating .env from template..."
        cp .env.template .env
        
        # Update for specific environments
        if [ "$ENVIRONMENT" = "runpod" ]; then
            # RunPod-specific updates
            sed -i 's/HOST=127.0.0.1/HOST=0.0.0.0/' .env
            sed -i 's/ENVIRONMENT=development/ENVIRONMENT=production/' .env
            if [ -n "$RUNPOD_POD_ID" ]; then
                echo "PUBLIC_URL=https://${RUNPOD_POD_ID}-8001.proxy.runpod.net" >> .env
            fi
        fi
        
        print_success "Created .env file"
    else
        print_warning "No .env.template found, creating basic .env..."
        cat > .env << EOF
# Basic LLM Proxy Configuration
ENVIRONMENT=$ENVIRONMENT
HOST=0.0.0.0
PORT=8001
LOG_LEVEL=INFO
DEBUG=false

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=mistral:7b-instruct-q4_0
OLLAMA_TIMEOUT=300

# Features
ENABLE_AUTH=false
ENABLE_STREAMING=true
ENABLE_DASHBOARD=true
ENABLE_CIRCUIT_BREAKER=true

# Security
DEFAULT_API_KEY=sk-dev-key-$(date +%s)
CORS_ORIGINS=["*"]
EOF
        print_success "Created basic .env file"
    fi
else
    print_success ".env file already exists"
fi

# Python dependencies
print_info "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    print_success "Python dependencies installed"
else
    print_warning "No requirements.txt found, installing basic dependencies..."
    pip install fastapi uvicorn pydantic-settings aiohttp psutil requests
    print_success "Basic Python dependencies installed"
fi

# Ollama setup for containers
if [ "$ENVIRONMENT" != "local" ]; then
    print_info "Setting up Ollama for container environment..."
    
    # Check if Ollama is installed
    if ! command -v ollama &> /dev/null; then
        print_info "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        
        if command -v ollama &> /dev/null; then
            print_success "Ollama installed successfully"
        else
            print_error "Ollama installation failed"
        fi
    else
        print_success "Ollama already installed"
    fi
    
    # Start Ollama service
    print_info "Starting Ollama service..."
    
    # Kill any existing processes
    pkill -f ollama || true
    sleep 2
    
    # Set container environment
    export OLLAMA_HOST=0.0.0.0:11434
    export OLLAMA_ORIGINS="*"
    export OLLAMA_KEEP_ALIVE=5m
    
    # Start Ollama serve in background
    nohup ollama serve > data/logs/ollama.log 2>&1 &
    OLLAMA_PID=$!
    print_info "Started Ollama service (PID: $OLLAMA_PID)"
    
    # Wait for Ollama to be ready
    print_info "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Ollama failed to start within 60 seconds"
        else
            echo "  Attempt $i/30..."
            sleep 2
        fi
    done
    
    # Pull default model if Ollama is ready
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_info "Checking for default model..."
        if ! curl -s http://localhost:11434/api/tags | grep -q "mistral:7b-instruct-q4_0"; then
            print_info "Pulling default model (this may take several minutes)..."
            ollama pull mistral:7b-instruct-q4_0 &
            PULL_PID=$!
            print_info "Model pull started in background (PID: $PULL_PID)"
        else
            print_success "Default model already available"
        fi
    fi
fi

# Frontend setup (if Node.js is available)
if command -v npm &> /dev/null && [ -f frontend/package.json ]; then
    print_info "Setting up frontend..."
    cd frontend
    npm install
    
    if [ "$ENVIRONMENT" != "local" ]; then
        print_info "Building frontend for production..."
        npm run build
        if [ -d build ] && [ -f build/index.html ]; then
            print_success "Frontend built successfully"
        else
            print_warning "Frontend build may have issues"
        fi
    fi
    cd ..
elif [ -d frontend ] && [ ! -f frontend/package.json ]; then
    print_warning "Frontend directory exists but no package.json found"
fi

# Create startup scripts
print_info "Creating startup scripts..."

# Main startup script
cat > start.sh << 'EOF'
#!/bin/bash
# start.sh - Main application startup

echo "🚀 Starting LLM Proxy..."

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama service..."
    if command -v ollama &> /dev/null; then
        export OLLAMA_HOST=0.0.0.0:11434
        export OLLAMA_ORIGINS="*"
        nohup ollama serve > data/logs/ollama.log 2>&1 &
        
        # Wait for ready
        for i in {1..15}; do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo "✅ Ollama ready"
                break
            fi
            sleep 2
        done
    else
        echo "⚠️ Ollama not found - LLM features will be limited"
    fi
fi

# Start the main application
echo "Starting LLM Proxy application..."
if [ -f main_master.py ]; then
    python main_master.py
elif [ -f main.py ]; then
    python main.py
else
    echo "❌ No main application file found"
    exit 1
fi
EOF

chmod +x start.sh

# Development script
cat > dev.sh << 'EOF'
#!/bin/bash
# dev.sh - Development mode startup

echo "🧪 Starting LLM Proxy in development mode..."

# Set development environment
export DEBUG=true
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development

# Start with hot reload if uvicorn is available
if python -c "import uvicorn" 2>/dev/null; then
    echo "Starting with uvicorn hot reload..."
    if [ -f main.py ]; then
        uvicorn main:app --host 0.0.0.0 --port 8001 --reload
    else
        echo "❌ main.py not found"
        exit 1
    fi
else
    echo "Starting with Python (no hot reload)..."
    python main.py
fi
EOF

chmod +x dev.sh

print_success "Startup scripts created (start.sh, dev.sh)"

# Health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
# health_check.sh - Quick health check

echo "🏥 LLM Proxy Health Check"
echo "========================"

# Check main service
if curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo "✅ Main service: Running"
    curl -s http://localhost:8001/health | jq . 2>/dev/null || echo "Response received"
else
    echo "❌ Main service: Not responding"
fi

# Check Ollama
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama service: Running"
    model_count=$(curl -s http://localhost:11434/api/tags | jq '.models | length' 2>/dev/null || echo "unknown")
    echo "   Models available: $model_count"
else
    echo "❌ Ollama service: Not responding"
fi

# Check ports
if ss -tuln 2>/dev/null | grep -q ":8001 "; then
    echo "✅ Port 8001: Open"
else
    echo "❌ Port 8001: Closed"
fi

if ss -tuln 2>/dev/null | grep -q ":11434 "; then
    echo "✅ Port 11434: Open"
else
    echo "❌ Port 11434: Closed"
fi
EOF

chmod +x health_check.sh

print_success "Health check script created"

# Summary
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "==================="

print_info "Environment: $ENVIRONMENT"
print_info "Configuration: .env created"
print_info "Dependencies: Python packages installed"

if [ "$ENVIRONMENT" != "local" ]; then
    print_info "Ollama: Configured for container environment"
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review and edit .env file if needed"
echo "2. Start the application:"
echo "   • Production: ./start.sh"
echo "   • Development: ./dev.sh"
echo "3. Health check: ./health_check.sh"
echo "4. Test API: curl http://localhost:8001/health"

if [ "$ENVIRONMENT" = "runpod" ] && [ -n "$RUNPOD_POD_ID" ]; then
    echo ""
    echo -e "${BLUE}RunPod Access:${NC}"
    echo "   • Public URL: https://${RUNPOD_POD_ID}-8001.proxy.runpod.net"
    echo "   • Dashboard: https://${RUNPOD_POD_ID}-8001.proxy.runpod.net/app"
fi

echo ""
print_success "Your LLM Proxy is ready to go! 🎉"
