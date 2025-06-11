#!/bin/bash
# setup.sh - FIXED for RunPod (No Loops, No Interactive Prompts)

set -e

echo "🚀 RunPod LLM Proxy Setup (Loop-Safe)"
echo "====================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# CRITICAL: Loop prevention
SETUP_MARKER="/tmp/llm_setup_complete"
SERVICES_MARKER="/tmp/llm_services_running"

# Check if setup already completed
if [ -f "$SETUP_MARKER" ]; then
    print_status "Setup already completed - starting services only"
    
    # Kill any existing processes
    pkill -f "ollama serve" 2>/dev/null || true
    pkill -f "python.*main" 2>/dev/null || true
    sleep 2
    
    # Start services
    export OLLAMA_HOST=0.0.0.0:11434
    export CUDA_VISIBLE_DEVICES=0
    
    print_info "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 8
    
    print_info "Starting Python service..."
    python3 main.py > /tmp/service.log 2>&1 &
    sleep 5
    
    # Test services
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_status "✅ Services running - NO MORE SETUP NEEDED"
        print_info "Health: http://localhost:8000/health"
        print_info "Docs: http://localhost:8000/docs"
    else
        print_warning "Services starting... check in a moment"
    fi
    
    echo "$(date): Services restarted" > "$SERVICES_MARKER"
    exit 0
fi

# First-time setup
print_info "Running first-time setup..."

# Update system
print_info "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y curl wget git python3 python3-pip build-essential jq dos2unix

# Install Ollama
if ! command -v ollama >/dev/null 2>&1; then
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    if command -v ollama >/dev/null 2>&1; then
        print_status "Ollama installed successfully"
    else
        print_error "Ollama installation failed"
        exit 1
    fi
else
    print_status "Ollama already installed"
fi

# Set environment variables
export OLLAMA_HOST=0.0.0.0:11434
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_MAX_LOADED_MODELS=2

# Start Ollama
print_info "Starting Ollama service..."
ollama serve > /tmp/ollama.log 2>&1 &
sleep 10

# Wait for Ollama to be ready
print_info "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_status "Ollama is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Ollama failed to start"
        print_info "Ollama logs:"
        tail -10 /tmp/ollama.log
        exit 1
    fi
    sleep 2
done

# Install Python dependencies
print_info "Installing Python dependencies..."
pip3 install --no-cache-dir --upgrade pip
pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    psutil==5.9.6

print_status "Core dependencies installed"

# Optional dependencies (graceful fallback)
pip3 install --no-cache-dir sentence-transformers faiss-cpu sse-starlette redis 2>/dev/null || print_warning "Some enhanced features may not be available"

# Download a model
MODEL_COUNT=$(ollama list 2>/dev/null | grep -v "NAME" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    print_info "Downloading Mistral 7B model..."
    if timeout 600 ollama pull mistral:7b-instruct-q4_0; then
        print_status "Model downloaded successfully"
    else
        print_warning "Model download failed/timed out - trying smaller model"
        timeout 300 ollama pull llama3.2:1b || print_warning "No models available"
    fi
else
    print_status "Models already available"
fi

# Create .env configuration
print_info "Creating configuration..."
cat > .env << 'EOF'
# RunPod Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=0.0.0.0:11434

# Authentication - DISABLED
ENABLE_AUTH=false
ENABLE_RATE_LIMITING=false

# CORS - Open
CORS_ORIGINS=["*"]

# Memory
MAX_MEMORY_MB=12288

# Features - Minimal
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=false

# Logging
LOG_LEVEL=INFO
EOF

print_status "Configuration created"

# Start Python service
print_info "Starting Python service..."
python3 main.py > /tmp/service.log 2>&1 &
sleep 8

# Test services
print_info "Testing services..."
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    print_status "✅ All services running successfully!"
    
    # Get health status
    HEALTH_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "healthy")
    print_info "Health status: $HEALTH_STATUS"
    
else
    print_warning "Services may still be starting..."
    print_info "Check logs: tail -f /tmp/service.log"
fi

# Mark setup as completed (CRITICAL - prevents loops)
echo "$(date): Setup completed successfully" > "$SETUP_MARKER"
echo "$(date): Services started" > "$SERVICES_MARKER"

# Create management scripts
cat > start_services.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LLM Proxy Services..."
export OLLAMA_HOST=0.0.0.0:11434
export CUDA_VISIBLE_DEVICES=0

if ! pgrep -f "ollama serve" >/dev/null; then
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
fi

if ! pgrep -f "python.*main" >/dev/null; then
    python3 main.py > /tmp/service.log 2>&1 &
    sleep 3
fi

echo "✅ Services started"
curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
EOF

chmod +x start_services.sh

# Final summary
echo ""
echo "🎯 === SETUP COMPLETE ==="
print_status "LLM Proxy is ready!"

echo ""
echo "🌐 Endpoints:"
echo "• Health: http://localhost:8000/health"
echo "• Docs: http://localhost:8000/docs"
echo "• Chat: http://localhost:8000/v1/chat/completions"
echo "• Ollama: http://localhost:11434"

echo ""
echo "📝 Management:"
echo "• Start services: ./start_services.sh"
echo "• View logs: tail -f /tmp/service.log /tmp/ollama.log"
echo "• Check status: curl http://localhost:8000/health"

echo ""
print_info "🔒 Loop prevention: Setup will not run again"
print_info "🔄 Next runs will only start services"

# Test final status
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    print_status "🎉 SUCCESS: All systems operational!"
else
    print_warning "⏳ Services may still be initializing..."
fi
