#!/bin/bash
# start.sh - Enhanced LLM Proxy Startup Script
# Maintainable startup process with proper error handling

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🚀 Starting Comprehensive Enhanced LLM Proxy..."
echo "=============================================="

# Export GPU environment variables
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export OLLAMA_HOST=0.0.0.0:11434

# Verify GPU detection
log_info "Checking GPU availability..."
if nvidia-smi; then
    log_success "GPU detected and accessible"
else
    log_warning "GPU detection may have issues - continuing with CPU mode"
fi

# Start Ollama service
log_info "Starting Ollama service..."
CUDA_VISIBLE_DEVICES=0 ollama serve &
OLLAMA_PID=$!

# Wait for Ollama with comprehensive timeout
log_info "Waiting for Ollama to start (up to 5 minutes)..."
OLLAMA_READY=false
for i in {1..60}; do
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        log_success "Ollama is ready!"
        OLLAMA_READY=true
        break
    fi
    echo "   Attempt $i/60 - waiting 5 seconds..."
    sleep 5
done

# Verify Ollama started successfully
if [ "$OLLAMA_READY" = false ]; then
    log_error "Failed to start Ollama service after 5 minutes"
    log_info "Attempting fallback startup..."
    pkill ollama || true
    sleep 5
    ollama serve &
    sleep 30
    
    # Final check
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        log_error "Ollama fallback startup also failed"
        log_warning "Continuing without Ollama - some features will be limited"
    else
        log_success "Ollama fallback startup successful"
    fi
else
    log_success "Ollama service started successfully"
fi

# Pull essential models in background
log_info "Pulling essential models in background..."
{
    CUDA_VISIBLE_DEVICES=0 ollama pull mistral:7b-instruct-q4_0 >/dev/null 2>&1 &
} || {
    log_warning "Model pull failed - models will be downloaded on first use"
}

# Create default .env if not exists
if [ ! -f .env ]; then
    log_info "Creating default environment configuration..."
    echo 'PORT=8001' > .env
    log_success "Default .env created"
fi

# Verify application files
log_info "Verifying application structure..."
if [ ! -f "main_master.py" ] && [ ! -f "main.py" ]; then
    log_error "Main application file not found!"
    log_info "Looking for alternative entry points..."
    
    # Try to find any Python main file
    MAIN_FILE=$(find . -name "*main*.py" -o -name "app.py" -o -name "server.py" | head -1)
    if [ -n "$MAIN_FILE" ]; then
        log_warning "Using alternative entry point: $MAIN_FILE"
        PYTHON_MAIN="$MAIN_FILE"
    else
        log_error "No suitable Python entry point found"
        exit 1
    fi
else
    # Use the standard main file
    if [ -f "main_master.py" ]; then
        PYTHON_MAIN="main_master.py"
    else
        PYTHON_MAIN="main.py"
    fi
fi

# Final system status
log_info "System Status Summary:"
echo "   • GPU Available: $(nvidia-smi >/dev/null 2>&1 && echo "Yes" || echo "No")"
echo "   • Ollama Ready: $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "Yes" || echo "No")"
echo "   • Python Entry: $PYTHON_MAIN"
echo "   • Memory Limit: ${MAX_MEMORY_MB:-12288}MB"
echo "   • Enhanced Features: $([ "$ENABLE_SEMANTIC_CLASSIFICATION" = "true" ] && echo "Enabled" || echo "Disabled")"

# Start the comprehensive FastAPI application
log_success "Starting Enhanced FastAPI application..."
log_info "✅ System Ready: http://localhost:8001"
log_info "📚 API Documentation: http://localhost:8001/docs"
log_info "🏥 Health Check: http://localhost:8001/health"
log_info "📊 Metrics: http://localhost:8001/metrics"

echo ""
echo "=============================================="
echo "🌐 Enhanced LLM Proxy is now running!"
echo "=============================================="

# Execute the main application
exec python3 "$PYTHON_MAIN"
