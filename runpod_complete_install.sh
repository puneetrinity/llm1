#!/bin/bash
# runpod.sh - Complete RunPod Installation Script for LLM Proxy
# Usage: curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/runpod.sh | bash

set -e

echo "🚀 Enhanced LLM Proxy - RunPod Installation"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Configuration
GITHUB_REPO="https://github.com/YOUR_USERNAME/YOUR_REPO.git"  # UPDATE THIS
WORKSPACE_DIR="/workspace"
PROJECT_DIR="$WORKSPACE_DIR/llm-proxy"

# Move to workspace
cd "$WORKSPACE_DIR"

echo -e "\n${BLUE}📋 Step 1: Environment Check${NC}"

# Check GPU
if nvidia-smi > /dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_status "GPU detected: $GPU_INFO"
    GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    
    if [ "$GPU_MEMORY" -gt 20000 ]; then
        ENABLE_ENHANCED=true
        MAX_MEMORY=16384
    else
        ENABLE_ENHANCED=false
        MAX_MEMORY=8192
    fi
else
    print_warning "No GPU detected - CPU mode"
    ENABLE_ENHANCED=false
    MAX_MEMORY=4096
fi

# Check system memory
TOTAL_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $2}')
print_info "System memory: ${TOTAL_MEMORY}MB"

echo -e "\n${BLUE}📥 Step 2: Download Project${NC}"

# Remove existing directory if present
if [ -d "$PROJECT_DIR" ]; then
    print_warning "Removing existing installation"
    rm -rf "$PROJECT_DIR"
fi

# Clone repository
print_info "Cloning repository..."
git clone "$GITHUB_REPO" "$PROJECT_DIR" || {
    print_error "Failed to clone repository"
    print_error "Make sure to update GITHUB_REPO in the script with your actual repo URL"
    exit 1
}

cd "$PROJECT_DIR"
print_status "Project downloaded"

echo -e "\n${BLUE}🐍 Step 3: Python Setup${NC}"

# Check Python
if ! python3 --version > /dev/null 2>&1; then
    print_error "Python3 not found"
    exit 1
fi

# Install core dependencies system-wide (RunPod friendly)
print_info "Installing Python dependencies..."
pip3 install --upgrade pip --quiet

# Core dependencies
pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    psutil==5.9.6 || {
    print_error "Failed to install core dependencies"
    exit 1
}

print_status "Core dependencies installed"

# Enhanced features if GPU is good
if [ "$ENABLE_ENHANCED" = true ]; then
    print_info "Installing enhanced features..."
    pip3 install --no-cache-dir \
        sentence-transformers \
        scikit-learn \
        redis \
        sse-starlette || {
        print_warning "Enhanced features failed - continuing with basic setup"
        ENABLE_ENHANCED=false
    }
    
    if [ "$ENABLE_ENHANCED" = true ]; then
        print_status "Enhanced features installed"
    fi
fi

echo -e "\n${BLUE}⚙️ Step 4: Configuration${NC}"

# Create optimized .env file
cat > .env << EOF
# RunPod Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434

# Memory Management
MAX_MEMORY_MB=$MAX_MEMORY
CACHE_MEMORY_LIMIT_MB=$((MAX_MEMORY / 8))
MODEL_MEMORY_LIMIT_MB=$((MAX_MEMORY / 2))

# Features
ENABLE_SEMANTIC_CLASSIFICATION=$ENABLE_ENHANCED
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true

# GPU
GPU_MEMORY_FRACTION=0.9
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2

# Security
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-$(openssl rand -hex 16)
CORS_ORIGINS=["*"]
EOF

print_status "Configuration created"

echo -e "\n${BLUE}🤖 Step 5: Ollama Setup${NC}"

# Install Ollama if needed
if ! command -v ollama &> /dev/null; then
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh || {
        print_error "Failed to install Ollama"
        exit 1
    }
    print_status "Ollama installed"
fi

# Start Ollama service
if ! pgrep -f "ollama serve" > /dev/null; then
    print_info "Starting Ollama..."
    ollama serve &
    sleep 10
    
    # Wait for Ollama
    for i in {1..30}; do
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_status "Ollama is ready!"
            break
        fi
        print_info "Waiting for Ollama... ($i/30)"
        sleep 3
    done
    
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_error "Ollama failed to start"
        exit 1
    fi
else
    print_status "Ollama already running"
fi

echo -e "\n${BLUE}📦 Step 6: Model Download${NC}"

# Download essential model
print_info "Downloading Mistral 7B model (this may take a few minutes)..."
ollama pull mistral:7b-instruct-q4_0 || {
    print_error "Failed to download Mistral model"
    exit 1
}
print_status "Mistral 7B model ready"

# Download additional models if enhanced features enabled
if [ "$ENABLE_ENHANCED" = true ]; then
    print_info "Downloading additional models for enhanced features..."
    ollama pull deepseek-v2:7b-q4_0 &
    DEEPSEEK_PID=$!
    
    # Wait for download
    wait $DEEPSEEK_PID || print_warning "DeepSeek download failed"
    print_status "Enhanced models ready"
fi

echo -e "\n${BLUE}🗄️ Step 7: Cache Setup${NC}"

# Install Redis if enhanced features enabled
if [ "$ENABLE_ENHANCED" = true ]; then
    if ! command -v redis-server &> /dev/null; then
        print_info "Installing Redis..."
        apt-get update -qq && apt-get install -y redis-server || {
            print_warning "Redis installation failed - using memory cache"
            ENABLE_ENHANCED=false
        }
    fi
    
    if [ "$ENABLE_ENHANCED" = true ]; then
        # Start Redis
        if ! pgrep redis-server > /dev/null; then
            redis-server --daemonize yes --port 6379 --bind 127.0.0.1
            sleep 2
        fi
        
        if redis-cli ping | grep -q PONG; then
            print_status "Redis cache ready"
        else
            print_warning "Redis failed - using memory cache"
        fi
    fi
else
    print_info "Using memory cache only"
fi

echo -e "\n${BLUE}🚀 Step 8: Application Setup${NC}"

# Create startup script
cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Enhanced LLM Proxy..."

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "📡 Starting Ollama..."
    ollama serve &
    sleep 8
    
    # Wait for Ollama
    for i in {1..15}; do
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama ready"
            break
        fi
        echo "   Waiting... ($i/15)"
        sleep 2
    done
fi

# Start Redis if available
if command -v redis-server &> /dev/null && ! pgrep redis-server > /dev/null; then
    echo "🗄️ Starting Redis..."
    redis-server --daemonize yes --port 6379 --bind 127.0.0.1
    sleep 2
fi

# Warm up model
echo "🔥 Warming up model..."
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral:7b-instruct-q4_0", "messages": [{"role": "user", "content": "Hello"}], "stream": false, "options": {"num_predict": 5}}' \
  >/dev/null 2>&1 || echo "⚠️ Warmup failed (normal on first run)"

# Check port
if ss -tulpn | grep -q ":8000 "; then
    echo "⚠️ Port 8000 in use - killing existing process"
    pkill -f 'python.*main.py' || true
    sleep 3
fi

# Start application
echo "🌐 Starting LLM Proxy on http://localhost:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo "🏥 Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo "===================="

python3 main.py
EOF

chmod +x start.sh
print_status "Startup script created"

echo -e "\n${BLUE}🧪 Step 9: Quick Test${NC}"

# Test the installation
print_info "Testing installation..."

# Start app briefly for test
python3 main.py &
APP_PID=$!
sleep 8

# Test endpoints
HEALTH_TEST=false
ROOT_TEST=false

if timeout 10 curl -f http://localhost:8000/health >/dev/null 2>&1; then
    print_status "Health endpoint working"
    HEALTH_TEST=true
fi

if timeout 10 curl -f http://localhost:8000/ >/dev/null 2>&1; then
    print_status "Root endpoint working"
    ROOT_TEST=true
fi

# Stop test app
kill $APP_PID 2>/dev/null || true
sleep 2

echo -e "\n${GREEN}🎉 Installation Complete!${NC}"
echo "=================================="
echo ""

print_info "Installation Summary:"
echo "   📍 Location: $PROJECT_DIR"
echo "   🧠 GPU Support: $([ "$ENABLE_ENHANCED" = true ] && echo "✅ Enabled" || echo "⏸️ Basic")"
echo "   💾 Memory: ${MAX_MEMORY}MB"
echo "   🤖 Models: Mistral 7B $([ "$ENABLE_ENHANCED" = true ] && echo "+ Enhanced models" || echo "")"
echo "   🗄️ Cache: $([ "$ENABLE_ENHANCED" = true ] && echo "Redis + Memory" || echo "Memory only")"
echo "   🧪 Tests: $([ "$HEALTH_TEST" = true ] && echo "✅ Passed" || echo "⚠️ Check needed")"
echo ""

print_info "Ready to Use:"
echo "1. Start service:  ./start.sh"
echo "2. Test API:       curl http://localhost:8000/health"
echo "3. View docs:      http://localhost:8000/docs"
echo "4. Stop service:   Ctrl+C or pkill -f 'python.*main.py'"
echo ""

if [ "$HEALTH_TEST" = true ] && [ "$ROOT_TEST" = true ]; then
    echo "🚀 All systems ready! Run './start.sh' to begin."
else
    echo "⚠️ Some tests failed - check the installation and try './start.sh'"
fi

echo ""
echo "💡 Pro tip: Run 'screen -S llm-proxy ./start.sh' to run in background"
echo "         Then 'screen -r llm-proxy' to reconnect"
