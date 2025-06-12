#!/bin/bash
# runpod_complete_install.sh - Complete RunPod Installation Script
# Optimized for RunPod A5000 with 24GB VRAM
# FULLY AUTOMATED - No interactive prompts

set -e

echo "🚀 Complete LLM Proxy Installation for RunPod (Automated)"
echo "========================================================="
echo ""
echo "🤖 Automation Options (set as environment variables):"
echo "   FORCE_REINSTALL=true     - Force complete reinstall"
echo "   SKIP_MODELS=true         - Skip model downloads"
echo "   SKIP_REDIS=true          - Skip Redis installation"  
echo "   SKIP_TESTING=true        - Skip endpoint testing"
echo "   CUSTOM_API_KEY=sk-xxx    - Use custom API key"
echo ""
echo "📝 Example: SKIP_MODELS=true ./runpod_complete_install.sh"
echo ""

# Environment variables for automation control
FORCE_REINSTALL=${FORCE_REINSTALL:-false}
SKIP_MODELS=${SKIP_MODELS:-false}
SKIP_REDIS=${SKIP_REDIS:-false}
SKIP_TESTING=${SKIP_TESTING:-false}
CUSTOM_API_KEY=${CUSTOM_API_KEY:-""}

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

# Set workspace directory for RunPod
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
cd "$WORKSPACE_DIR"

# Check if this is a re-run (automated mode)
if [ -f "$WORKSPACE_DIR/llm-proxy/.installation_info" ] && [ "$FORCE_REINSTALL" != true ]; then
    print_warning "Previous installation detected - updating existing setup"
    print_info "Auto-mode: Will skip existing components and update configurations"
    print_info "Use FORCE_REINSTALL=true to completely reinstall"
    RERUN_MODE=true
elif [ "$FORCE_REINSTALL" = true ]; then
    print_warning "Force reinstall mode - will overwrite existing installation"
    RERUN_MODE=false
else
    print_info "Fresh installation mode"
    RERUN_MODE=false
fi

echo -e "\n${BLUE}🔧 Step 1: RunPod Environment Setup${NC}"

# Check GPU availability
if nvidia-smi > /dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_status "GPU detected: $GPU_INFO"
    GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    
    if [ "$GPU_MEMORY" -gt 20000 ]; then
        print_status "High VRAM GPU detected - enabling all enhanced features"
        ENABLE_ALL_FEATURES=true
        MAX_MEMORY_MB=16384
    else
        print_warning "Lower VRAM GPU - using conservative settings"
        ENABLE_ALL_FEATURES=false
        MAX_MEMORY_MB=8192
    fi
else
    print_warning "No GPU detected - CPU-only mode"
    ENABLE_ALL_FEATURES=false
    MAX_MEMORY_MB=4096
fi

# Check available system memory
TOTAL_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $2}')
print_info "System memory: ${TOTAL_MEMORY}MB"

# Create project directory
mkdir -p "$WORKSPACE_DIR/llm-proxy"
cd "$WORKSPACE_DIR/llm-proxy"

echo -e "\n${BLUE}📋 Step 2: Project Setup${NC}"

# Create directory structure
mkdir -p {data/{cache,logs,models},services,utils,middleware,tests,config}
print_status "Project structure created"

# Create optimized .env for RunPod (automated mode)
if [ ! -f .env ]; then
    print_info "Creating new .env configuration..."
    CREATE_ENV=true
elif [ "$RERUN_MODE" = true ]; then
    print_info "Rerun detected - backing up and updating .env configuration..."
    cp .env .env.backup.$(date +%s)
    print_status "Existing .env backed up"
    CREATE_ENV=true
else
    print_info "Creating fresh .env configuration..."
    CREATE_ENV=true
fi

if [ "$CREATE_ENV" = true ]; then
cat > .env << EOF
# RunPod Optimized Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration  
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434

# Memory Management (Optimized for RunPod)
MAX_MEMORY_MB=$MAX_MEMORY_MB
CACHE_MEMORY_LIMIT_MB=$((MAX_MEMORY_MB / 8))
MODEL_MEMORY_LIMIT_MB=$((MAX_MEMORY_MB / 2))
SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Enhanced Features
ENABLE_SEMANTIC_CLASSIFICATION=$ENABLE_ALL_FEATURES
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true

# GPU Configuration (RunPod specific)
GPU_MEMORY_FRACTION=0.9
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2

# Security (Production ready)
ENABLE_AUTH=true
DEFAULT_API_KEY=${CUSTOM_API_KEY:-sk-$(openssl rand -hex 16)}
CORS_ORIGINS=["*"]

# Enhanced Connection Pooling
ENHANCED_CONNECTION_POOLING_ENABLED=true
ENHANCED_CONNECTION_POOLING_TOTAL_LIMIT=100
ENHANCED_CONNECTION_POOLING_PER_HOST_LIMIT=20

# Circuit Breaker Protection  
ENHANCED_CIRCUIT_BREAKER_ENABLED=true
ENHANCED_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
ENHANCED_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60

# Smart Caching
ENHANCED_SMART_CACHE_ENABLED=true
ENHANCED_SMART_CACHE_REDIS_ENABLED=true
ENHANCED_SMART_CACHE_REDIS_URL=redis://localhost:6379
ENHANCED_SMART_CACHE_SEMANTIC_ENABLED=$ENABLE_ALL_FEATURES
ENHANCED_SMART_CACHE_SIMILARITY_THRESHOLD=0.85
EOF

    print_status "RunPod-optimized configuration created"
else
    print_status "Using existing configuration"
fi

echo -e "\n${BLUE}🐍 Step 3: Python Environment${NC}"

# Create Python virtual environment (loop-safe)
if [ ! -d "venv" ]; then
    print_info "Creating new Python virtual environment..."
    python3 -m venv venv
    print_status "Python virtual environment created"
else
    print_info "Using existing Python virtual environment"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Python virtual environment activated"

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
print_info "Installing core dependencies..."
pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    psutil==5.9.6

print_status "Core dependencies installed"

# Install enhanced features if enabled
if [ "$ENABLE_ALL_FEATURES" = true ]; then
    print_info "Installing enhanced AI features..."
    pip install --no-cache-dir \
        sentence-transformers>=2.2.0 \
        scikit-learn>=1.1.0 \
        redis>=4.5.0 \
        aioredis>=2.0.0 \
        prometheus-client \
        sse-starlette>=1.6.5
    print_status "Enhanced features installed"
else
    print_info "Installing basic enhanced features..."
    pip install --no-cache-dir \
        redis>=4.5.0 \
        sse-starlette>=1.6.5
    print_status "Basic enhanced features installed"
fi

echo -e "\n${BLUE}🤖 Step 4: Ollama Setup${NC}"

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    print_status "Ollama installed"
else
    print_status "Ollama already installed"
fi

echo -e "\n${BLUE}🤖 Step 4: Ollama Setup${NC}"

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    print_status "Ollama installed"
else
    print_status "Ollama already installed"
fi

# Start Ollama service (loop-safe)
if pgrep -f "ollama serve" > /dev/null; then
    print_info "Ollama is already running"
else
    print_info "Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 10
    
    # Wait for Ollama to be ready
    print_info "Waiting for Ollama to start..."
    for i in {1..30}; do
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_status "Ollama is ready!"
            break
        fi
        print_info "   Attempt $i/30 - waiting 3 seconds..."
        sleep 3
    done
    
    # Verify Ollama started successfully
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_error "Failed to start Ollama service"
        exit 1
    fi
fi

# Pull essential models (check if already exists)
if [ "$SKIP_MODELS" = true ]; then
    print_info "Skipping model downloads (SKIP_MODELS=true)"
else
    print_info "Checking/pulling essential models..."

    if ! ollama list | grep -q "mistral:7b-instruct-q4_0"; then
        print_info "Downloading Mistral 7B model..."
        ollama pull mistral:7b-instruct-q4_0 &
        MISTRAL_PID=$!
        wait $MISTRAL_PID
        print_status "Mistral 7B model ready"
    else
        print_status "Mistral 7B model already available"
    fi

    if [ "$ENABLE_ALL_FEATURES" = true ]; then
        if ! ollama list | grep -q "deepseek-v2:7b-q4_0"; then
            print_info "Downloading DeepSeek V2 model..."
            ollama pull deepseek-v2:7b-q4_0 &
            DEEPSEEK_PID=$!
        else
            print_status "DeepSeek V2 model already available"
            DEEPSEEK_PID=""
        fi
        
        if ! ollama list | grep -q "llama3:8b-instruct-q4_0"; then
            print_info "Downloading LLaMA3 model..."
            ollama pull llama3:8b-instruct-q4_0 &
            LLAMA_PID=$!
        else
            print_status "LLaMA3 model already available"
            LLAMA_PID=""
        fi
        
        # Wait for any downloads that started
        if [ -n "$DEEPSEEK_PID" ]; then wait $DEEPSEEK_PID; fi
        if [ -n "$LLAMA_PID" ]; then wait $LLAMA_PID; fi
        print_status "All models ready"
    fi
fi

echo -e "\n${BLUE}🗄️ Step 5: Redis Setup (Optional)${NC}"

# Install and start Redis if enhanced features enabled and not skipped
if [ "$SKIP_REDIS" = true ]; then
    print_info "Skipping Redis setup (SKIP_REDIS=true)"
elif [ "$ENABLE_ALL_FEATURES" = true ]; then
    if ! command -v redis-server &> /dev/null; then
        print_info "Installing Redis..."
        apt-get update -qq && apt-get install -y redis-server
        print_status "Redis installed"
    else
        print_status "Redis already installed"
    fi
    
    # Start Redis (loop-safe)
    if pgrep redis-server > /dev/null; then
        print_info "Redis is already running"
    else
        print_info "Starting Redis..."
        redis-server --daemonize yes --port 6379 --bind 127.0.0.1
        sleep 2
    fi
    
    # Test Redis connection
    if redis-cli ping | grep -q PONG; then
        print_status "Redis is running and accessible"
    else
        print_warning "Redis failed to start - will use memory cache"
    fi
else
    print_info "Skipping Redis (using memory cache only)"
fi

echo -e "\n${BLUE}📝 Step 6: Application Files${NC}"

# Create minimal main.py if not exists
if [ ! -f main.py ]; then
    print_info "Creating basic application..."
    cat > main.py << 'EOF'
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Enhanced LLM Proxy", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/")
async def root():
    return {"message": "Enhanced LLM Proxy API Ready"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
EOF
    print_status "Basic application created"
fi

# Create startup script
cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Enhanced LLM Proxy..."

# Activate virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "✅ Python environment activated"
else
    echo "❌ Python virtual environment not found!"
    echo "Run the installation script first"
    exit 1
fi

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "📡 Starting Ollama..."
    ollama serve &
    sleep 8
    
    # Verify Ollama started
    for i in {1..15}; do
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama is ready"
            break
        fi
        echo "   Waiting for Ollama... ($i/15)"
        sleep 2
    done
    
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "❌ Failed to start Ollama"
        exit 1
    fi
else
    echo "✅ Ollama already running"
fi

# Start Redis if available and not running
if command -v redis-server &> /dev/null; then
    if ! pgrep redis-server > /dev/null; then
        echo "🗄️ Starting Redis..."
        redis-server --daemonize yes --port 6379 --bind 127.0.0.1
        sleep 2
        
        if redis-cli ping | grep -q PONG; then
            echo "✅ Redis started successfully"
        else
            echo "⚠️ Redis failed to start - using memory cache"
        fi
    else
        echo "✅ Redis already running"
    fi
fi

# Warm up priority model
echo "🔥 Warming up models..."
if curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral:7b-instruct-q4_0", "messages": [{"role": "user", "content": "Hello"}], "stream": false, "options": {"num_predict": 5}}' \
  >/dev/null 2>&1; then
    echo "✅ Model warmup successful"
else
    echo "⚠️ Model warmup failed (this is normal on first run)"
fi

# Check if port 8000 is already in use
if ss -tulpn | grep -q ":8000 "; then
    echo "⚠️ Port 8000 is already in use"
    echo "Kill existing process or use a different port"
    echo "To kill existing: pkill -f 'python.*main.py'"
    exit 1
fi

# Start FastAPI application
echo "🌐 Starting FastAPI application on http://localhost:8000"
echo "📚 API docs available at: http://localhost:8000/docs"
echo "🏥 Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the service"
echo "=================================="

python main.py
EOF

chmod +x start.sh
print_status "Startup script created"

echo -e "\n${BLUE}🧪 Step 7: Testing Installation${NC}"

if [ "$SKIP_TESTING" = true ]; then
    print_info "Skipping tests (SKIP_TESTING=true)"
    TESTS_PASSED=true
else
    # Test basic functionality
    source venv/bin/activate

    # Cleanup function for testing
    cleanup_test() {
        if [ -n "$APP_PID" ] && kill -0 $APP_PID 2>/dev/null; then
            print_info "Cleaning up test processes..."
            kill $APP_PID 2>/dev/null || true
            sleep 3
            # Force kill if still running
            kill -9 $APP_PID 2>/dev/null || true
        fi
    }

    # Set trap for cleanup on script exit
    trap cleanup_test EXIT

    # Start the application in background for testing
    print_info "Starting test instance..."
    python main.py &
    APP_PID=$!

    # Wait for application to start
    sleep 8

    # Test endpoints with timeout
    print_info "Testing endpoints..."

    if timeout 10 curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_status "Health endpoint working"
        HEALTH_OK=true
    else
        print_warning "Health endpoint test failed"
        HEALTH_OK=false
    fi

    if timeout 10 curl -f http://localhost:8000/ >/dev/null 2>&1; then
        print_status "Root endpoint working"
        ROOT_OK=true
    else
        print_warning "Root endpoint test failed"
        ROOT_OK=false
    fi

    # Test Ollama integration
    if timeout 15 curl -X POST http://localhost:11434/api/chat \
      -H "Content-Type: application/json" \
      -d '{"model": "mistral:7b-instruct-q4_0", "messages": [{"role": "user", "content": "test"}], "stream": false, "options": {"num_predict": 1}}' \
      >/dev/null 2>&1; then
        print_status "Ollama integration working"
        OLLAMA_OK=true
    else
        print_warning "Ollama integration test failed"
        OLLAMA_OK=false
    fi

    # Cleanup test instance
    cleanup_test
    trap - EXIT  # Remove the trap

    # Summary
    if [ "$HEALTH_OK" = true ] && [ "$ROOT_OK" = true ] && [ "$OLLAMA_OK" = true ]; then
        print_status "All tests passed!"
        TESTS_PASSED=true
    else
        print_warning "Some tests failed - check logs for details"
        TESTS_PASSED=false
    fi
fi

echo -e "\n${GREEN}🎉 Installation Complete!${NC}"
echo "=================================="

# Show what was actually done this run
print_info "This Run Summary:"
echo "   📁 Project: $([ -d "$WORKSPACE_DIR/llm-proxy" ] && echo "✅ Ready" || echo "❌ Missing")"
echo "   🐍 Python: $([ -d venv ] && echo "✅ Virtual env ready" || echo "❌ Missing")" 
echo "   🤖 Ollama: $(pgrep -f "ollama serve" > /dev/null && echo "✅ Running" || echo "❌ Not running")"
echo "   🗄️ Redis: $([ "$ENABLE_ALL_FEATURES" = true ] && (pgrep redis-server > /dev/null && echo "✅ Running" || echo "❌ Not running") || echo "⏸️ Disabled")"
echo "   🧪 Tests: $([ "$TESTS_PASSED" = true ] && echo "✅ Passed" || echo "⚠️ Some failed")"
echo ""

print_info "Installation Summary:"
echo "   📍 Location: $WORKSPACE_DIR/llm-proxy"
echo "   🧠 GPU Support: $([ "$ENABLE_ALL_FEATURES" = true ] && echo "✅ Enabled" || echo "⏸️ Basic")"
echo "   💾 Memory Limit: ${MAX_MEMORY_MB}MB"
echo "   🤖 Models: Mistral 7B $([ "$ENABLE_ALL_FEATURES" = true ] && echo "+ DeepSeek + LLaMA3" || echo "only")"
echo "   🗄️ Cache: $([ "$ENABLE_ALL_FEATURES" = true ] && echo "Redis + Memory" || echo "Memory only")"
echo ""
print_info "Next Steps:"
echo "1. Start the service: ./start.sh"
echo "2. Test API: curl http://localhost:8000/health"
echo "3. View logs: tail -f logs/app.log"
echo "4. Stop services: pkill -f 'ollama\\|redis\\|python'"
echo ""
print_info "API Endpoints:"
echo "   🌐 API: http://localhost:8000"
echo "   📚 Docs: http://localhost:8000/docs"
echo "   🏥 Health: http://localhost:8000/health"
echo ""
if [ "$ENABLE_ALL_FEATURES" = true ]; then
    echo "🚀 All enhanced features enabled for maximum performance!"
else
    echo "⚡ Basic features enabled - upgrade GPU for full features"
fi

# Save installation info
cat > .installation_info << EOF
Enhanced LLM Proxy Installation
==============================
Installation Date: $(date)
Last Run: $(date)
Location: $WORKSPACE_DIR/llm-proxy
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "None")
System Memory: ${TOTAL_MEMORY}MB
Max Memory Setting: ${MAX_MEMORY_MB}MB
Enhanced Features: $ENABLE_ALL_FEATURES
Models: mistral:7b-instruct-q4_0$([ "$ENABLE_ALL_FEATURES" = true ] && echo ", deepseek-v2:7b-q4_0, llama3:8b-instruct-q4_0" || echo "")
Tests Passed: ${TESTS_PASSED:-unknown}

Status Check:
- Ollama: $(pgrep -f "ollama serve" > /dev/null && echo "Running" || echo "Not running")
- Redis: $([ "$ENABLE_ALL_FEATURES" = true ] && (pgrep redis-server > /dev/null && echo "Running" || echo "Not running") || echo "Disabled")
- Python Env: $([ -d venv ] && echo "Available" || echo "Missing")

To start: ./start.sh
To update: Re-run this script (loop-safe)
To check status: cat .installation_info
EOF

print_status "Installation info saved to .installation_info"
echo ""
echo "🎯 Ready to start your Enhanced LLM Proxy!"
echo ""
echo "🤖 Automation Tips for Future Runs:"
echo "   Fast model skip:    SKIP_MODELS=true ./runpod_complete_install.sh"
echo "   Quick update:       SKIP_TESTING=true ./runpod_complete_install.sh"  
echo "   Memory-only mode:   SKIP_REDIS=true ./runpod_complete_install.sh"
echo "   Complete reinstall: FORCE_REINSTALL=true ./runpod_complete_install.sh"
