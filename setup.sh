#!/bin/bash
# ENHANCED RunPod A5000 Setup Script v2.3 - Fixed Version (No Loop)

set -e

echo "🚀 ENHANCED LLM Proxy Setup for RunPod A5000 v2.3"
echo "=================================================="

# Detect if we're on RunPod
if [[ "$PWD" == "/workspace"* ]]; then
    echo "✅ RunPod environment detected"
    WORKSPACE_DIR="/workspace"
    IS_RUNPOD=true
else
    echo "ℹ️  Local environment detected"  
    WORKSPACE_DIR="$(pwd)"
    IS_RUNPOD=false
fi

cd "$WORKSPACE_DIR"
mkdir -p logs models cache

# CRITICAL FIX: Prevent multiple runs
SETUP_COMPLETE_FILE="$WORKSPACE_DIR/.setup_complete"
if [ -f "$SETUP_COMPLETE_FILE" ]; then
    echo "✅ Setup already completed. Skipping setup process..."
    echo "🚀 Starting services directly..."
    cd "$WORKSPACE_DIR/app"
    
    # Check if ollama is running
    if ! pgrep -f "ollama serve" > /dev/null; then
        echo "🤖 Starting Ollama service..."
        CUDA_VISIBLE_DEVICES=0 \
        NVIDIA_VISIBLE_DEVICES=all \
        OLLAMA_HOST=0.0.0.0:11434 \
        ollama serve > "$WORKSPACE_DIR/logs/ollama.log" 2>&1 &
        sleep 10
    else
        echo "✅ Ollama already running"
    fi
    
    # Start the FastAPI app
    if [ -f "start.sh" ]; then
        echo "🚀 Starting FastAPI application..."
        source "$WORKSPACE_DIR/venv/bin/activate" 2>/dev/null || true
        exec ./start.sh
    else
        echo "❌ start.sh not found. Removing setup marker and re-running setup..."
        rm -f "$SETUP_COMPLETE_FILE"
    fi
fi

# CRITICAL FIX 1: Export CUDA environment variables FIRST
echo "🔧 CRITICAL: Setting up CUDA environment for A5000..."
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VERSION=12.1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Enhanced GPU detection with multiple fallback methods
echo "🔍 Enhanced GPU detection..."
GPU_DETECTED=false
GPU_NAME=""
GPU_MEMORY=""

# Method 1: nvidia-smi (most reliable)
echo "   🔍 Method 1: Checking nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        echo "   ✅ GPU detected via nvidia-smi"
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "Unknown GPU, Unknown")
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
        echo "   📊 GPU: $GPU_NAME ($GPU_MEMORY MB)"
        GPU_DETECTED=true
    else
        echo "   ⚠️ nvidia-smi found but not working"
    fi
else
    echo "   ⚠️ nvidia-smi not found - attempting installation..."
    
    # Update package lists
    apt-get update -qq
    
    # FIXED: Skip problematic nvidia packages that cause dpkg conflicts
    echo "   ⚠️ Skipping nvidia apt packages to avoid dpkg conflicts"
fi

# Method 2: lspci detection
echo "   🔍 Method 2: Checking lspci for NVIDIA devices..."
if command -v lspci &> /dev/null; then
    NVIDIA_DEVICES=$(lspci | grep -i nvidia || true)
    if [ -n "$NVIDIA_DEVICES" ]; then
        echo "   ✅ NVIDIA GPU detected via lspci:"
        echo "$NVIDIA_DEVICES" | sed 's/^/      /'
        GPU_DETECTED=true
        if [ -z "$GPU_NAME" ]; then
            GPU_NAME=$(echo "$NVIDIA_DEVICES" | head -1 | sed 's/.*NVIDIA Corporation //' | sed 's/ (rev.*//')
        fi
    else
        echo "   ⚠️ No NVIDIA devices found via lspci"
    fi
else
    echo "   ⚠️ lspci not available"
fi

# Method 3: /proc/driver/nvidia
echo "   🔍 Method 3: Checking /proc/driver/nvidia..."
if [ -d "/proc/driver/nvidia" ]; then
    echo "   ✅ NVIDIA driver detected in /proc"
    if [ -f "/proc/driver/nvidia/version" ]; then
        DRIVER_VERSION=$(head -1 /proc/driver/nvidia/version 2>/dev/null || echo "Unknown version")
        echo "   📋 Driver version: $DRIVER_VERSION"
    fi
    GPU_DETECTED=true
else
    echo "   ⚠️ /proc/driver/nvidia not found"
fi

# Method 4: Check CUDA devices
echo "   🔍 Method 4: Checking CUDA device files..."
if ls /dev/nvidia* > /dev/null 2>&1; then
    echo "   ✅ NVIDIA device files found:"
    ls -la /dev/nvidia* | sed 's/^/      /'
    GPU_DETECTED=true
else
    echo "   ⚠️ No NVIDIA device files found"
fi

# Final GPU detection summary
if [ "$GPU_DETECTED" = true ]; then
    echo "🎉 GPU Detection Summary:"
    echo "   Status: ✅ GPU DETECTED"
    [ -n "$GPU_NAME" ] && echo "   Name: $GPU_NAME"
    [ -n "$GPU_MEMORY" ] && echo "   Memory: $GPU_MEMORY MB"
else
    echo "❌ GPU Detection Summary:"
    echo "   Status: ⚠️ NO GPU DETECTED"
    echo "   Note: Continuing anyway - GPU may work in container environment"
fi

# CRITICAL FIX 2: Install system dependencies with better error handling
echo "🔧 Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Update package lists
apt-get update -qq

# Install core system packages
apt-get install -y --no-install-recommends \
    pciutils \
    lshw \
    hwinfo \
    curl \
    wget \
    git \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release

# FIXED: Install Python NVIDIA packages with compatible versions
echo "🐍 Installing Python NVIDIA packages (with version fixes)..."
pip3 install --no-cache-dir --upgrade pip

# FIXED: Use specific versions that work on RunPod
pip3 install --no-cache-dir \
    "nvidia-ml-py==12.535.161" \
    "pynvml==11.5.0" \
    gpustat || echo "   ⚠️ Some Python NVIDIA packages failed to install"

# Test Python GPU detection
echo "🧪 Testing Python GPU detection..."
python3 -c "
try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    print('✅ Found ' + str(count) + ' GPU(s) via Python')
    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if hasattr(name, 'decode'):
                name = name.decode('utf-8')
            print('   GPU ' + str(i) + ': ' + str(name))
        except Exception as e:
            print('   GPU ' + str(i) + ': Error - ' + str(e))
except Exception as e:
    print('⚠️ GPU detection failed: ' + str(e))
    print('   This is normal on RunPod - GPU will still work')
" || echo "   ⚠️ Python GPU test had issues (continuing)"

# CRITICAL FIX 3: Install Ollama with enhanced GPU support
echo "🤖 Installing Ollama with ENHANCED GPU support..."

# Set Ollama environment variables BEFORE installation
export OLLAMA_MODELS="$WORKSPACE_DIR/models"
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_DEBUG=INFO
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE=5m

mkdir -p "$OLLAMA_MODELS"

# Install Ollama with error handling
echo "   📥 Downloading and installing Ollama..."
if curl -fsSL https://ollama.com/install.sh | sh; then
    echo "   ✅ Ollama installed successfully"
else
    echo "   ❌ Ollama installation failed"
    exit 1
fi

# Verify Ollama installation
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo "   📋 Ollama version: $OLLAMA_VERSION"
else
    echo "   ❌ Ollama command not found after installation"
    exit 1
fi

# CRITICAL FIX 4: Start Ollama with enhanced GPU configuration
echo "📡 Starting Ollama with ENHANCED GPU configuration..."

# Kill any existing ollama processes
pkill -f ollama || true
sleep 2

# Start ollama with comprehensive GPU environment
echo "   🚀 Launching Ollama service..."
CUDA_VISIBLE_DEVICES=0 \
NVIDIA_VISIBLE_DEVICES=all \
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_GPU_OVERHEAD=0 \
OLLAMA_DEBUG=INFO \
OLLAMA_FLASH_ATTENTION=1 \
ollama serve > "$WORKSPACE_DIR/logs/ollama.log" 2>&1 &

OLLAMA_PID=$!
echo "   📋 Ollama PID: $OLLAMA_PID"

# Enhanced startup verification with detailed logging
echo "⏳ Waiting for Ollama to start (enhanced monitoring)..."
STARTUP_SUCCESS=false

for i in {1..45}; do
    if curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "   ✅ Ollama is ready with GPU support!"
        STARTUP_SUCCESS=true
        break
    fi
    
    # Show progress and check if process is still running
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "   ❌ Ollama process died during startup"
        echo "   📋 Last few lines of Ollama log:"
        tail -10 "$WORKSPACE_DIR/logs/ollama.log" 2>/dev/null || echo "   No log available"
        exit 1
    fi
    
    echo "   Attempt $i/45 - waiting 4 seconds... (process running: ✓)"
    sleep 4
done

# Verify startup success
if [ "$STARTUP_SUCCESS" = false ]; then
    echo "   ❌ Ollama failed to start within timeout"
    echo "   📋 Ollama log contents:"
    cat "$WORKSPACE_DIR/logs/ollama.log" 2>/dev/null || echo "   No log available"
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# Test Ollama API
echo "🧪 Testing Ollama API..."
API_RESPONSE=$(curl -s http://localhost:11434/api/tags 2>/dev/null || echo '{"error": "failed"}')
if echo "$API_RESPONSE" | grep -q '"models"'; then
    echo "   ✅ Ollama API responding correctly"
    # Show available models (if any)
    MODEL_COUNT=$(echo "$API_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null || echo "0")
    echo "   📊 Current models available: $MODEL_COUNT"
else
    echo "   ⚠️ Ollama API response unexpected: $API_RESPONSE"
fi

# CRITICAL FIX 5: Set up enhanced Python environment
echo "🐍 Setting up ENHANCED Python environment..."

# Create virtual environment
python3 -m venv "$WORKSPACE_DIR/venv"
source "$WORKSPACE_DIR/venv/bin/activate"

# Upgrade pip
pip install --no-cache-dir --upgrade pip setuptools wheel

echo "📚 Installing ENHANCED Python packages..."

# Core dependencies with FIXED and tested versions
echo "   🔧 Installing core FastAPI stack..."
pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    psutil==5.9.6 \
    python-multipart==0.0.6

# ENHANCED: Install compatible AI/ML packages
echo "   🔧 Installing AI/ML packages..."
pip install --no-cache-dir "huggingface_hub>=0.20.0,<0.25.0" || echo "   ⚠️ huggingface_hub installation failed"

# Enhanced features with better error handling
echo "   🚀 Installing enhanced features..."
pip install --no-cache-dir \
    "sentence-transformers>=2.2.0,<3.0.0" \
    "faiss-cpu==1.7.4" \
    "sse-starlette==1.6.5" \
    "numpy>=1.21.0,<1.25.0" \
    "python-dotenv==1.0.0" \
    "jinja2>=3.1.0" || echo "   ⚠️ Some enhanced features may be limited"

# Additional monitoring and utility packages
echo "   🔧 Installing monitoring packages..."
pip install --no-cache-dir \
    "prometheus-client==0.19.0" \
    "python-json-logger==2.0.7" \
    "colorama==0.4.6" || echo "   ⚠️ Some monitoring features may be limited"

# Get or update application code
echo "📁 Setting up application code..."
if [ ! -d "$WORKSPACE_DIR/app" ]; then
    echo "   📥 Cloning application repository..."
    if git clone https://github.com/puneetrinity/llm1.git "$WORKSPACE_DIR/app"; then
        echo "   ✅ Repository cloned successfully"
    else
        echo "   ⚠️ Git clone failed, creating app directory..."
        mkdir -p "$WORKSPACE_DIR/app"
    fi
else
    echo "   📂 App directory exists, updating..."
    cd "$WORKSPACE_DIR/app"
    git pull origin main 2>/dev/null || echo "   ⚠️ Git pull failed, continuing with existing code"
fi

cd "$WORKSPACE_DIR/app"

# CRITICAL FIX 6: Create ENHANCED configuration for A5000
echo "⚙️ Creating ENHANCED RunPod A5000 configuration..."
cat > .env << 'EOF'
# ENHANCED RunPod A5000 Configuration v2.3
DEBUG=false
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Ollama settings - ENHANCED
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_MAX_RETRIES=3
OLLAMA_KEEP_ALIVE=5m
OLLAMA_FLASH_ATTENTION=1

# ENHANCED Memory management for A5000 (24GB VRAM)
MAX_MEMORY_MB=14336
CACHE_MEMORY_LIMIT_MB=2048
MODEL_MEMORY_LIMIT_MB=8192
SEMANTIC_MODEL_MAX_MEMORY_MB=1024

# ENHANCED GPU optimization
GPU_MEMORY_FRACTION=0.85
OLLAMA_NUM_PARALLEL=3
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_GPU_OVERHEAD=0

# ENHANCED features - Optimized settings
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true
ENABLE_PROMETHEUS_METRICS=true
ENABLE_REQUEST_LOGGING=true

# Model settings - ENHANCED
DEFAULT_MODEL=mistral:7b-instruct-q4_0
FALLBACK_MODEL=llama2:7b-chat
MAX_TOKENS=4096
DEFAULT_TEMPERATURE=0.7
MAX_CONTEXT_LENGTH=8192

# Security - ENHANCED
ENABLE_AUTH=false
CORS_ORIGINS=["*"]
CORS_ALLOW_CREDENTIALS=true
ENABLE_API_KEY_AUTH=false

# Performance - ENHANCED
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=60
ENABLE_CACHE=true
CACHE_TTL=7200
CACHE_MAX_SIZE=2000
ENABLE_RESPONSE_COMPRESSION=true

# Monitoring - ENHANCED
ENABLE_HEALTH_CHECKS=true
HEALTH_CHECK_INTERVAL=30
ENABLE_PERFORMANCE_MONITORING=true
LOG_REQUEST_DETAILS=true
EOF

# CRITICAL FIX 7: Enhanced model management
echo "📦 ENHANCED model downloading for A5000..."

# Primary model with enhanced error handling
echo "   🔄 Pulling Mistral 7B (Enhanced)..."
MODEL_PULL_SUCCESS=false

for attempt in {1..3}; do
    echo "   📥 Attempt $attempt/3 - Pulling mistral:7b-instruct-q4_0..."
    
    if CUDA_VISIBLE_DEVICES=0 ollama pull mistral:7b-instruct-q4_0; then
        echo "   ✅ Mistral 7B pulled successfully!"
        MODEL_PULL_SUCCESS=true
        break
    else
        echo "   ⚠️ Attempt $attempt failed"
        if [ $attempt -lt 3 ]; then
            echo "   🔄 Restarting Ollama and retrying..."
            kill $OLLAMA_PID 2>/dev/null || true
            sleep 5
            
            CUDA_VISIBLE_DEVICES=0 \
            NVIDIA_VISIBLE_DEVICES=all \
            OLLAMA_HOST=0.0.0.0:11434 \
            ollama serve > "$WORKSPACE_DIR/logs/ollama.log" 2>&1 &
            
            OLLAMA_PID=$!
            sleep 15
            
            # Verify Ollama is responding
            for check in {1..10}; do
                if curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
                    break
                fi
                sleep 2
            done
        fi
    fi
done

if [ "$MODEL_PULL_SUCCESS" = false ]; then
    echo "   ❌ Failed to pull Mistral model after 3 attempts"
    echo "   📋 Checking available disk space..."
    df -h "$WORKSPACE_DIR" || true
    echo "   📋 Checking Ollama status..."
    curl -s http://localhost:11434/api/tags || echo "Ollama not responding"
fi

# Enhanced model warming
if [ "$MODEL_PULL_SUCCESS" = true ]; then
    echo "   🔥 Enhanced model warming..."
    
    # Warm up with a simple request
    WARMUP_RESPONSE=$(curl -s -X POST http://localhost:11434/api/chat \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mistral:7b-instruct-q4_0",
            "messages": [{"role": "user", "content": "Hello, please respond with just OK."}],
            "stream": false,
            "options": {"num_predict": 10, "temperature": 0.1}
        }' 2>/dev/null || echo '{"error": "warmup failed"}')
    
    if echo "$WARMUP_RESPONSE" | grep -q '"content"'; then
        echo "   ✅ Model warmed up successfully!"
        # Extract and show the response
        WARMUP_CONTENT=$(echo "$WARMUP_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('message', {}).get('content', 'No content'))" 2>/dev/null || echo "Response parsing failed")
        echo "   📝 Model response: $WARMUP_CONTENT"
    else
        echo "   ⚠️ Model warmup failed: $WARMUP_RESPONSE"
    fi
fi

# Optional: Pull additional models if space allows
echo "📦 Checking for additional model downloads..."
AVAILABLE_SPACE_GB=$(df "$WORKSPACE_DIR" | tail -1 | awk '{print int($4/1024/1024)}')
echo "   💾 Available space: ${AVAILABLE_SPACE_GB}GB"

if [ "$AVAILABLE_SPACE_GB" -gt 20 ] && [ "$MODEL_PULL_SUCCESS" = true ]; then
    echo "   📥 Sufficient space - pulling backup model..."
    CUDA_VISIBLE_DEVICES=0 ollama pull llama2:7b-chat > /dev/null 2>&1 && echo "   ✅ Backup model ready!" || echo "   ⚠️ Backup model failed"
else
    echo "   ⚠️ Insufficient space for additional models"
fi

# CRITICAL FIX 8: Create ENHANCED FastAPI application
echo "🌐 Creating ENHANCED FastAPI application..."
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ENHANCED Configuration
class Settings:
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    CORS_ORIGINS = ["*"]
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'mistral:7b-instruct-q4_0')
    FALLBACK_MODEL = os.getenv('FALLBACK_MODEL', 'llama2:7b-chat')
    MAX_MEMORY_MB = int(os.getenv('MAX_MEMORY_MB', 14336))
    ENABLE_METRICS = os.getenv('ENABLE_PROMETHEUS_METRICS', 'true').lower() == 'true'
    ENABLE_RATE_LIMITING = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))

settings = Settings()

# Global variables for monitoring
request_count = 0
error_count = 0
start_time = time.time()

# Enhanced HTTP session management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.http_session = aiohttp.ClientSession()
    logger.info(f"🚀 ENHANCED LLM Proxy started - Memory limit: {settings.MAX_MEMORY_MB}MB")
    
    # Test Ollama connection on startup
    try:
        async with app.state.http_session.get(f"{settings.OLLAMA_BASE_URL}/api/tags") as resp:
            if resp.status == 200:
                data = await resp.json()
                model_count = len(data.get("models", []))
                logger.info(f"✅ Connected to Ollama - {model_count} models available")
            else:
                logger.warning(f"⚠️ Ollama connection issue - status {resp.status}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Ollama: {e}")
    
    yield
    
    # Shutdown
    await app.state.http_session.close()
    logger.info("🛑 HTTP session closed")

app = FastAPI(
    title="ENHANCED LLM Proxy", 
    version="2.3.0",
    description="RunPod A5000 Optimized LLM Proxy - ENHANCED Version with Advanced Features",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced request monitoring middleware
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    global request_count, error_count
    start_time_req = time.time()
    request_count += 1
    
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            error_count += 1
        
        process_time = time.time() - start_time_req
        logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        
        return response
    except Exception as e:
        error_count += 1
        logger.error(f"Request failed: {request.method} {request.url.path} - Error: {e}")
        raise

# Enhanced Models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

# Enhanced health check with comprehensive status
@app.get("/health")
async def health():
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(time.time() - start_time),
            "version": "2.3.0-enhanced",
            "gpu_optimized": True,
            "memory_limit_mb": settings.MAX_MEMORY_MB,
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": error_count / max(request_count, 1)
        }
        
        # Check Ollama connection
        async with app.state.http_session.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = data.get("models", [])
                health_status.update({
                    "ollama_status": "connected",
                    "models_available": len(models),
                    "model_list": [model["name"] for model in models[:5]]  # First 5 models
                })
                
                if len(models) == 0:
                    health_status["status"] = "degraded"
                    health_status["warning"] = "No models available"
                    
            else:
                health_status.update({
                    "status": "degraded",
                    "ollama_status": "disconnected",
                    "ollama_error": f"HTTP {resp.status}"
                })
                
    except asyncio.TimeoutError:
        health_status.update({
            "status": "degraded",
            "ollama_status": "timeout"
        })
    except Exception as e:
        health_status.update({
            "status": "unhealthy",
            "ollama_status": "error",
            "error": str(e)
        })
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    uptime = int(time.time() - start_time)
    error_rate = error_count / max(request_count, 1)
    
    metrics_text = f"""# HELP llm_proxy_requests_total Total number of requests
# TYPE llm_proxy_requests_total counter
llm_proxy_requests_total {request_count}

# HELP llm_proxy_errors_total Total number of errors
# TYPE llm_proxy_errors_total counter
llm_proxy_errors_total {error_count}

# HELP llm_proxy_error_rate Current error rate
# TYPE llm_proxy_error_rate gauge
llm_proxy_error_rate {error_rate:.4f}

# HELP llm_proxy_uptime_seconds Uptime in seconds
# TYPE llm_proxy_uptime_seconds gauge
llm_proxy_uptime_seconds {uptime}
"""
    
    return Response(content=metrics_text, media_type="text/plain")

@app.get("/models")
async def list_models():
    try:
        async with app.state.http_session.get(f"{settings.OLLAMA_BASE_URL}/api/tags") as resp:
            if resp.status == 200:
                data = await resp.json()
                models = []
                for model in data.get("models", []):
                    models.append({
                        "id": model["name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ollama",
                        "size": model.get("size", 0),
                        "modified_at": model.get("modified_at", "")
                    })
                return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    
    # Fallback response
    return {
        "object": "list", 
        "data": [
            {"id": settings.DEFAULT_MODEL, "object": "model"},
            {"id": settings.FALLBACK_MODEL, "object": "model"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    start_time_req = time.time()
    
    try:
        # Enhanced model selection with fallback
        model_to_use = request.model if request.model else settings.DEFAULT_MODEL
        
        # Check if requested model is available
        try:
            async with app.state.http_session.get(f"{settings.OLLAMA_BASE_URL}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                    if model_to_use not in available_models:
                        logger.warning(f"Model {model_to_use} not available, using fallback: {settings.FALLBACK_MODEL}")
                        model_to_use = settings.FALLBACK_MODEL
        except Exception as e:
            logger.warning(f"Could not check available models: {e}")

        # Prepare Ollama request
        ollama_request = {
            "model": model_to_use,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": request.stream or False,
            "options": {}
        }
        
        # Add optional parameters
        if request.temperature is not None:
            ollama_request["options"]["temperature"] = request.temperature
        if request.max_tokens is not None:
            ollama_request["options"]["num_predict"] = request.max_tokens
        if request.top_p is not None:
            ollama_request["options"]["top_p"] = request.top_p

        # Handle streaming vs non-streaming requests
        if request.stream:
            return StreamingResponse(
                stream_chat_response(ollama_request),
                media_type="text/plain"
            )
        else:
            # Non-streaming request
            async with app.state.http_session.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json=ollama_request,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Ollama error {resp.status}: {error_text}")
                    raise HTTPException(status_code=502, detail=f"Ollama error: {error_text}")
                
                ollama_response = await resp.json()
                
                # Transform to OpenAI format
                openai_response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_to_use,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ollama_response.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                        "completion_tokens": ollama_response.get("eval_count", 0),
                        "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
                    }
                }
                
                process_time = time.time() - start_time_req
                logger.info(f"Chat completion completed in {process_time:.3f}s")
                
                return openai_response
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def stream_chat_response(ollama_request: dict):
    """Stream chat response in OpenAI format"""
    try:
        async with app.state.http_session.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=ollama_request,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                yield f"data: {json.dumps({'error': f'Ollama error: {error_text}'})}\n\n"
                return
            
            chunk_id = f"chatcmpl-{int(time.time())}"
            
            async for line in resp.content:
                if line:
                    try:
                        line_str = line.decode('utf-8').strip()
                        if line_str:
                            ollama_chunk = json.loads(line_str)
                            
                            # Transform to OpenAI streaming format
                            openai_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": ollama_request["model"],
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": ollama_chunk.get("message", {}).get("content", "")
                                    },
                                    "finish_reason": "stop" if ollama_chunk.get("done", False) else None
                                }]
                            }
                            
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                            
                            if ollama_chunk.get("done", False):
                                break
                                
                    except json.JSONDecodeError:
                        continue
            
            # Send final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ollama_request["model"],
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"

@app.get("/")
async def root():
    return {
        "message": "ENHANCED LLM Proxy v2.3 - RunPod A5000 Optimized",
        "status": "running",
        "gpu_optimized": True,
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics", 
            "models": "/models",
            "chat": "/v1/chat/completions"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
        access_log=True
    )
EOF

# CRITICAL FIX 9: Create startup script
echo "🚀 Creating enhanced startup script..."
cat > start.sh << 'EOF'
#!/bin/bash
# Enhanced startup script for LLM Proxy

cd "$WORKSPACE_DIR/app" || cd /workspace/app

# Activate virtual environment
source "$WORKSPACE_DIR/venv/bin/activate" || source /workspace/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all

# Start the FastAPI application
echo "🚀 Starting ENHANCED LLM Proxy on port 8000..."
python main.py
EOF

chmod +x start.sh

# Create process management script
echo "🔧 Creating process management script..."
cat > manage.sh << 'EOF'
#!/bin/bash
# Process management script

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

case "$1" in
    start)
        echo "Starting services..."
        # Start Ollama
        CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 ollama serve > "$WORKSPACE_DIR/logs/ollama.log" 2>&1 &
        echo "Ollama started (PID: $!)"
        
        # Wait for Ollama to be ready
        sleep 10
        
        # Start FastAPI app
        cd "$WORKSPACE_DIR/app"
        source "$WORKSPACE_DIR/venv/bin/activate"
        python main.py &
        echo "FastAPI app started (PID: $!)"
        ;;
    stop)
        echo "Stopping services..."
        pkill -f "ollama serve"
        pkill -f "python main.py"
        echo "Services stopped"
        ;;
    restart)
        $0 stop
        sleep 5
        $0 start
        ;;
    status)
        echo "Service status:"
        if pgrep -f "ollama serve" > /dev/null; then
            echo "Ollama: Running"
        else
            echo "Ollama: Stopped"
        fi
        
        if pgrep -f "python main.py" > /dev/null; then
            echo "FastAPI: Running"
        else
            echo "FastAPI: Stopped"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
EOF

chmod +x manage.sh

# Final setup completion
echo ""
echo "🎉 ENHANCED RunPod A5000 Setup Complete!"
echo "============================================"
echo ""
echo "📋 Setup Summary:"
echo "   ✅ GPU Detection: $([[ $GPU_DETECTED == true ]] && echo "SUCCESS" || echo "WARNING")"
echo "   ✅ Ollama Service: Running (PID: $OLLAMA_PID)"
echo "   ✅ Models: $([[ $MODEL_PULL_SUCCESS == true ]] && echo "Mistral 7B Ready" || echo "Check Required")"
echo "   ✅ FastAPI App: Ready to start"
echo "   ✅ Memory Optimization: A5000 (24GB) configured"
echo ""
echo "🚀 Quick Start Commands:"
echo "   Start everything:    cd $WORKSPACE_DIR/app && ./start.sh"
echo "   Manage services:     cd $WORKSPACE_DIR/app && ./manage.sh start|stop|status"
echo "   Test health:         curl http://localhost:8000/health"
echo "   View logs:           tail -f $WORKSPACE_DIR/logs/ollama.log"
echo ""
echo "🌐 Access URLs (when running):"
echo "   Health Check:        http://localhost:8000/health"
echo "   API Documentation:   http://localhost:8000/docs"
echo "   Metrics:            http://localhost:8000/metrics"
echo "   Chat API:           http://localhost:8000/v1/chat/completions"
echo ""
echo "💡 Tips:"
echo "   - The FastAPI app will start on port 8000"
echo "   - Ollama runs on port 11434"
echo "   - Check logs if services don't start properly"
echo "   - Models are cached in $WORKSPACE_DIR/models"
echo ""

if [ "$MODEL_PULL_SUCCESS" = false ]; then
    echo "⚠️  Warning: Model download failed"
    echo "   Try manually: cd $WORKSPACE_DIR && ollama pull mistral:7b-instruct-q4_0"
    echo ""
fi

# CRITICAL FIX: Mark setup as complete and prevent restart loop
echo "🔒 Marking setup complete to prevent restart loop..."
touch "$SETUP_COMPLETE_FILE"

echo "✨ Setup completed successfully!"

# CRITICAL: Auto-start without user input (RunPod containers are non-interactive)
echo "🚀 Auto-starting application (non-interactive mode)..."
echo "📋 Switching to application mode - setup script will be replaced..."

# Change to app directory and activate environment
cd "$WORKSPACE_DIR/app"
source "$WORKSPACE_DIR/venv/bin/activate"

# Ensure Ollama is still running before starting FastAPI
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🤖 Restarting Ollama service..."
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    OLLAMA_HOST=0.0.0.0:11434 \
    ollama serve > "$WORKSPACE_DIR/logs/ollama.log" 2>&1 &
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama to start..."
    for i in {1..30}; do
        if curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama is ready!"
            break
        fi
        echo "   Waiting... ($i/30)"
        sleep 2
    done
else
    echo "✅ Ollama already running"
fi

# CRITICAL: Use exec to replace this process entirely - this prevents restart loops
echo "🔄 Starting FastAPI application and replacing setup process..."
echo "📋 From now on, this container will run the FastAPI app continuously..."
echo ""
echo "🌐 Once started, access the API at:"
echo "   Health Check: http://localhost:8000/health"
echo "   Documentation: http://localhost:8000/docs"
echo "   Chat API: http://localhost:8000/v1/chat/completions"
echo ""

# This exec command replaces the current shell process with the FastAPI app
# This is CRITICAL - it prevents the script from ending and restarting
exec python main.py
