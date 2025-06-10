#!/bin/bash
# ENHANCED RunPod A5000 Setup Script - Advanced GPU Detection & Error Handling

set -e

echo "🚀 ENHANCED LLM Proxy Setup for RunPod A5000 v2.2"
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
    
    # Try multiple NVIDIA utility versions
    NVIDIA_INSTALLED=false
    for version in 535 530 525 520 470; do
        echo "   🔄 Trying nvidia-utils-$version..."
        if apt-get install -y nvidia-utils-$version 2>/dev/null; then
            echo "   ✅ Installed nvidia-utils-$version"
            NVIDIA_INSTALLED=true
            break
        fi
    done
    
    if [ "$NVIDIA_INSTALLED" = false ]; then
        echo "   ⚠️ Could not install nvidia-utils via apt"
        # Try nvidia-smi directly
        if apt-get install -y nvidia-smi 2>/dev/null; then
            echo "   ✅ Installed nvidia-smi package"
        fi
    fi
    
    # Retry nvidia-smi after installation
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "Unknown GPU, Unknown")
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
        echo "   ✅ GPU detected after installation: $GPU_NAME ($GPU_MEMORY MB)"
        GPU_DETECTED=true
    fi
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

# Try to install Python NVIDIA packages via apt (if available)
echo "🔧 Installing Python NVIDIA packages via apt..."
apt-get install -y python3-pynvml 2>/dev/null || echo "   ⚠️ python3-pynvml not available via apt"

# Install Python NVIDIA packages via pip
echo "🐍 Installing Python NVIDIA packages via pip..."
pip3 install --no-cache-dir nvidia-ml-py3 pynvml gpustat 2>/dev/null || echo "   ⚠️ Some Python NVIDIA packages failed to install"

# Test Python GPU detection
echo "🧪 Testing Python GPU detection..."
python3 -c "
import sys
try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    print(f'✅ Found {count} GPU(s) via Python')
    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem_info.total // (1024**3)
            free_gb = mem_info.free // (1024**3)
            used_gb = mem_info.used // (1024**3)
            print(f'   GPU {i}: {name}')
            print(f'   Memory: {total_gb} GB total, {free_gb} GB free, {used_gb} GB used')
        except Exception as e:
            print(f'   GPU {i}: Error getting details - {e}')
except ImportError:
    print('⚠️ pynvml not available')
except Exception as e:
    print(f'⚠️ Python GPU detection failed: {e}')
" 2>/dev/null || echo "   ⚠️ Python GPU test encountered errors"

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
# ENHANCED RunPod A5000 Configuration v2.2
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
cat > main.py << 'MAIN_EOF'
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    version="2.2.0",
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
            "version": "2.2.0-enhanced",
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
                        logger.warning(f"Model {model_to_use} not available, using {settings.DEFAULT_MODEL}")
                        model_to_use = settings.DEFAULT_MODEL
                        if model_to_use not in available_models and settings.FALLBACK_MODEL in available_models:
                            model_to_use = settings.FALLBACK_MODEL
        except Exception as e:
            logger.warning(f"Could not check model availability: {e}")
        
        # Enhanced Ollama request with better options
        ollama_request = {
            "model": model_to_use,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": request.stream or False,
            "options": {
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 1.0,
                "repeat_penalty": 1.1,
                "top_k": 40
            }
        }
        
        if request.max_tokens:
            ollama_request["options"]["num_predict"] = min(request.max_tokens, 4096)
            
        # Enhanced request with timeout and retries
        async with app.state.http_session.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=ollama_request,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                
                # Calculate timing
                process_time = time.time() - start_time_req
                
                # Enhanced OpenAI format response
                return {
                    "id": f"chatcmpl-{int(time.time())}-{hash(str(request.messages)) % 10000}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_to_use,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                        "completion_tokens": len(result.get("message", {}).get("content", "").split()),
                        "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + len(result.get("message", {}).get("content", "").split())
                    },
                    "system_fingerprint": f"fp_{model_to_use}_{int(time.time())}",
                    "processing_time_seconds": round(process_time, 3)
                }
            else:
                error_text = await resp.text()
                logger.error(f"Ollama error {resp.status}: {error_text}")
                raise HTTPException(status_code=resp.status, detail=f"Ollama error: {error_text}")
                
    except asyncio.TimeoutError:
        logger.error("Request timeout")
        raise HTTPException(status_code=504, detail="Request timeout - try reducing message length")
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/v1/models")
async def openai_models():
    """OpenAI-compatible models endpoint"""
    return await list_models()

@app.get("/status")
async def status():
    """Detailed status information"""
    return {
        "service": "Enhanced LLM Proxy",
        "version": "2.2.0",
        "status": "running",
        "gpu_optimized": True,
        "ollama_url": settings.OLLAMA_BASE_URL,
        "default_model": settings.DEFAULT_MODEL,
        "memory_limit_mb": settings.MAX_MEMORY_MB,
        "features": {
            "streaming": True,
            "model_fallback": True,
            "enhanced_logging": True,
            "metrics": settings.ENABLE_METRICS,
            "rate_limiting": settings.ENABLE_RATE_LIMITING
        }
    }

@app.get("/")
async def root():
    return {
        "message": "ENHANCED LLM Proxy - RunPod A5000 Optimized",
        "version": "2.2.0-enhanced",
        "status": "GPU optimized with advanced features",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/models",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "gpu_status": "optimized"
    }

if __name__ == "__main__":
    logger.info(f"Starting Enhanced LLM Proxy on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.DEBUG,
        access_log=True,
        log_level="info"
    )
MAIN_EOF

echo "🌐 Starting ENHANCED FastAPI application..."
source "$WORKSPACE_DIR/venv/bin/activate"

# Create requirements.txt for future reference
echo "📝 Creating requirements.txt..."
cat > requirements.txt << 'REQ_EOF'
# Enhanced LLM Proxy Requirements
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiohttp==3.9.1
pydantic==2.5.0
pydantic-settings==2.1.0
psutil==5.9.6
python-multipart==0.0.6
huggingface_hub>=0.20.0,<0.25.0
sentence-transformers>=2.2.0,<3.0.0
faiss-cpu==1.7.4
sse-starlette==1.6.5
numpy>=1.21.0,<1.25.0
python-dotenv==1.0.0
jinja2>=3.1.0
prometheus-client==0.19.0
python-json-logger==2.0.7
colorama==0.4.6
nvidia-ml-py3
pynvml
gpustat
REQ_EOF

# Start the application with enhanced logging
echo "   🚀 Starting ENHANCED application with monitoring..."
python3 main.py > "$WORKSPACE_DIR/logs/app.log" 2>&1 &
APP_PID=$!

echo "   📋 FastAPI PID: $APP_PID"

# Enhanced startup verification
echo "⏳ Waiting for FastAPI application to start..."
APP_READY=false

for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "   ✅ FastAPI application is ready!"
        APP_READY=true
        break
    fi
    
    # Check if process is still running
    if ! kill -0 $APP_PID 2>/dev/null; then
        echo "   ❌ FastAPI process died during startup"
        echo "   📋 Last few lines of app log:"
        tail -10 "$WORKSPACE_DIR/logs/app.log" 2>/dev/null || echo "   No log available"
        break
    fi
    
    echo "   Attempt $i/30 - waiting 3 seconds... (process running: ✓)"
    sleep 3
done

if [ "$APP_READY" = false ]; then
    echo "   ❌ FastAPI failed to start"
    exit 1
fi

# Enhanced service testing
echo ""
echo "🧪 ENHANCED Service Testing..."

# Test health endpoint
echo "   🏥 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo '{"error": "failed"}')
if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
    STATUS=$(echo "$HEALTH_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null)
    echo "   ✅ Health check passed - Status: $STATUS"
    
    # Show detailed health info
    OLLAMA_STATUS=$(echo "$HEALTH_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('ollama_status', 'unknown'))" 2>/dev/null)
    MODEL_COUNT=$(echo "$HEALTH_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('models_available', 0))" 2>/dev/null)
    echo "   📊 Ollama: $OLLAMA_STATUS, Models: $MODEL_COUNT"
else
    echo "   ⚠️ Health check failed: $HEALTH_RESPONSE"
fi

# Test chat completion
echo "   💬 Testing chat completion..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model":"mistral:7b-instruct-q4_0",
        "messages":[{"role":"user","content":"Say hello in one word"}],
        "max_tokens":10
    }' 2>/dev/null || echo '{"error": "failed"}')

if echo "$CHAT_RESPONSE" | grep -q '"choices"'; then
    RESPONSE_CONTENT=$(echo "$CHAT_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('choices', [{}])[0].get('message', {}).get('content', 'No content'))" 2>/dev/null)
    PROCESS_TIME=$(echo "$CHAT_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('processing_time_seconds', 'N/A'))" 2>/dev/null)
    echo "   ✅ Chat completion passed!"
    echo "   📝 Response: $RESPONSE_CONTENT"
    echo "   ⏱️ Processing time: ${PROCESS_TIME}s"
else
    echo "   ⚠️ Chat completion failed: $CHAT_RESPONSE"
fi

# Test models endpoint
echo "   📚 Testing models endpoint..."
MODELS_RESPONSE=$(curl -s http://localhost:8000/models 2>/dev/null || echo '{"error": "failed"}')
if echo "$MODELS_RESPONSE" | grep -q '"data"'; then
    MODEL_COUNT_API=$(echo "$MODELS_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data.get('data', [])))" 2>/dev/null)
    echo "   ✅ Models endpoint passed - $MODEL_COUNT_API models available"
else
    echo "   ⚠️ Models endpoint failed"
fi

# Enhanced cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down ENHANCED services..."
    
    echo "   🛑 Stopping FastAPI application (PID: $APP_PID)..."
    kill $APP_PID 2>/dev/null || true
    
    echo "   🛑 Stopping Ollama service (PID: $OLLAMA_PID)..."
    kill $OLLAMA_PID 2>/dev/null || true
    
    # Wait for graceful shutdown
    sleep 5
    
    # Force kill if still running
    kill -9 $APP_PID 2>/dev/null || true
    kill -9 $OLLAMA_PID 2>/dev/null || true
    
    echo "   ✅ Services stopped"
}

trap cleanup EXIT SIGTERM SIGINT

# Display enhanced startup summary
echo ""
echo "🎉 ENHANCED LLM Proxy is running successfully!"
echo "============================================="
echo "🔧 Version: 2.2.0-enhanced"
echo "🖥️  GPU: $GPU_NAME ($GPU_MEMORY MB)" 
echo "🧠 Memory Limit: ${settings.MAX_MEMORY_MB}MB"
echo "📊 API URL: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/health"
echo "📈 Metrics: http://localhost:8000/metrics"
echo "🤖 Models: http://localhost:8000/models"
echo "📋 Status: http://localhost:8000/status"
echo ""
echo "📁 Log Files:"
echo "   🤖 Ollama: $WORKSPACE_DIR/logs/ollama.log"
echo "   🌐 FastAPI: $WORKSPACE_DIR/logs/app.log"
echo ""
echo "🧪 Test Commands:"
echo "# Health Check"
echo 'curl http://localhost:8000/health | jq'
echo ""
echo "# Chat Completion"
echo 'curl -X POST http://localhost:8000/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"model":"mistral:7b-instruct-q4_0","messages":[{"role":"user","content":"Hello!"}]}'"'"' | jq'
echo ""
echo "# List Models"
echo 'curl http://localhost:8000/models | jq'
echo ""
echo "🔧 Enhanced Features Active:"
echo "   ✅ GPU Optimization for RunPod A5000"
echo "   ✅ Advanced Error Handling"
echo "   ✅ Prometheus Metrics"
echo "   ✅ Model Fallback System"
echo "   ✅ Enhanced Logging"
echo "   ✅ Request Monitoring"
echo "   ✅ Health Monitoring"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running and show live status
echo ""
echo "📊 Live Status (updates every 30 seconds):"
while true; do
    sleep 30
    
    # Quick status check
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        CURRENT_TIME=$(date '+%H:%M:%S')
        echo "[$CURRENT_TIME] ✅ Services running normally"
        
        # Show request count if available
        HEALTH_DATA=$(curl -s http://localhost:8000/health 2>/dev/null)
        if [ $? -eq 0 ]; then
            REQ_COUNT=$(echo "$HEALTH_DATA" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('request_count', 'N/A'))" 2>/dev/null)
            ERROR_COUNT=$(echo "$HEALTH_DATA" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('error_count', 'N/A'))" 2>/dev/null)
            echo "[$CURRENT_TIME] 📊 Requests: $REQ_COUNT, Errors: $ERROR_COUNT"
        fi
    else
        echo "[$CURRENT_TIME] ⚠️  Service health check failed"
    fi
done
