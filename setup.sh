#!/bin/bash
# Enhanced setup.sh - Gets Dockerfile.runpod features without Docker
set -e

echo "🚀 Enhanced LLM Proxy Setup for RunPod A5000"
echo "============================================="

# Detect if we're on RunPod
if [[ "$PWD" == "/workspace"* ]]; then
    echo "✅ RunPod environment detected"
    WORKSPACE_DIR="/workspace"
else
    echo "ℹ️  Local environment detected"
    WORKSPACE_DIR="$(pwd)"
fi

cd "$WORKSPACE_DIR"
mkdir -p logs models cache

# System setup with RunPod optimizations
echo "🔧 Setting up system packages..."
export DEBIAN_FRONTEND=noninteractive
dpkg --configure -a || true
apt-get clean
apt-get update -qq

# Install packages including GPU support
apt-get install -y --no-install-recommends \
    curl python3-pip python3-venv python3-dev \
    git wget build-essential \
    nvidia-cuda-toolkit || echo "CUDA toolkit installation failed (may already be installed)"

# Install Ollama with GPU support
echo "🤖 Installing Ollama with GPU support..."
curl -fsSL https://ollama.com/install.sh | sh

# Set up optimized environment for A5000
export OLLAMA_MODELS="$WORKSPACE_DIR/models"
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=3
export NVIDIA_VISIBLE_DEVICES=all
export GPU_MEMORY_FRACTION=0.9

# Enhanced memory settings for A5000 (24GB VRAM)
export MAX_MEMORY_MB=16384
export CACHE_MEMORY_LIMIT_MB=2048
export MODEL_MEMORY_LIMIT_MB=8192
export SEMANTIC_MODEL_MAX_MEMORY_MB=500

mkdir -p "$OLLAMA_MODELS"

echo "🐍 Setting up enhanced Python environment..."
python3 -m venv "$WORKSPACE_DIR/venv"
source "$WORKSPACE_DIR/venv/bin/activate"

# Install enhanced dependencies
echo "📚 Installing enhanced Python packages..."
pip install --no-cache-dir --upgrade pip

# Core dependencies
pip install --no-cache-dir \
    fastapi[all]==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    pydantic-settings==2.1.0 \
    psutil==5.9.6

# Enhanced features (with error handling)
echo "🚀 Installing enhanced features..."
pip install --no-cache-dir sentence-transformers==2.2.2 || echo "⚠️  Semantic features will be disabled"
pip install --no-cache-dir faiss-cpu==1.7.4 || echo "⚠️  FAISS features will be disabled"
pip install --no-cache-dir sse-starlette==1.6.5 || echo "⚠️  Streaming features will be disabled"

# Get application code
echo "📁 Setting up application..."
if [ ! -d "$WORKSPACE_DIR/app" ]; then
    git clone https://github.com/puneetrinity/llm1.git "$WORKSPACE_DIR/app" || {
        echo "⚠️  Git clone failed, creating app directory..."
        mkdir -p "$WORKSPACE_DIR/app"
    }
fi

cd "$WORKSPACE_DIR/app"

# Create enhanced configuration
echo "⚙️ Creating RunPod A5000 optimized configuration..."
cat > .env << 'EOF'
# RunPod A5000 Optimized Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_MAX_RETRIES=3

# Memory management for A5000 (24GB VRAM)
MAX_MEMORY_MB=16384
CACHE_MEMORY_LIMIT_MB=2048
MODEL_MEMORY_LIMIT_MB=8192
SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Enhanced features (auto-detected)
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true

# GPU optimization
GPU_MEMORY_FRACTION=0.9
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=3

# Model settings
DEFAULT_MODEL=mistral:7b-instruct-q4_0
MAX_TOKENS=2048
DEFAULT_TEMPERATURE=0.7

# Security
ENABLE_AUTH=false
CORS_ORIGINS=["*"]
CORS_ALLOW_CREDENTIALS=true

# Performance
ENABLE_RATE_LIMITING=false
ENABLE_CACHE=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
EOF

# Start Ollama with optimizations
echo "📡 Starting Ollama service with GPU optimizations..."
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS="$WORKSPACE_DIR/models"
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "   Attempt $i/30 - waiting 3 seconds..."
    sleep 3
done

if ! curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull models optimized for A5000
echo "📦 Downloading optimized models for A5000..."
echo "   🔄 Pulling Mistral 7B (Priority 1)..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

echo "   🔄 Pulling DeepSeek V2 7B (Priority 2)..."
ollama pull deepseek-v2:7b-q4_0 &
DEEPSEEK_PID=$!

echo "   🔄 Pulling LLaMA3 8B (Priority 2)..."
ollama pull llama3:8b-instruct-q4_0 &
LLAMA_PID=$!

# Wait for priority model
echo "   ⏳ Waiting for priority model (Mistral)..."
wait $MISTRAL_PID
echo "   ✅ Mistral 7B ready!"

# Warm up the priority model
echo "   🔥 Warming up Mistral..."
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b-instruct-q4_0",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false,
    "options": {"num_predict": 5}
  }' >/dev/null 2>&1 && echo "   ✅ Mistral warmed up!" || echo "   ⚠️  Warmup failed"

# Download semantic model if features enabled
if pip list | grep -q "sentence-transformers"; then
    echo "🧠 Pre-downloading semantic model..."
    python3 -c "
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Semantic model downloaded')
except Exception as e:
    print(f'⚠️  Semantic model download failed: {e}')
    " &
fi

echo "🌐 Starting enhanced FastAPI application..."
source "$WORKSPACE_DIR/venv/bin/activate"

# Check for main files and start the best one
if [ -f main_enhanced.py ]; then
    echo "   🚀 Starting enhanced version..."
    python3 main_enhanced.py
elif [ -f main.py ]; then
    echo "   🚀 Starting main application..."
    python3 main.py
else
    echo "   📝 Creating enhanced main.py..."
    cat > main.py << 'MAIN_EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import asyncio
import aiohttp
import json
from datetime import datetime

# Configuration
class Settings:
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    CORS_ORIGINS = ["*"]
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'mistral:7b-instruct-q4_0')

settings = Settings()

app = FastAPI(
    title="Enhanced LLM Proxy", 
    version="2.0.0",
    description="RunPod A5000 Optimized LLM Proxy"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# Global HTTP session
http_session = None

@app.on_event("startup")
async def startup():
    global http_session
    http_session = aiohttp.ClientSession()

@app.on_event("shutdown") 
async def shutdown():
    if http_session:
        await http_session.close()

@app.get("/health")
async def health():
    try:
        async with http_session.get(f"{settings.OLLAMA_BASE_URL}/api/tags") as resp:
            ollama_healthy = resp.status == 200
        
        return {
            "status": "healthy" if ollama_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "ollama_status": "connected" if ollama_healthy else "disconnected",
            "gpu_optimized": True,
            "enhanced_features": True
        }
    except:
        return {"status": "unhealthy", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models():
    try:
        async with http_session.get(f"{settings.OLLAMA_BASE_URL}/api/tags") as resp:
            if resp.status == 200:
                data = await resp.json()
                models = [{"id": model["name"], "object": "model"} for model in data.get("models", [])]
                return {"object": "list", "data": models}
    except:
        pass
    
    return {"object": "list", "data": [{"id": settings.DEFAULT_MODEL, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Convert to Ollama format
        ollama_request = {
            "model": request.model if request.model else settings.DEFAULT_MODEL,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": False,
            "options": {"temperature": request.temperature}
        }
        
        if request.max_tokens:
            ollama_request["options"]["num_predict"] = request.max_tokens
            
        async with http_session.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=ollama_request
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                
                # Convert to OpenAI format
                return {
                    "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": ollama_request["model"],
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(" ".join(msg.content for msg in request.messages).split()),
                        "completion_tokens": len(result.get("message", {}).get("content", "").split()),
                        "total_tokens": len(" ".join(msg.content for msg in request.messages).split()) + len(result.get("message", {}).get("content", "").split())
                    }
                }
            else:
                raise HTTPException(status_code=resp.status, detail="Ollama request failed")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Enhanced LLM Proxy - RunPod A5000 Optimized",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
MAIN_EOF

    echo "   🚀 Starting enhanced application..."
    python3 main.py
fi

# Cleanup function
cleanup() {
    echo "🛑 Shutting down services..."
    kill $OLLAMA_PID 2>/dev/null || true
    # Wait for other model downloads to finish
    wait $DEEPSEEK_PID 2>/dev/null && echo "   ✅ DeepSeek V2 7B downloaded!" || true
    wait $LLAMA_PID 2>/dev/null && echo "   ✅ LLaMA3 8B downloaded!" || true
}

trap cleanup EXIT SIGTERM SIGINT

echo ""
echo "🎉 Enhanced LLM Proxy is running!"
echo "================================="
echo "📊 API URL: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/health"
echo "🤖 Models: http://localhost:8000/models"
echo ""
echo "🧪 Test with:"
echo 'curl -X POST http://localhost:8000/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello!"}]}'"'"
echo ""
