#!/bin/bash
# setup.sh - RunPod Optimized LLM App Installer

set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_status()  { echo -e "${GREEN}✅ $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo -e "${BLUE}🚀 Starting LLM App Setup for RunPod${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 🔹 Step 1: Python version check
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.8"
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
  print_error "Python $REQUIRED_VERSION+ is required. Found: $PYTHON_VERSION"
  exit 1
fi
print_status "Python version $PYTHON_VERSION OK"

# 🔹 Step 2: Install system dependencies (RunPod has sudo)
print_info "Installing system packages..."
sudo apt-get update -qq && sudo apt-get install -y \
  python3-pip \
  build-essential \
  cmake \
  libopenblas-dev \
  libssl-dev \
  jq \
  curl \
  git

# 🔹 Step 3: Create project structure
print_info "Creating project structure..."
mkdir -p app/{services,models,middleware,utils,tests,data/{cache,logs,models},security}
touch app/data/{cache,logs,models}/.gitkeep
print_status "Project structure ready"

# 🔹 Step 4: Create optimized requirements.txt
cat > app/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
aiohttp==3.9.1
psutil==5.9.6
python-multipart==0.0.6
numpy>=1.21.0,<1.25.0
redis>=4.5.0
aioredis>=2.0.0
sse-starlette==1.6.5
sentence-transformers>=2.2.0
faiss-cpu==1.7.4
torch>=2.0.0
transformers>=4.35.0
EOF
print_status "requirements.txt created"

# 🔹 Step 5: Install Python dependencies with optimized flags
print_info "Installing Python dependencies..."
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
pip3 install --upgrade pip setuptools wheel
pip3 install --no-cache-dir -r app/requirements.txt

# Additional optimizations for RunPod
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
print_status "Dependencies installed"

# 🔹 Step 6: Generate .env with RunPod-specific defaults
ENV_PATH="app/.env"
if [ ! -f "$ENV_PATH" ]; then
  cat > "$ENV_PATH" << 'EOF'
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
ENABLE_AUTH=true
DEFAULT_API_KEY=sk-runpod-key-$(openssl rand -hex 12)
MAX_MEMORY_MB=8192
CACHE_MEMORY_LIMIT_MB=1024
MODEL_MEMORY_LIMIT_MB=4096
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
EOF
  print_status ".env created with secure defaults"
else
  print_info ".env already exists - preserving existing configuration"
fi

# 🔹 Step 7: Ollama setup (if needed)
if ! command -v ollama >/dev/null; then
  print_info "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  print_status "Ollama installed"
  
  # Start Ollama service
  sudo systemctl start ollama
  sleep 5  # Wait for service to start
fi

# Pull models in background to avoid blocking setup
print_info "Starting model download in background..."
nohup ollama pull mistral:7b-instruct-q4_0 > model_download.log 2>&1 &

# 🔹 Step 8: Create main.py if not exists
if [ ! -f "app/main.py" ]; then
  cat > "app/main.py" << 'EOF'
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="LLM API", version="1.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "LLM API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info")
    )
EOF
  print_status "Created minimal main.py"
fi

# 🔹 Step 9: Create start script with RunPod optimizations
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/app"

# Activate NVIDIA GPU if available
if [ -x "$(command -v nvidia-smi)" ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  export CUDA_HOME=/usr/local/cuda
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Start the server
PORT=$(grep PORT .env | cut -d '=' -f2 | tr -d '[:space:]')
PORT=${PORT:-8000}
echo "🚀 Starting FastAPI on port $PORT with $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'CPU')"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1
EOF
chmod +x start.sh
print_status "start.sh created with GPU support"

# 🔹 Step 10: Create RunPod-specific health check
cat > health_check.sh << 'EOF'
#!/bin/bash
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$RESPONSE" -eq 200 ]; then
  echo "Health check passed"
  exit 0
else
  echo "Health check failed"
  exit 1
fi
EOF
chmod +x health_check.sh
print_status "Health check script created"

# 🔹 Final Summary
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════╗"
echo "║          RunPod Setup Complete!              ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ Next steps:                                  ║"
echo "║ 1. Check model download: tail -f model_download.log"
echo "║ 2. Start the app: ./start.sh                ║"
echo "║ 3. Monitor GPU: watch -n 1 nvidia-smi       ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"
