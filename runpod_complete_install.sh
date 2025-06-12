#!/bin/bash
# Enhanced Installation Script for RunPod LLM Proxy
# RunPod/CI-safe, no venv, auto-enhanced features, logging, error handling

set -euo pipefail
IFS=$'\n\t'

# Color logging
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_status()  { echo -e "${GREEN}✅ $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo -e "${BLUE}🚀 Starting LLM Proxy Installation (Enhanced for RunPod)${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: install system packages if root
if [ "$(id -u)" -eq 0 ]; then
  print_info "Installing system packages (as root)..."
  apt-get update -qq && apt-get install -y curl git python3 python3-pip jq build-essential libopenblas-dev
else
  print_warning "Skipping apt-get system dependencies (not root)"
fi

# Ensure Python is available
command -v python3 >/dev/null || { print_error "Python3 is not installed."; exit 1; }
command -v pip3 >/dev/null || { print_error "pip3 is not installed."; exit 1; }

# Project structure
mkdir -p app/{services,models,middleware,utils,tests,data/{cache,logs,models},security}
touch app/data/{cache,logs,models}/.gitkeep
print_status "Directory structure created"

# Create requirements.txt
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
EOF
print_status "requirements.txt created"

# Install Python packages globally
pip3 install --upgrade pip || { print_error "pip upgrade failed"; exit 1; }
pip3 install -r app/requirements.txt || { print_error "Dependency install failed"; exit 1; }
print_status "Python dependencies installed globally"

# Create .env if not present
ENV_PATH="app/.env"
if [ ! -f "$ENV_PATH" ]; then
  cat > "$ENV_PATH" << 'EOF'
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-key
MAX_MEMORY_MB=8192
CACHE_MEMORY_LIMIT_MB=1024
MODEL_MEMORY_LIMIT_MB=4096
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
EOF
  print_status ".env created with default values"
else
  print_info ".env already exists – using existing config"
fi

# Pull Ollama model
if command -v ollama >/dev/null && ! ollama list | grep -q "mistral:7b-instruct-q4_0"; then
  print_info "Pulling mistral:7b-instruct-q4_0..."
  ollama pull mistral:7b-instruct-q4_0 || {
    print_warning "Mistral model pull failed – attempting fallback"
    ollama pull llama3.2:1b || print_error "No model pulled. Ollama setup incomplete."
  }
else
  print_status "Model already available or Ollama not installed"
fi

# Create a simple test script
cat > app/test_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
def test_core():
    try:
        import fastapi, uvicorn, pydantic, aiohttp
        print("✅ Core dependencies OK")
        return True
    except Exception as e:
        print(f"❌ Core import failed: {e}")
        return False
def test_optional():
    for pkg in ['redis', 'sentence_transformers', 'sse_starlette']:
        try:
            __import__(pkg)
            print(f"✅ {pkg} installed")
        except:
            print(f"⏸️ {pkg} missing")
if __name__ == "__main__":
    print("🧪 Testing installation...")
    if test_core():
        test_optional()
        sys.exit(0)
    else:
        sys.exit(1)
EOF
chmod +x app/test_installation.py
python3 app/test_installation.py || { print_error "Installation test failed"; exit 1; }

# Create start.sh using .env PORT
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/app"
PORT=$(grep PORT .env | cut -d '=' -f2 | tr -d '[:space:]')
PORT=${PORT:-8000}
echo "🚀 Starting server on port $PORT"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT"
EOF
chmod +x start.sh
print_status "start.sh created"

# Done
print_status "LLM Proxy setup complete. Run ./start.sh to launch the app."
