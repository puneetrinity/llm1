#!/bin/bash
# runpod_complete_install.sh - Automated Setup Script for LLM Proxy (RunPod-ready, no prompts)

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status()  { echo -e "${GREEN}✅ $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo "🚀 Starting LLM Proxy Installation (RunPod Safe)"
echo "==============================================="

# 1. Install Python if not present
print_info "Checking Python and pip..."
apt-get update -qq
apt-get install -y python3 python3-pip python3-venv git curl jq

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    exit 1
fi

# 2. Setup directory structure
mkdir -p app/{services,models,middleware,utils,tests,data/{cache,logs,models},security}
touch app/data/{cache,logs,models}/.gitkeep
print_status "Project folder structure ready"

# 3. Create virtual environment
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
else
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# 4. Activate venv
source venv/bin/activate
print_status "Virtual environment activated"

# 5. Upgrade pip and create requirements.txt
pip install --upgrade pip
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

# 6. Install all dependencies
cd app
pip install -r requirements.txt
cd ..
print_status "Python dependencies installed"

# 7. .env setup
if [ ! -f "app/.env" ]; then
cat > app/.env << 'EOF'
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-change-me
MAX_MEMORY_MB=8192
CACHE_MEMORY_LIMIT_MB=1024
MODEL_MEMORY_LIMIT_MB=4096
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
EOF
    print_status ".env file created"
else
    print_warning ".env already exists"
fi

# 8. Optional install is now auto-enabled
install_enhanced="y"
if [[ $install_enhanced =~ ^[Yy]$ ]]; then
    print_info "Installing enhanced features..."
    cd app
    pip install redis aioredis sse-starlette
    pip install sentence-transformers faiss-cpu || {
        print_warning "Semantic packages failed to install — continuing anyway"
    }
    cd ..
    print_status "Enhanced features installed"
else
    print_info "Skipping enhanced features"
fi

# 9. Create test file
cat > app/test_installation.py << 'EOF'
#!/usr/bin/env python3
import sys

def test_imports():
    try:
        import fastapi, uvicorn, pydantic, aiohttp
        print("✅ Core packages OK")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_optional():
    print("📦 Testing optional packages:")
    try: import redis; print("✅ Redis OK")
    except: print("⏸️ Redis missing")
    try: import sentence_transformers; print("✅ Sentence Transformers OK")
    except: print("⏸️ ST missing")
    try: import sse_starlette; print("✅ SSE OK")
    except: print("⏸️ SSE missing")

if __name__ == "__main__":
    print("🧪 Running test...")
    if test_imports():
        test_optional()
        sys.exit(0)
    else:
        sys.exit(1)
EOF

python app/test_installation.py

# 10. Create start script
cat > start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
cd app
echo "🚀 Starting FastAPI..."
uvicorn main:app --host 0.0.0.0 --port 8000
EOF
chmod +x start.sh
print_status "start.sh created"

# 11. Final Summary
echo -e "\n${GREEN}🎉 Setup Complete! Use ./start.sh to run your API.${NC}"
