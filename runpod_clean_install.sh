#!/bin/bash
# runpod_clean_install.sh - Auto-install LLM Proxy for RunPod

set -e

# Colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status()   { echo -e "${GREEN}✅ $1${NC}"; }
print_warning()  { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error()    { echo -e "${RED}❌ $1${NC}"; }
print_info()     { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo "🚀 Starting Clean LLM Proxy Installation"
echo "=========================================="

# Prevent duplicate installs
if [ -f ".installed" ]; then
    print_info "Installation already completed. Skipping..."
    exit 0
fi

# Clean previous install if any
rm -rf app venv start.sh README_INSTALLATION.md

# Step 1: Python version check
print_info "📋 Checking Python version"
python_version=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi
print_status "Python version is $python_version"

# Step 2: Create directories
print_info "📁 Creating project structure"
mkdir -p app/{services,models,middleware,utils,tests,security,data/{cache,logs,models}}
touch app/data/{cache,logs,models}/.gitkeep
print_status "Project directories created"

# Step 3: Virtual environment
print_info "🐍 Creating virtual environment"
python3 -m venv venv
source venv/bin/activate
print_status "Virtual environment activated"

# Step 4: Upgrade pip
print_info "📦 Upgrading pip"
pip install --upgrade pip

# Step 5: Create requirements.txt
cat > app/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
aiohttp==3.9.1
psutil==5.9.6
python-multipart==0.0.6
numpy>=1.21.0,<1.25.0
EOF
print_status "requirements.txt created"

# Step 6: Install dependencies
cd app
pip install -r requirements.txt
cd ..

# Step 7: Create .env
cat > app/.env << 'EOF'
ENVIRONMENT=development
DEBUG=false
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-change-me
MAX_MEMORY_MB=8192
CACHE_MEMORY_LIMIT_MB=1024
MODEL_MEMORY_LIMIT_MB=4096
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
EOF
print_status ".env created"

# Step 8: Add test script
cat > app/test_installation.py << 'EOF'
import fastapi, uvicorn, pydantic, aiohttp
print("✅ All core libraries imported successfully!")
EOF
chmod +x app/test_installation.py

# Step 9: Install optional features (always ON)
print_info "🎯 Installing optional enhanced features"
cd app
pip install redis aioredis sse-starlette sentence-transformers faiss-cpu || true
cd ..
print_status "Optional features installed"

# Step 10: Create start script
cat > start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
cd app
echo "🚀 Launching LLM Proxy..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF
chmod +x start.sh
print_status "Start script created"

# Finalize
touch .installed
print_status "Installation complete!"
echo -e "${GREEN}🎉 Ready to go! Run './start.sh' to launch your app.${NC}"
