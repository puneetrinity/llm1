#!/bin/bash
# setup.sh – Clone full app from GitHub, install, and configure (RunPod safe, no venv)

set -e

# Color functions
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🚀 Starting RunPod App Setup..."

# 1. Install base packages
apt-get update && apt-get install -y curl git python3 python3-pip jq
print_status "System packages installed"

# 2. Clone your repo to /workspace/llm1
cd /workspace
if [ ! -d "llm1" ]; then
  git clone https://github.com/puneetrinity/llm1.git
  print_status "Repo cloned: llm1"
else
  print_warning "Repo already exists. Skipping clone."
fi

cd llm1

# 3. Install Python dependencies
print_info "Installing Python packages..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r app/requirements.txt || {
  print_error "Dependency install failed"
  exit 1
}
print_status "Python dependencies installed"

# 4. Pull Ollama model (Mistral 7B)
if ! ollama list | grep -q "mistral:7b-instruct-q4_0"; then
  print_info "Pulling mistral:7b-instruct-q4_0..."
  ollama pull mistral:7b-instruct-q4_0 || print_warning "Model pull failed or timed out"
else
  print_status "Model already available"
fi

# 5. Create .env if missing
if [ ! -f "app/.env" ]; then
  cat > app/.env << 'EOF'
# Default Environment
HOST=0.0.0.0
PORT=8000
DEBUG=false
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_STREAMING=true
ENABLE_AUTH=false
ENABLE_SEMANTIC_CLASSIFICATION=false
LOG_LEVEL=INFO
EOF
  print_status ".env created"
else
  print_warning ".env already exists"
fi

# 6. Make startup script executable
chmod +x start_services.sh

# 7. Done!
print_status "Setup complete!"
echo "ℹ️  To start services, run: ./start_services.sh"
