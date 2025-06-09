#!/bin/bash
# setup.sh - Clean LLM Proxy Setup for RunPod
set -e

echo "🚀 Starting LLM Proxy Setup..."
cd /workspace
mkdir -p logs

# Fix any package issues first
echo "🔧 Fixing package system..."
export DEBIAN_FRONTEND=noninteractive
dpkg --configure -a || true
apt-get clean
apt-get update -qq

# Install essential packages
echo "📦 Installing system packages..."
apt-get install -y --no-install-recommends curl python3-pip python3-venv git wget

# Install Ollama
echo "🤖 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Set up environment
export OLLAMA_MODELS=/workspace/.ollama
export PATH=/usr/local/bin:$PATH
mkdir -p /workspace/.ollama

# Check Ollama installation
if [ ! -f "/usr/local/bin/ollama" ]; then
    echo "❌ Ollama installation failed"
    exit 1
fi

echo "🐍 Setting up Python environment..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --no-cache-dir \
    fastapi \
    uvicorn \
    aiohttp \
    pydantic \
    pydantic-settings \
    psutil

# Get application code
echo "📁 Getting application code..."
rm -rf /workspace/app
git clone https://github.com/puneetrinity/llm1.git /workspace/app
cd /workspace/app

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/app/logs
mkdir -p /workspace/app/data/cache
mkdir -p /workspace/app/data/logs
mkdir -p /workspace/app/data/models

# Create .env file
echo "⚙️ Creating configuration..."
cat > .env << 'EOF'
DEBUG=false
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_AUTH=false
DEFAULT_MODEL=mistral:7b-instruct-q4_0
MAX_MEMORY_MB=4096
CORS_ORIGINS=["*"]
EOF

# Start services
echo "🔧 Starting services..."
/usr/local/bin/ollama serve &
OLLAMA_PID=$!

# Wait for Ollama
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 3
done

# Check if Ollama started
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull model
echo "📦 Pulling Mistral model..."
/usr/local/bin/ollama pull mistral:7b-instruct-q4_0

# Start the application
echo "🌐 Starting FastAPI application..."
source /workspace/venv/bin/activate
cd /workspace/app
python3 main.py

# Cleanup on exit
cleanup() {
    echo "🛑 Shutting down..."
    kill $OLLAMA_PID 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT
