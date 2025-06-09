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
apt-get install -y --no-install-recommends curl python3-pip python3-venv git wget build-essential

# Install Ollama
echo "🤖 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh -o install.sh
chmod +x install.sh
./install.sh

# Move Ollama to persistent location if needed
mkdir -p /workspace/ollama/bin
mv /usr/local/bin/ollama /workspace/ollama/bin/ollama || true
chmod +x /workspace/ollama/bin/ollama

# Set up environment
export OLLAMA_MODELS=/workspace/.ollama
export PATH=/workspace/ollama/bin:$PATH
mkdir -p "$OLLAMA_MODELS"

# Check Ollama installation
if [ ! -f "/workspace/ollama/bin/ollama" ]; then
    echo "❌ Ollama installation failed"
    exit 1
fi

echo "🐍 Setting up Python environment..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --no-cache-dir \
    fastapi[all]==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    sentence-transformers==2.2.2 \
    faiss-cpu==1.7.4 \
    numpy==1.24.3 \
    sse-starlette==1.6.5 \
    psutil==5.9.6 \
    pydantic-settings==2.1.0 \
    PyJWT==2.8.0 \
    huggingface_hub==0.14.1

# Get application code
echo "📁 Getting application code..."
rm -rf /workspace/app
git clone https://github.com/puneetrinity/llm1.git /workspace/app
cd /workspace/app

# Create required folders
echo "📁 Creating directories..."
mkdir -p logs data/cache data/logs data/models

# Create .env config
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

# Start Ollama service
echo "📡 Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -fs http://localhost:11434/api/tags >/dev/null; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 3
done

if ! curl -fs http://localhost:11434/api/tags >/dev/null; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull model
echo "📦 Pulling mistral:7b-instruct-q4_0..."
ollama pull mistral:7b-instruct-q4_0

# Start FastAPI app
echo "🌐 Starting FastAPI application..."
source /workspace/venv/bin/activate
cd /workspace/app

if [ -f main.py ]; then
    python3 main.py
elif [ -f main_enhanced.py ]; then
    python3 main_enhanced.py
else
    echo "❌ No main.py or main_enhanced.py found. Please check your app entrypoint."
    exit 1
fi

# Cleanup
cleanup() {
    echo "🛑 Shutting down services..."
    kill $OLLAMA_PID 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT
