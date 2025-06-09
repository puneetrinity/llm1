#!/bin/bash
# setup.sh - Complete LLM Proxy Setup for RunPod
# Place this file in your GitHub repo root

set -e
echo "🚀 Starting Enhanced LLM Proxy Setup..."

# Check if running in /workspace
if [[ "$PWD" != "/workspace" ]]; then
    echo "⚠️  Switching to /workspace..."
    cd /workspace
fi

export DEBIAN_FRONTEND=noninteractive

echo "📦 Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

echo "🤖 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama for /workspace
export OLLAMA_MODELS=/workspace/.ollama
export OLLAMA_HOST=0.0.0.0:11434
mkdir -p /workspace/.ollama

echo "🐍 Setting up Python environment..."
cd /workspace
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip

echo "📚 Installing Python dependencies..."
pip install --no-cache-dir \
    fastapi[all]==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    psutil==5.9.6 \
    PyJWT==2.8.0 \
    python-multipart==0.0.6

echo "🧠 Installing ML dependencies..."
pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    faiss-cpu==1.7.4 \
    numpy==1.24.3 \
    sse-starlette==1.6.5 \
    huggingface_hub==0.14.1

echo "🔮 Pre-downloading transformer model..."
export TRANSFORMERS_CACHE=/workspace/.cache/transformers
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/.cache/transformers
mkdir -p /workspace/.cache/huggingface

python -c "
import os
try:
    from sentence_transformers import SentenceTransformer
    print('📥 Downloading sentence transformer model...')
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/workspace/.cache/transformers')
    print('✅ Model downloaded successfully!')
except Exception as e:
    print(f'⚠️ Model download failed: {e}')
"

echo "📁 Setting up application..."
cd /workspace
rm -rf /workspace/app
git clone https://github.com/puneetrinity/llm1.git app
cd /workspace/app

echo "⚙️ Creating configuration..."
cat > /workspace/app/.env << 'EOF'
# Enhanced LLM Proxy Configuration
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_MAX_RETRIES=3

# Enable enhanced features
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true
ENABLE_AUTH=false
ENABLE_RATE_LIMITING=true
DEFAULT_RATE_LIMIT=60

# Memory settings
MAX_MEMORY_MB=8192
MODEL_MEMORY_LIMIT_MB=4096
CACHE_MEMORY_LIMIT_MB=1024
SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Resource limits
MAX_CONCURRENT_REQUESTS=10
MAX_QUEUE_SIZE=100
REQUEST_TIMEOUT=300

# Model settings
DEFAULT_MODEL=mistral:7b-instruct-q4_0
MAX_TOKENS=2048
DEFAULT_TEMPERATURE=0.7

# Cache settings
ENABLE_CACHE=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# CORS settings
CORS_ORIGINS=["*"]
CORS_ALLOW_CREDENTIALS=true

# Enhanced feature limits
FAISS_INDEX_MAX_SIZE=10000
CLASSIFICATION_CACHE_MAX_SIZE=1000

# Cache directories
TRANSFORMERS_CACHE=/workspace/.cache/transformers
HF_HOME=/workspace/.cache/huggingface
EOF

echo "🔧 Creating startup script..."
cat > /workspace/app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Enhanced LLM Proxy..."

# Set environment
export PATH=/usr/local/bin:$PATH
export OLLAMA_MODELS=/workspace/.ollama
export OLLAMA_HOST=0.0.0.0:11434
export TRANSFORMERS_CACHE=/workspace/.cache/transformers
export HF_HOME=/workspace/.cache/huggingface

# Activate Python environment
source /workspace/venv/bin/activate

# Ensure directories exist
mkdir -p /workspace/.cache/transformers
mkdir -p /workspace/.cache/huggingface

cd /workspace/app

echo "📡 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama..."
for i in {1..30}; do
  if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  echo "   Attempt $i/30 - waiting 3 seconds..."
  sleep 3
done

if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "❌ Failed to start Ollama"
  exit 1
fi

echo "📦 Pulling essential models..."
echo "   📥 Pulling Mistral 7B..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

echo "   📥 Pulling DeepSeek V2..."
ollama pull deepseek-v2:7b-q4_0 &

echo "   ⏳ Waiting for Mistral..."
wait $MISTRAL_PID
echo "   ✅ Mistral ready!"

echo "🌐 Starting FastAPI application..."
python main.py

cleanup() {
  echo "🛑 Shutting down..."
  kill $OLLAMA_PID 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT

wait
EOF

chmod +x /workspace/app/start.sh

echo "📊 Storage summary:"
echo "Container filesystem:"
df -h / | head -2
echo ""
echo "/workspace volume:"
df -h /workspace | head -2
echo ""

echo "✅ Setup complete! Starting application..."
cd /workspace/app
source /workspace/venv/bin/activate
./start.sh
