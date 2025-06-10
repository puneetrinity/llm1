#!/bin/bash
set -e

echo "🚀 Starting LLM Proxy Setup..."
cd /workspace
mkdir -p logs

# Fix package system
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
curl -fsSL https://ollama.com/install.sh | sh

# Set up environment
export OLLAMA_MODELS=/workspace/.ollama
export PATH=/usr/local/bin:$PATH
mkdir -p "$OLLAMA_MODELS"

echo "🐍 Setting up Python environment..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --no-cache-dir fastapi[all]==0.104.1 uvicorn[standard]==0.24.0 aiohttp==3.9.1 pydantic-settings==2.1.0 psutil==5.9.6

# Get application code
echo "📁 Getting application code..."
rm -rf /workspace/app
git clone https://github.com/puneetrinity/llm1.git /workspace/app || {
    echo "⚠️  Git clone failed, creating app directory..."
    mkdir -p /workspace/app
}
cd /workspace/app

# Create directories
echo "📁 Creating directories..."
mkdir -p logs data/cache data/logs security services utils models middleware

# Create fixed configuration
echo "⚙️ Creating fixed configuration..."
cat > config_enhanced.py << 'EOF'
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List

class EnhancedSettings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    DEBUG: bool = Field(default=False)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_TIMEOUT: int = Field(default=300)
    DEFAULT_MODEL: str = Field(default="mistral:7b-instruct-q4_0")
    MAX_MEMORY_MB: int = Field(default=4096)
    CORS_ORIGINS: List[str] = Field(default=["*"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    ENABLE_AUTH: bool = Field(default=False)
    ENABLE_RATE_LIMITING: bool = Field(default=False)
    DEFAULT_RATE_LIMIT: str = Field(default="100/hour")
    ENABLE_SEMANTIC_CLASSIFICATION: bool = Field(default=False)
    ENABLE_STREAMING: bool = Field(default=True)
    ENABLE_MODEL_WARMUP: bool = Field(default=True)

def get_settings():
    return EnhancedSettings()
EOF

# Create .env file
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

# Start Ollama
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
echo "📦 Pulling mistral model..."
ollama pull mistral:7b-instruct-q4_0 &

# Start the application
echo "🌐 Starting FastAPI application..."
source /workspace/venv/bin/activate
cd /workspace/app

# Check which main file exists and start it
if [ -f main.py ]; then
    python3 main.py
elif [ -f main_enhanced.py ]; then
    python3 main_enhanced.py
else
    echo "❌ No main.py found. Creating a basic one..."
    # Create a minimal main.py if none exists
    cat > main.py << 'MAIN_EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Basic configuration
class Settings:
    DEBUG = False
    HOST = "0.0.0.0"
    PORT = 8000
    CORS_ORIGINS = ["*"]

settings = Settings()

app = FastAPI(title="LLM Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "LLM Proxy is running"}

@app.get("/")
async def root():
    return {"message": "LLM Proxy API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT)
MAIN_EOF
    python3 main.py
fi

# Cleanup function
cleanup() {
    echo "🛑 Shutting down services..."
    kill $OLLAMA_PID 2>/dev/null || true
}

trap cleanup EXIT SIGTERM SIGINT
