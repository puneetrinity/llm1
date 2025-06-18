# Dockerfile.clean - Clean Enhanced LLM Proxy Image
# Uses the working main.py file with all enhanced features

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Enhanced dependencies for full functionality
RUN pip install --no-cache-dir \
    sentence-transformers \
    scikit-learn \
    sse-starlette \
    python-multipart \
    aiofiles

# Copy application files (ONLY the working ones)
COPY main.py .
COPY config.py .
COPY config_enhanced.py .

# Copy working services
COPY services/ ./services/
COPY utils/ ./utils/
COPY middleware/ ./middleware/
COPY models/ ./models/

# Create directories
RUN mkdir -p logs data cache models static

# Create optimized environment file
RUN cat > .env << 'EOF'
# Enhanced LLM Proxy Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=false
LOG_LEVEL=INFO

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_MAX_LOADED_MODELS=4

# Enhanced Features (ALL ENABLED)
ENABLE_AUTH=false
ENABLE_DASHBOARD=true
ENABLE_ENHANCED_FEATURES=true
ENABLE_WEBSOCKET=false
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_MODEL_ROUTING=true
ENABLE_4_MODEL_ROUTING=true

# Memory Management
MAX_MEMORY_MB=16384
CACHE_MEMORY_LIMIT_MB=2048
MODEL_MEMORY_LIMIT_MB=8192

# Model Configuration
PHI_MODEL=phi3.5
MISTRAL_MODEL=mistral:7b-instruct-q4_0
GEMMA_MODEL=gemma:7b-instruct
LLAMA_MODEL=llama3:8b-instruct-q4_0

# CORS
CORS_ORIGINS=["*"]
CORS_ALLOW_CREDENTIALS=true
EOF

# Create clean startup script
RUN cat > start_clean.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Enhanced LLM Proxy - Clean Start"
echo "=================================="
echo "Using: main.py (complete version)"
echo "Models: Phi3.5 | Mistral 7B | Gemma 7B | Llama3 8B"
echo ""

# Start Ollama
echo "📡 Starting Ollama..."
ollama serve > /tmp/ollama.log 2>&1 &
sleep 15

# Download models if not present
echo "📦 Checking models..."
if ! ollama list | grep -q "phi3.5"; then
    echo "Downloading Phi3.5..."
    ollama pull phi3.5 &
fi

if ! ollama list | grep -q "mistral:7b-instruct-q4_0"; then
    echo "Downloading Mistral 7B..."
    ollama pull mistral:7b-instruct-q4_0 &
fi

if ! ollama list | grep -q "gemma:7b-instruct"; then
    echo "Downloading Gemma 7B..."
    ollama pull gemma:7b-instruct &
fi

if ! ollama list | grep -q "llama3:8b-instruct-q4_0"; then
    echo "Downloading Llama3 8B..."
    ollama pull llama3:8b-instruct-q4_0 &
fi

wait
echo "✅ All models ready"

# Start the complete application
echo "🌐 Starting Enhanced LLM Proxy..."
python main.py
EOF

RUN chmod +x start_clean.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Use the clean startup script
CMD ["./start_clean.sh"]
