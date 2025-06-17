# Complete LLM Proxy with Ollama and 4 Models
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS="*"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ca-certificates \
    gnupg \
    software-properties-common \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/data/models /app/logs /var/log/supervisor

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ALL enhanced features
RUN pip3 install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    sse-starlette \
    redis \
    aioredis \
    prometheus-client \
    numpy \
    scikit-learn \
    transformers \
    torch \
    torchvision \
    torchaudio

# Copy application code
COPY . .

# Build frontend if exists
RUN if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then \
        cd frontend && \
        npm install --legacy-peer-deps && \
        npm run build && \
        cp -r build/* /app/static/ 2>/dev/null || mkdir -p /app/static; \
    else \
        mkdir -p /app/static && \
        echo '<!DOCTYPE html><html><head><title>LLM Proxy Complete</title></head><body><h1>🚀 LLM Proxy Complete</h1><p>API: <a href="/v1/models">/v1/models</a></p><p>Health: <a href="/health">/health</a></p></body></html>' > /app/static/index.html; \
    fi

# Create enhanced .env configuration
RUN cat > .env << 'EOF'
# Complete LLM Proxy Configuration
PORT=8001
HOST=0.0.0.0
DEBUG=false
LOG_LEVEL=INFO

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_ORIGINS=*

# Enhanced Features - ALL ENABLED
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_CACHING=true
ENABLE_MONITORING=true

# Memory Configuration
MAX_MEMORY_MB=16384
MODEL_MEMORY_LIMIT_MB=8192
CACHE_MEMORY_LIMIT_MB=2048

# Model Configuration
DEFAULT_MODEL=mistral:7b-instruct-q4_0
FALLBACK_MODEL=llama3.2:3b-instruct-q4_0

# Security
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-complete-proxy-key

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300
ENABLE_METRICS=true
EOF

# Create model download script
RUN cat > /app/download_models.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

sleep 10

echo "📥 Downloading 4 LLM models..."

# Model 1: Mistral 7B (General purpose)
echo "📦 Downloading Mistral 7B..."
ollama pull mistral:7b-instruct-q4_0

# Model 2: Llama 3.2 3B (Fast responses)
echo "📦 Downloading Llama 3.2 3B..."
ollama pull llama3.2:3b-instruct-q4_0

# Model 3: Codellama (Code generation)
echo "📦 Downloading CodeLlama 7B..."
ollama pull codellama:7b-instruct-q4_0

# Model 4: Gemma 2B (Lightweight)
echo "📦 Downloading Gemma 2B..."
ollama pull gemma2:2b-instruct-q4_0

echo "✅ All 4 models downloaded!"
ollama list

# Keep Ollama running
wait $OLLAMA_PID
EOF

RUN chmod +x /app/download_models.sh

# Create supervisor configuration
RUN cat > /etc/supervisor/conf.d/supervisord.conf << 'EOF'
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[program:ollama]
command=ollama serve
environment=OLLAMA_HOST="0.0.0.0:11434",OLLAMA_ORIGINS="*"
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/ollama.err.log
stdout_logfile=/var/log/supervisor/ollama.out.log

[program:llm-proxy]
command=python3 main.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/llm-proxy.err.log
stdout_logfile=/var/log/supervisor/llm-proxy.out.log
environment=PYTHONPATH="/app"
EOF

# Create startup script that downloads models then starts services
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Complete LLM Proxy..."

# Start Ollama in background
echo "📡 Starting Ollama..."
ollama serve &
sleep 10

# Download models if not already present
if [ ! -f "/root/.ollama/models/manifests/registry.ollama.ai/library/mistral/7b-instruct-q4_0" ]; then
    echo "📥 Downloading models for first time..."
    
    ollama pull mistral:7b-instruct-q4_0 &
    ollama pull llama3.2:3b-instruct-q4_0 &
    ollama pull codellama:7b-instruct-q4_0 &
    ollama pull gemma2:2b-instruct-q4_0 &
    
    wait
    echo "✅ All models ready!"
else
    echo "✅ Models already downloaded"
fi

# List available models
echo "📋 Available models:"
ollama list

# Start the proxy
echo "🚀 Starting LLM Proxy..."
cd /app
exec python3 main.py
EOF

RUN chmod +x /app/start.sh

# Create health check script
RUN cat > /app/health_check.sh << 'EOF'
#!/bin/bash
# Check if both Ollama and proxy are running
curl -f http://localhost:11434/api/tags >/dev/null 2>&1 && \
curl -f http://localhost:8001/health >/dev/null 2>&1
EOF

RUN chmod +x /app/health_check.sh

# Expose ports
EXPOSE 8001 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/health_check.sh

# Set volumes for model persistence
VOLUME ["/root/.ollama"]

# Default command
CMD ["/app/start.sh"]
