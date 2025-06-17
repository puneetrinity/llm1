# Enhanced 4-Model LLM Proxy - Complete Production Dockerfile
# Supports: Phi-3.5, Mistral 7B, Gemma 7B, Llama3 8B with smart routing
# Features: React Dashboard, FastAPI Backend, Ollama Integration, Enhanced Routing

# =============================================================================
# Stage 1: Frontend Build (React Dashboard)
# =============================================================================
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Install build dependencies for Alpine
RUN apk add --no-cache python3 make g++

# Copy package files first for better caching
COPY frontend/package*.json ./

# Install dependencies with legacy peer deps support
RUN npm ci --legacy-peer-deps --silent || npm install --legacy-peer-deps --silent

# Copy frontend source code
COPY frontend/ ./

# Build the React app
RUN npm run build && ls -la build/ && test -f build/index.html

# =============================================================================
# Stage 2: Python Dependencies (Multi-arch compatible)
# =============================================================================
FROM python:3.11-slim as python-deps

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
WORKDIR /app
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install enhanced features (optional, won't fail if unavailable)
RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    sse-starlette \
    redis \
    aioredis \
    prometheus-client \
    || echo "Some enhanced features may not be available"

# =============================================================================
# Stage 3: Production Image
# =============================================================================
FROM python:3.11-slim as production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# 4-Model System Configuration
# =============================================================================
ENV DEFAULT_MODEL="mistral:7b-instruct-q4_0" \
    ENABLE_4_MODEL_ROUTING=true \
    PHI_MODEL="phi:3.5" \
    MISTRAL_MODEL="mistral:7b-instruct-q4_0" \
    GEMMA_MODEL="gemma:7b-instruct" \
    LLAMA_MODEL="llama3:8b-instruct-q4_0"

# =============================================================================
# Ollama Configuration (Updated for 4 models)
# =============================================================================
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_NUM_PARALLEL=4 \
    OLLAMA_MAX_LOADED_MODELS=4 \
    OLLAMA_GPU_OVERHEAD=0 \
    OLLAMA_DEBUG=INFO

# =============================================================================
# Memory Management (Optimized for 4 models)
# =============================================================================
ENV MAX_MEMORY_MB=20480 \
    CACHE_MEMORY_LIMIT_MB=4096 \
    MODEL_MEMORY_LIMIT_MB=12288 \
    SEMANTIC_MODEL_MAX_MEMORY_MB=1024

# =============================================================================
# Enhanced Features Configuration
# =============================================================================
ENV ENABLE_SEMANTIC_CLASSIFICATION=true \
    ENABLE_STREAMING=true \
    ENABLE_MODEL_WARMUP=true \
    ENABLE_DETAILED_METRICS=true \
    ENABLE_DASHBOARD=true \
    ENABLE_REACT_DASHBOARD=true \
    ENABLE_WEBSOCKET_DASHBOARD=true \
    ENABLE_ENHANCED_ROUTING=true

# =============================================================================
# Performance & Caching
# =============================================================================
ENV ENABLE_REDIS_CACHE=true \
    REDIS_URL=redis://redis:6379 \
    ENABLE_SEMANTIC_CACHE=true \
    SEMANTIC_SIMILARITY_THRESHOLD=0.85 \
    ENABLE_CIRCUIT_BREAKER=true \
    ENABLE_CONNECTION_POOLING=true \
    ENABLE_PERFORMANCE_MONITORING=true

# =============================================================================
# Security & CORS
# =============================================================================
ENV ENABLE_AUTH=false \
    CORS_ORIGINS='["*"]' \
    CORS_ALLOW_CREDENTIALS=true

# =============================================================================
# Core Application Settings
# =============================================================================
ENV HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO \
    DEBUG=false \
    DASHBOARD_PATH=/app/static

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for any runtime frontend needs)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy Python dependencies from previous stage
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy built frontend from build stage
COPY --from=frontend-build /app/frontend/build ./static
RUN ls -la ./static/ && test -f ./static/index.html

# Copy application code
COPY . .

# Copy frontend source (for development/debugging)
COPY frontend/ ./frontend/

# Create necessary directories with proper permissions
RUN mkdir -p data/{cache,logs,models} logs cache models static && \
    chmod 755 data data/cache data/logs data/models logs cache models static

# Make scripts executable
RUN find . -name "*.sh" -type f -exec chmod +x {} \; || true

# Create a comprehensive startup script
RUN cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Enhanced 4-Model LLM Proxy"
echo "======================================"
echo "Models: Phi-3.5 | Mistral 7B | Gemma 7B | Llama3 8B"
echo ""

# Start Ollama in background if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "📡 Starting Ollama service..."
    ollama serve > /app/logs/ollama.log 2>&1 &
    sleep 10
    
    # Wait for Ollama to be ready
    for i in {1..30}; do
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama is ready"
            break
        fi
        echo "   Waiting for Ollama... ($i/30)"
        sleep 2
    done
fi

# Download models if AUTO_DOWNLOAD_MODELS is enabled
if [ "$AUTO_DOWNLOAD_MODELS" = "true" ]; then
    echo "📦 Auto-downloading 4 models..."
    
    models=(
        "phi:3.5:🧠 Phi-3.5 (Reasoning)"
        "mistral:7b-instruct-q4_0:⚡ Mistral 7B (General)"
        "gemma:7b-instruct:⚙️ Gemma 7B (Technical)"
        "llama3:8b-instruct-q4_0:🎨 Llama3 8B (Creative)"
    )
    
    for model_info in "${models[@]}"; do
        IFS=':' read -r model desc <<< "$model_info"
        echo "Downloading $desc..."
        ollama pull "$model" || echo "Failed to download $model"
    done
    
    echo "✅ Model downloads complete"
fi

# Start the main application
echo "🌐 Starting FastAPI application..."
echo "📊 Configuration:"
echo "   • Host: $HOST"
echo "   • Port: $PORT"
echo "   • Debug: $DEBUG"
echo "   • 4-Model Routing: $ENABLE_4_MODEL_ROUTING"
echo "   • Enhanced Features: $ENABLE_SEMANTIC_CLASSIFICATION"
echo ""

# Determine which Python file to use
if [ -f "main_master.py" ]; then
    MAIN_FILE="main_master.py"
elif [ -f "main_with_react.py" ]; then
    MAIN_FILE="main_with_react.py"
elif [ -f "main.py" ]; then
    MAIN_FILE="main.py"
else
    echo "❌ No suitable main file found!"
    exit 1
fi

echo "📂 Using: $MAIN_FILE"
echo "🎯 Access points:"
echo "   • Main API: http://localhost:$PORT"
echo "   • Dashboard: http://localhost:$PORT/app"
echo "   • Health: http://localhost:$PORT/health"
echo "   • API Docs: http://localhost:$PORT/docs"
echo ""

exec python "$MAIN_FILE"
EOF

RUN chmod +x start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Set proper ownership
RUN chown -R nobody:nogroup /app/data /app/logs /app/cache /app/models || true

# Create download script for manual model downloads
RUN cat > download_models.sh << 'EOF'
#!/bin/bash
echo "📦 Downloading 4 Models for Enhanced LLM Proxy"
echo "==============================================="

models=(
    "phi:3.5"
    "mistral:7b-instruct-q4_0" 
    "gemma:7b-instruct"
    "llama3:8b-instruct-q4_0"
)

for model in "${models[@]}"; do
    echo "📥 Downloading $model..."
    ollama pull "$model" || echo "⚠️ Failed to download $model"
done

echo "✅ Download complete!"
ollama list
EOF

RUN chmod +x download_models.sh

# Default command
CMD ["./start.sh"]

# Labels for better organization
LABEL maintainer="LLM Proxy Team" \
      version="2.2.0" \
      description="Enhanced 4-Model LLM Proxy with React Dashboard" \
      models="phi-3.5,mistral-7b,gemma-7b,llama3-8b" \
      features="smart-routing,semantic-classification,react-dashboard"
