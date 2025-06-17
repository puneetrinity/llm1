# Enhanced 4-Model LLM Proxy Dockerfile
# Frontend built externally - Space optimized
# Supports: Phi-3.5, Mistral 7B, Gemma 7B, Llama3 8B with smart routing

FROM python:3.11-slim

# Metadata
LABEL maintainer="LLM Proxy Team" \
      version="2.2.0" \
      description="Enhanced 4-Model LLM Proxy with Pre-built Frontend" \
      models="phi-3.5,mistral-7b,gemma-7b,llama3-8b" \
      frontend="pre-built"

# =============================================================================
# Environment Variables
# =============================================================================

# System settings
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
# Ollama Configuration
# =============================================================================
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_NUM_PARALLEL=4 \
    OLLAMA_MAX_LOADED_MODELS=4 \
    OLLAMA_GPU_OVERHEAD=0 \
    OLLAMA_DEBUG=INFO

# =============================================================================
# Memory Management (Production Optimized)
# =============================================================================
ENV MAX_MEMORY_MB=16384 \
    CACHE_MEMORY_LIMIT_MB=2048 \
    MODEL_MEMORY_LIMIT_MB=8192 \
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
# Application Settings
# =============================================================================
ENV HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO \
    DEBUG=false \
    DASHBOARD_PATH=/app/static

# =============================================================================
# System Dependencies Installation
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# =============================================================================
# Python Dependencies Installation
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libopenblas-dev \
    libomp-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Copy requirements first for better Docker layer caching
COPY requirements.txt ./

# Install core Python dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Install enhanced features (with fallback)
RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    sse-starlette \
    redis \
    aioredis \
    prometheus-client \
    numpy \
    scikit-learn \
    || echo "⚠️ Some enhanced features may be limited" && \
    pip cache purge || true

# =============================================================================
# Application Code and Static Files
# =============================================================================

# Copy pre-built frontend static files
# These should be built externally and placed in frontend/build/ or static/
COPY frontend/build/ ./static/
# Alternative: COPY static/ ./static/

# Verify frontend files exist
RUN ls -la ./static/ && \
    test -f ./static/index.html && \
    echo "✅ Frontend files verified" || \
    (echo "❌ Frontend build not found! Please build frontend first:" && \
     echo "   cd frontend && npm install && npm run build" && \
     exit 1)

# Copy application source code
COPY main*.py ./
COPY config*.py ./
COPY services/ ./services/
COPY *.sh ./

# Copy frontend source (for reference/debugging)
COPY frontend/package*.json ./frontend/
COPY frontend/src/ ./frontend/src/

# =============================================================================
# Directory Setup and Permissions
# =============================================================================

# Create application directories
RUN mkdir -p \
    data/cache \
    data/logs \
    data/models \
    logs \
    cache \
    models \
    && chmod 755 data data/cache data/logs data/models logs cache models static

# Make scripts executable
RUN find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

# =============================================================================
# Startup Script Creation
# =============================================================================
RUN cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Enhanced 4-Model LLM Proxy Starting"
echo "======================================"
echo "Version: 2.2.0"
echo "Frontend: Pre-built"
echo "Models: Phi-3.5 | Mistral 7B | Gemma 7B | Llama3 8B"
echo ""

# Environment info
echo "📊 Configuration:"
echo "  • Host: $HOST"
echo "  • Port: $PORT"
echo "  • Debug: $DEBUG"
echo "  • 4-Model Routing: $ENABLE_4_MODEL_ROUTING"
echo "  • Enhanced Features: $ENABLE_SEMANTIC_CLASSIFICATION"
echo "  • Ollama URL: $OLLAMA_BASE_URL"
echo ""

# Wait for Ollama service
echo "📡 Waiting for Ollama service..."
OLLAMA_READY=false
for i in {1..60}; do
    if curl -f $OLLAMA_BASE_URL/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        OLLAMA_READY=true
        break
    fi
    echo "   Waiting for Ollama... ($i/60)"
    sleep 2
done

if [ "$OLLAMA_READY" = false ]; then
    echo "⚠️ Ollama not responding, but starting anyway..."
    echo "   Make sure Ollama service is running at: $OLLAMA_BASE_URL"
fi

# Auto-download models if enabled
if [ "$AUTO_DOWNLOAD_MODELS" = "true" ]; then
    echo "📦 Auto-downloading 4 models..."
    
    models=("$PHI_MODEL" "$MISTRAL_MODEL" "$GEMMA_MODEL" "$LLAMA_MODEL")
    model_names=("🧠 Phi-3.5" "⚡ Mistral 7B" "⚙️ Gemma 7B" "🎨 Llama3 8B")
    
    for i in "${!models[@]}"; do
        model="${models[$i]}"
        name="${model_names[$i]}"
        echo "Downloading $name ($model)..."
        curl -X POST $OLLAMA_BASE_URL/api/pull \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"$model\"}" \
            >/dev/null 2>&1 || echo "⚠️ Failed to download $model"
    done
    
    echo "✅ Model downloads initiated"
fi

# Determine main file
MAIN_FILE=""
if [ -f "main_master.py" ]; then
    MAIN_FILE="main_master.py"
elif [ -f "main_with_react.py" ]; then
    MAIN_FILE="main_with_react.py"
elif [ -f "main.py" ]; then
    MAIN_FILE="main.py"
else
    echo "❌ No suitable main file found!"
    echo "Available Python files:"
    ls -la *.py
    exit 1
fi

echo "📂 Using main file: $MAIN_FILE"
echo ""
echo "🎯 Access points will be:"
echo "  • Dashboard: http://localhost:$PORT/app"
echo "  • API Docs:  http://localhost:$PORT/docs"
echo "  • Health:    http://localhost:$PORT/health"
echo "  • Metrics:   http://localhost:$PORT/metrics"
echo ""
echo "🌐 Starting FastAPI application..."

# Start the application
exec python "$MAIN_FILE"
EOF

RUN chmod +x start.sh

# =============================================================================
# Model Download Helper Script
# =============================================================================
RUN cat > download_models.sh << 'EOF'
#!/bin/bash
echo "📦 Downloading 4 Models for Enhanced LLM Proxy"
echo "==============================================="

models=(
    "phi:3.5:🧠 Phi-3.5 (Math & Reasoning)"
    "mistral:7b-instruct-q4_0:⚡ Mistral 7B (General & Quick Facts)"
    "gemma:7b-instruct:⚙️ Gemma 7B (Technical & Coding)"
    "llama3:8b-instruct-q4_0:🎨 Llama3 8B (Creative & Conversations)"
)

for model_info in "${models[@]}"; do
    IFS=':' read -r model desc <<< "$model_info"
    echo "📥 Downloading $desc..."
    
    if command -v ollama >/dev/null 2>&1; then
        ollama pull "$model" || echo "⚠️ Failed to download $model"
    else
        curl -X POST http://localhost:11434/api/pull \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"$model\"}" || echo "⚠️ Failed to download $model"
    fi
done

echo ""
echo "✅ Model downloads complete!"
echo "📊 Verify with: curl http://localhost:11434/api/tags"
EOF

RUN chmod +x download_models.sh

# =============================================================================
# Health Check Configuration
# =============================================================================
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# =============================================================================
# Security and User Configuration
# =============================================================================

# Create non-root user for security
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -d /app -s /bin/bash appuser && \
    chown -R appuser:appgroup /app

# Expose port
EXPOSE 8001

# Switch to non-root user
USER appuser

# =============================================================================
# Default Command
# =============================================================================
CMD ["./start.sh"]

# =============================================================================
# Build Instructions
# =============================================================================

# To build this image:
# 1. First build the frontend:
#    cd frontend && npm install && npm run build && cd ..
#
# 2. Then build the Docker image:
#    docker build -t llm-proxy-enhanced:latest .
#
# 3. Or use with docker-compose:
#    docker-compose up --build
#
# 4. For production with specific models:
#    docker run -e AUTO_DOWNLOAD_MODELS=true llm-proxy-enhanced:latest
