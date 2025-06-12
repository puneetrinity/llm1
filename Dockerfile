# Dockerfile.comprehensive - Complete Production Deployment
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    nodejs \
    npm \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set Ollama environment variables for optimal performance
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_NUM_PARALLEL=2
ENV OLLAMA_MAX_LOADED_MODELS=2
ENV OLLAMA_GPU_OVERHEAD=0

# Memory management for comprehensive deployment
ENV MAX_MEMORY_MB=12288
ENV CACHE_MEMORY_LIMIT_MB=1024
ENV MODEL_MEMORY_LIMIT_MB=6144
ENV SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Enable all comprehensive features
ENV ENABLE_SEMANTIC_CLASSIFICATION=true
ENV ENABLE_STREAMING=true
ENV ENABLE_MODEL_WARMUP=true
ENV ENABLE_DETAILED_METRICS=true
ENV ENABLE_DASHBOARD=true
ENV DASHBOARD_PATH=/app

# Security settings for production
ENV ENABLE_AUTH=false
ENV DEFAULT_API_KEY=sk-change-me-in-production
ENV API_KEY_HEADER=X-API-Key

WORKDIR /app

# Copy package files first for better caching
COPY requirements.txt .
COPY frontend/package*.json frontend/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install enhanced dependencies with error handling
RUN pip3 install --no-cache-dir \
    sentence-transformers>=2.2.0 \
    faiss-cpu==1.7.4 \
    sse-starlette==1.6.5 \
    redis>=4.5.0 \
    aioredis>=2.0.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.1.0 \
    || echo "Some enhanced features may be limited"

# Install Node.js dependencies and build React dashboard
WORKDIR /app/frontend
RUN npm install
COPY frontend/ .
RUN npm run build

# Copy application code
WORKDIR /app
COPY . .

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; && \
    find . -name "*.sh" -exec dos2unix {} \; && \
    chmod +x *.sh

# Create required directories
RUN mkdir -p /app/cache /app/logs /app/models /app/data

# Comprehensive health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 11434

# Use comprehensive startup script
CMD ["/bin/bash", "-c", "\
    echo '🚀 Starting Comprehensive Enhanced LLM Proxy...' && \
    \
    # Export GPU environment variables \
    export CUDA_VISIBLE_DEVICES=0 && \
    export NVIDIA_VISIBLE_DEVICES=all && \
    export OLLAMA_HOST=0.0.0.0:11434 && \
    \
    # Verify GPU detection \
    echo '🔍 Checking GPU availability...' && \
    nvidia-smi || echo 'GPU detection may have issues' && \
    \
    # Start Ollama service \
    echo '📡 Starting Ollama service...' && \
    CUDA_VISIBLE_DEVICES=0 ollama serve & \
    OLLAMA_PID=$$! && \
    \
    # Wait for Ollama with timeout \
    echo '⏳ Waiting for Ollama to start...' && \
    for i in {1..60}; do \
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then \
            echo '✅ Ollama is ready!'; \
            break; \
        fi; \
        echo \"   Attempt $$i/60 - waiting 5 seconds...\"; \
        sleep 5; \
    done && \
    \
    # Verify Ollama started \
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then \
        echo '❌ Failed to start Ollama service'; \
        exit 1; \
    fi && \
    \
    # Pull essential models in background \
    echo '📦 Pulling essential models...' && \
    (CUDA_VISIBLE_DEVICES=0 ollama pull mistral:7b-instruct-q4_0 &) && \
    \
    # Start the comprehensive FastAPI application \
    echo '🌐 Starting Comprehensive FastAPI application...' && \
    python3 main.py \
"]
