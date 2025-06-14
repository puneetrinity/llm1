# Multi-arch Dockerfile - Uses CUDA on AMD64, Ubuntu on ARM64
# Frontend build stage
FROM node:18-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build && test -f build/index.html

# Conditional base image based on architecture
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS base-amd64
FROM ubuntu:22.04 AS base-arm64

# Use the appropriate base image
FROM base-${TARGETARCH} AS production

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# CUDA and GPU environment variables (only effective on AMD64)
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ollama configuration
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_NUM_PARALLEL=2 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    OLLAMA_GPU_OVERHEAD=0 \
    OLLAMA_DEBUG=INFO

# Memory management
ENV MAX_MEMORY_MB=12288 \
    CACHE_MEMORY_LIMIT_MB=1024 \
    MODEL_MEMORY_LIMIT_MB=6144 \
    SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Feature toggles
ENV ENABLE_SEMANTIC_CLASSIFICATION=false \
    ENABLE_STREAMING=true \
    ENABLE_MODEL_WARMUP=true \
    ENABLE_DETAILED_METRICS=true \
    ENABLE_DASHBOARD=true \
    ENABLE_WEBSOCKET_DASHBOARD=false

# Core settings
ENV HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO \
    DEBUG=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    dos2unix \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt ./
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
    prometheus-client \
    || echo "Some enhanced features may be limited"

# Copy built frontend
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Copy application code
COPY *.py ./

# Create directories
RUN mkdir -p data logs

# Rest of your existing Dockerfile content...
EXPOSE 8001 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python3", "main.py"]
