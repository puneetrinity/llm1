# Base image with CUDA
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# CUDA and GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ollama configuration for enhanced performance
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_NUM_PARALLEL=2 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    OLLAMA_GPU_OVERHEAD=0 \
    OLLAMA_DEBUG=INFO

# Memory management - Enhanced configuration
ENV MAX_MEMORY_MB=12288 \
    CACHE_MEMORY_LIMIT_MB=1024 \
    MODEL_MEMORY_LIMIT_MB=6144 \
    SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Enhanced feature toggles - Progressive enablement
ENV ENABLE_SEMANTIC_CLASSIFICATION=false \
    ENABLE_STREAMING=true \
    ENABLE_MODEL_WARMUP=true \
    ENABLE_DETAILED_METRICS=true \
    ENABLE_DASHBOARD=true \
    ENABLE_WEBSOCKET_DASHBOARD=true

# Dashboard configuration
ENV DASHBOARD_PATH=/app/static \
    ENABLE_REACT_DASHBOARD=true

# Performance and caching
ENV ENABLE_REDIS_CACHE=true \
    REDIS_URL=redis://localhost:6379 \
    ENABLE_SEMANTIC_CACHE=true \
    SEMANTIC_SIMILARITY_THRESHOLD=0.85

# Security settings (change in production) - Fixed JSON syntax
ENV ENABLE_AUTH=false \
    DEFAULT_API_KEY=sk-change-me-in-production \
    API_KEY_HEADER=X-API-Key \
    CORS_ORIGINS='["*"]'

# Advanced features
ENV ENABLE_CIRCUIT_BREAKER=true \
    ENABLE_CONNECTION_POOLING=true \
    ENABLE_PERFORMANCE_MONITORING=true

# Core application settings
ENV HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO \
    DEBUG=false

WORKDIR /app

# Install system dependencies including Node.js for React dashboard
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

# Install Node.js 18.x for React dashboard
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and package files first for better caching
COPY requirements.txt ./
COPY frontend/package*.json ./frontend/

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
    prometheus-client \
    || echo "Some enhanced features may be limited"

# Install Node.js dependencies for React dashboard
WORKDIR /app/frontend
RUN if [ -f "package.json" ]; then \
        echo "Installing Node.js dependencies..." && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm install --legacy-peer-deps --prefer-offline --no-optional; \
    else \
        echo "No frontend package.json found"; \
    fi

# Copy frontend source code and build React dashboard
COPY frontend/ ./
RUN if [ -f "package.json" ] && [ -d "src" ]; then \
        echo "Building React frontend..." && \
        GENERATE_SOURCEMAP=false CI=true NODE_OPTIONS="--max-old-space-size=4096" npm run build && \
        echo "React build completed successfully!"; \
    else \
        echo "Skipping frontend build - no source found" && \
        mkdir -p build; \
    fi

# Copy application code
WORKDIR /app
COPY . ./

# Create fallback frontend if build failed
RUN if [ ! -f "frontend/build/index.html" ]; then \
        echo "Creating fallback HTML page..." && \
        mkdir -p frontend/build; \
    else \
        echo "Frontend build exists"; \
    fi

# Copy fallback HTML (clean approach)
COPY fallback.html frontend/build/index.html

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \;

# Create required directories
RUN mkdir -p logs cache models data static frontend/build

# Copy startup script
COPY start.sh ./
RUN chmod +x start.sh

# Comprehensive health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Use the startup script
CMD ["./start.sh"]
