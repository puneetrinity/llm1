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
    ENABLE_WEBSOCKET_DASHBOARD=false

# Dashboard configuration
ENV DASHBOARD_PATH=/app/static \
    ENABLE_REACT_DASHBOARD=true

# Performance and caching
ENV ENABLE_REDIS_CACHE=true \
    REDIS_URL=redis://localhost:6379 \
    ENABLE_SEMANTIC_CACHE=true \
    SEMANTIC_SIMILARITY_THRESHOLD=0.85

# Security settings - Set at runtime for production security
ENV ENABLE_AUTH=false \
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

# Install system dependencies including Node.js for React
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

# Install Node.js 18.x for Create React App
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy Python requirements first
COPY requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install enhanced Python dependencies with error handling
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

# === FRONTEND BUILD SECTION ===
# Copy frontend source code with fixed package.json
COPY frontend/ ./frontend/

# Verify the fixed package.json
RUN echo "=== Verifying Fixed package.json ===" && \
    echo "Checking for dependency resolutions..." && \
    if grep -q "resolutions\|overrides" /app/frontend/package.json; then \
        echo "✅ Dependency resolutions found in package.json" && \
        echo "Resolutions:" && \
        grep -A 10 '"resolutions"' /app/frontend/package.json 2>/dev/null || echo "No resolutions section" && \
        echo "Overrides:" && \
        grep -A 10 '"overrides"' /app/frontend/package.json 2>/dev/null || echo "No overrides section"; \
    else \
        echo "⚠️ No dependency resolutions found - may still have ajv conflicts"; \
    fi

# Install dependencies with the fixed package.json
WORKDIR /app/frontend
RUN echo "📦 Installing Create React App dependencies..." && \
    npm config set fund false && \
    npm config set audit-level none && \
    echo "" && \
    echo "Using npm install with resolved dependencies..." && \
    npm install 2>&1 | tee /tmp/npm-install.log && \
    echo "" && \
    echo "✅ Dependencies installed. Checking ajv resolution..." && \
    if [ -d "node_modules/ajv" ]; then \
        echo "AJV version installed:" && \
        cat node_modules/ajv/package.json | grep '"version"' | head -1 && \
        echo "Checking ajv structure..." && \
        if [ -f "node_modules/ajv/dist/compile/codegen/index.js" ] || [ -f "node_modules/ajv/dist/compile/codegen.js" ]; then \
            echo "✅ AJV codegen found - dependency conflict resolved"; \
        else \
            echo "⚠️ AJV codegen structure:" && \
            find node_modules/ajv -name "*codegen*" 2>/dev/null | head -5 || echo "No codegen files found"; \
        fi; \
    else \
        echo "❌ AJV not installed"; \
    fi && \
    if [ -d "node_modules/ajv-keywords" ]; then \
        echo "AJV-Keywords version:" && \
        cat node_modules/ajv-keywords/package.json | grep '"version"' | head -1; \
    fi

# Build Create React App
RUN echo "🏗️ Building Create React App..." && \
    echo "Setting up build environment..." && \
    export NODE_OPTIONS="--max-old-space-size=4096" && \
    export GENERATE_SOURCEMAP=false && \
    export CI=true && \
    echo "" && \
    echo "🚀 Running react-scripts build..." && \
    npm run build 2>&1 | tee /tmp/build.log && \
    echo "" && \
    echo "🔍 Checking build output..." && \
    if [ -d "build" ] && [ -f "build/index.html" ]; then \
        echo "✅ Create React App build successful!" && \
        echo "Build contents:" && \
        ls -la build/ && \
        echo "Build size:" && \
        du -sh build/ && \
        echo "Static assets:" && \
        find build -name "*.js" -o -name "*.css" | wc -l && echo " files generated"; \
    else \
        echo "❌ Build failed!" && \
        echo "Build log errors:" && \
        grep -i "error\|failed" /tmp/build.log | tail -10 && \
        echo "" && \
        echo "Available directories:" && \
        ls -la && \
        exit 1; \
    fi

# Copy application code
WORKDIR /app
COPY . ./

# Final verification
RUN echo "=== FINAL VERIFICATION ===" && \
    echo "✅ React build:" && \
    ls -la /app/frontend/build/index.html && \
    echo "✅ Python files:" && \
    ls -la /app/main*.py && \
    echo "🎉 Container ready!"

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \;

# Create required directories
RUN mkdir -p logs cache models data static

# Ensure start.sh exists and is executable
RUN if [ ! -f "/app/start.sh" ]; then \
        echo "Creating fallback start.sh..." && \
        printf '%s\n' \
            '#!/bin/bash' \
            'echo "🚀 Starting LLM Proxy Server..."' \
            'echo "📍 Server: http://0.0.0.0:8001"' \
            'echo "📊 Dashboard: http://0.0.0.0:8001/app"' \
            'python3 -m uvicorn main_master:app --host 0.0.0.0 --port 8001 --reload' \
        > /app/start.sh && \
        chmod +x /app/start.sh; \
    else \
        chmod +x /app/start.sh; \
    fi

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Start the application
CMD ["./start.sh"]
