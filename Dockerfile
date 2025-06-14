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

# Install system dependencies including Node.js for React + Vite
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

# Install Node.js 18.x for Vite
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
# Copy frontend source code first
COPY frontend/ ./frontend/

# Check if frontend exists and has the right structure
RUN echo "=== Checking Frontend Structure ===" && \
    ls -la /app/frontend/ && \
    echo "Looking for package.json:" && \
    ls -la /app/frontend/package.json 2>/dev/null || echo "❌ No package.json found" && \
    echo "Looking for vite.config:" && \
    ls -la /app/frontend/vite.config.* 2>/dev/null || echo "No vite config found" && \
    echo "Looking for src directory:" && \
    ls -la /app/frontend/src/ 2>/dev/null || echo "No src directory found"

# Install Node.js dependencies for your React + Vite frontend
WORKDIR /app/frontend
RUN if [ -f "package.json" ]; then \
        echo "🎯 Installing Node.js dependencies for React + Vite..." && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm install --prefer-offline && \
        echo "✅ Dependencies installed successfully"; \
    else \
        echo "❌ No package.json found - cannot install dependencies"; \
        exit 1; \
    fi

# Build the React + Vite frontend
RUN if [ -f "package.json" ] && [ -d "src" ]; then \
        echo "🏗️ Building React + Vite frontend..." && \
        npm run build && \
        echo "✅ Vite build completed successfully!" && \
        echo "Built files:" && \
        ls -la dist/ 2>/dev/null || ls -la build/ 2>/dev/null || echo "No build output found"; \
    else \
        echo "❌ Cannot build - missing package.json or src directory"; \
        exit 1; \
    fi

# Copy the application code
WORKDIR /app
COPY . ./

# === DEBUGGING SECTION ===
RUN echo "=== DEBUGGING: Final Structure Check ===" && \
    echo "App directory:" && ls -la /app/ && \
    echo "Frontend build output:" && \
    (ls -la /app/frontend/dist/ 2>/dev/null || ls -la /app/frontend/build/ 2>/dev/null || echo "No build directory found") && \
    echo "Looking for main files:" && \
    find /app -name "main*.py" | head -5 && \
    echo "Start script:" && \
    (ls -la /app/start.sh && echo "Contents:" && head -10 /app/start.sh) || echo "No start.sh found"

# Test Python imports
RUN echo "=== Testing Python Imports ===" && \
    python3 -c "import sys; sys.path.insert(0, '/app')" && \
    (python3 -c "import main_master" && echo "✅ main_master imports successfully") || \
    (python3 -c "import main" && echo "✅ main imports successfully") || \
    echo "❌ No main files import successfully"

# Copy frontend build to the right location for FastAPI to serve
RUN echo "=== Setting up Frontend for FastAPI ===" && \
    if [ -d "/app/frontend/dist" ]; then \
        echo "📁 Copying Vite dist to frontend/build for FastAPI compatibility..." && \
        cp -r /app/frontend/dist /app/frontend/build && \
        ls -la /app/frontend/build/ && \
        echo "✅ Frontend ready for FastAPI"; \
    elif [ -d "/app/frontend/build" ]; then \
        echo "✅ Frontend build directory already exists"; \
    else \
        echo "❌ No frontend build found - creating fallback"; \
        mkdir -p /app/frontend/build && \
        printf '%s\n' \
            '<!DOCTYPE html>' \
            '<html lang="en">' \
            '<head><meta charset="UTF-8"><title>LLM Proxy</title></head>' \
            '<body><h1>🚀 LLM Proxy Dashboard</h1><p>Frontend build failed - using fallback</p></body>' \
            '</html>' \
        > /app/frontend/build/index.html; \
    fi

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

# Final verification
RUN echo "=== FINAL VERIFICATION ===" && \
    echo "✅ Start script:" && ls -la /app/start.sh && \
    echo "✅ Frontend:" && ls -la /app/frontend/build/index.html && \
    echo "✅ Python files:" && ls -la /app/main*.py && \
    echo "🎉 Container ready!"

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Start the application
CMD ["./start.sh"]
