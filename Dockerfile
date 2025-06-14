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

# Verify Node.js installation
RUN echo "=== Node.js Version Check ===" && \
    node --version && \
    npm --version

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

# === FRONTEND BUILD SECTION FOR CREATE REACT APP ===
# Copy frontend source code first
COPY frontend/ ./frontend/

# Detect project type and validate structure
RUN echo "=== Frontend Project Detection ===" && \
    echo "Contents of /app/frontend/:" && \
    ls -la /app/frontend/ && \
    echo "" && \
    PROJECT_TYPE="unknown" && \
    if [ -f "/app/frontend/package.json" ]; then \
        echo "✅ package.json found" && \
        echo "" && \
        echo "🔍 Detecting project type..." && \
        if grep -q "react-scripts" /app/frontend/package.json; then \
            PROJECT_TYPE="create-react-app" && \
            echo "✅ Create React App detected (react-scripts)" && \
            echo "CRA version:" && \
            grep -A 1 -B 1 "react-scripts" /app/frontend/package.json; \
        elif grep -q "vite" /app/frontend/package.json; then \
            PROJECT_TYPE="vite" && \
            echo "✅ Vite project detected" && \
            ls -la /app/frontend/vite.config.* 2>/dev/null || echo "No vite.config found"; \
        elif grep -q "next" /app/frontend/package.json; then \
            PROJECT_TYPE="nextjs" && \
            echo "✅ Next.js project detected"; \
        else \
            echo "⚠️ Unknown React project type" && \
            echo "Available scripts:" && \
            cat /app/frontend/package.json | grep -A 10 '"scripts"'; \
        fi && \
        echo "PROJECT_TYPE=$PROJECT_TYPE" > /tmp/project_type; \
    else \
        echo "❌ No package.json found"; \
        exit 1; \
    fi && \
    echo "" && \
    if [ -d "/app/frontend/src" ]; then \
        echo "✅ src directory found" && \
        echo "Source files:" && \
        find /app/frontend/src -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" | head -10; \
    else \
        echo "❌ src directory not found"; \
        exit 1; \
    fi

# Install Node.js dependencies with Create React App specific handling
WORKDIR /app/frontend
RUN echo "📦 Installing dependencies for Create React App..." && \
    npm config set fund false && \
    npm config set audit-level none && \
    npm config set legacy-peer-deps true && \
    echo "" && \
    echo "🔧 Fixing ajv dependency conflict..." && \
    npm install --legacy-peer-deps 2>&1 | tee /tmp/npm-install.log && \
    echo "" && \
    echo "🔧 Installing/fixing ajv dependencies..." && \
    npm install ajv@^8.0.0 ajv-keywords@^5.0.0 --legacy-peer-deps --save-dev 2>/dev/null || \
    npm install ajv@^7.0.0 ajv-keywords@^4.0.0 --legacy-peer-deps --save-dev 2>/dev/null || \
    echo "Could not fix ajv automatically" && \
    echo "" && \
    echo "✅ Dependencies installed. Checking critical packages..." && \
    if [ -d "node_modules/react-scripts" ]; then \
        echo "✅ react-scripts installed:" && \
        cat node_modules/react-scripts/package.json | grep '"version"' | head -1; \
    else \
        echo "❌ react-scripts not found"; \
        exit 1; \
    fi && \
    if [ -d "node_modules/ajv" ]; then \
        echo "✅ ajv version:" && \
        cat node_modules/ajv/package.json | grep '"version"' | head -1; \
    else \
        echo "⚠️ ajv not found"; \
    fi && \
    echo "Total packages installed: $(ls node_modules/ | wc -l)"

# Build Create React App with dependency fixes
RUN echo "🏗️ Building Create React App..." && \
    echo "Pre-build environment setup..." && \
    export NODE_OPTIONS="--max-old-space-size=4096 --openssl-legacy-provider" && \
    export GENERATE_SOURCEMAP=false && \
    export CI=true && \
    export BUILD_PATH=build && \
    echo "" && \
    echo "🔧 Attempting to fix any remaining dependency issues..." && \
    (npm audit fix --force 2>/dev/null || echo "Audit fix skipped") && \
    echo "" && \
    echo "🚀 Starting Create React App build..." && \
    npm run build 2>&1 | tee /tmp/cra-build.log && \
    echo "" && \
    echo "🔍 Checking Create React App build output..." && \
    if [ -d "build" ] && [ -f "build/index.html" ]; then \
        echo "✅ Create React App build successful!" && \
        echo "Build directory contents:" && \
        ls -la build/ && \
        echo "" && \
        echo "Static assets:" && \
        find build -name "*.js" -o -name "*.css" | head -10 && \
        echo "" && \
        echo "Index.html size: $(wc -c < build/index.html) bytes" && \
        if [ $(wc -c < build/index.html) -lt 200 ]; then \
            echo "⚠️ Index.html seems small, checking contents:" && \
            cat build/index.html; \
        fi; \
    else \
        echo "❌ Create React App build failed" && \
        echo "" && \
        echo "Build log analysis:" && \
        echo "Errors found:" && \
        grep -i "error\|failed\|cannot find module" /tmp/cra-build.log | tail -10 && \
        echo "" && \
        echo "Full build log (last 30 lines):" && \
        tail -30 /tmp/cra-build.log && \
        echo "" && \
        echo "Available directories:" && \
        ls -la && \
        exit 1; \
    fi

# Copy application code
WORKDIR /app
COPY . ./

# Final verification for Create React App
RUN echo "=== FINAL VERIFICATION ===" && \
    echo "✅ Create React App build:" && \
    ls -la /app/frontend/build/index.html && \
    echo "" && \
    echo "✅ Build contents:" && \
    ls -la /app/frontend/build/ && \
    echo "" && \
    echo "✅ Python files:" && \
    ls -la /app/main*.py && \
    echo "" && \
    echo "✅ Start script:" && \
    ls -la /app/start.sh 2>/dev/null || echo "No start.sh found" && \
    echo "" && \
    echo "🎉 Container verification complete!"

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
