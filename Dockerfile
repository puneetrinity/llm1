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

# === FRONTEND BUILD SECTION WITH ERROR HANDLING ===
# Copy frontend source code first
COPY frontend/ ./frontend/

# Check frontend structure with detailed output
RUN echo "=== Frontend Structure Analysis ===" && \
    echo "Contents of /app/frontend/:" && \
    ls -la /app/frontend/ && \
    echo "" && \
    echo "🔍 Checking for Vite project indicators..." && \
    VITE_CONFIG_FOUND=false && \
    PACKAGE_JSON_FOUND=false && \
    SRC_DIR_FOUND=false && \
    echo "" && \
    if [ -f "/app/frontend/vite.config.js" ] || [ -f "/app/frontend/vite.config.ts" ] || [ -f "/app/frontend/vite.config.mjs" ]; then \
        VITE_CONFIG_FOUND=true && \
        echo "✅ Vite config found:" && \
        ls -la /app/frontend/vite.config.* && \
        echo "Config contents:" && \
        cat /app/frontend/vite.config.* | head -15; \
    else \
        echo "❌ No vite.config found - this may not be a Vite project"; \
        echo "Looking for other config files:" && \
        ls -la /app/frontend/*config* 2>/dev/null || echo "No config files found"; \
    fi && \
    echo "" && \
    if [ -f "/app/frontend/package.json" ]; then \
        PACKAGE_JSON_FOUND=true && \
        echo "✅ package.json found" && \
        echo "Checking for Vite in dependencies..." && \
        if grep -q "vite" /app/frontend/package.json; then \
            echo "✅ Vite found in package.json" && \
            grep -A 2 -B 2 "vite" /app/frontend/package.json; \
        else \
            echo "❌ Vite not found in package.json - may not be a Vite project"; \
        fi && \
        echo "" && \
        echo "Build script check:" && \
        if grep -q '"build"' /app/frontend/package.json; then \
            echo "✅ Build script found:" && \
            grep -A 1 -B 1 '"build"' /app/frontend/package.json; \
        else \
            echo "❌ No build script found in package.json"; \
        fi; \
    else \
        echo "❌ package.json not found"; \
    fi && \
    echo "" && \
    if [ -d "/app/frontend/src" ]; then \
        SRC_DIR_FOUND=true && \
        echo "✅ src directory found" && \
        echo "Source files:" && \
        find /app/frontend/src -type f -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.vue" | head -10; \
    else \
        echo "❌ src directory not found"; \
    fi && \
    echo "" && \
    echo "🎯 Project validation:" && \
    if [ "$VITE_CONFIG_FOUND" = "true" ] && [ "$PACKAGE_JSON_FOUND" = "true" ] && [ "$SRC_DIR_FOUND" = "true" ]; then \
        echo "✅ Valid Vite project detected - proceeding with build"; \
    elif [ "$PACKAGE_JSON_FOUND" = "true" ] && [ "$SRC_DIR_FOUND" = "true" ]; then \
        echo "⚠️ Node.js project found but no Vite config - will attempt build anyway"; \
    else \
        echo "❌ Invalid project structure - missing required components"; \
        echo "Required: package.json + src/ directory"; \
        echo "Recommended: vite.config.js for Vite projects"; \
    fi

# Install Node.js dependencies with Vite-specific validation
WORKDIR /app/frontend
RUN echo "🔧 Validating Vite project before dependency installation..." && \
    if [ ! -f "package.json" ]; then \
        echo "❌ FATAL: No package.json found in frontend directory" && \
        echo "Cannot proceed with frontend build" && \
        exit 1; \
    fi && \
    if [ ! -f "vite.config.js" ] && [ ! -f "vite.config.ts" ] && [ ! -f "vite.config.mjs" ]; then \
        echo "⚠️ WARNING: No vite.config found - this may not be a Vite project" && \
        echo "Checking package.json for Vite dependency..." && \
        if ! grep -q "vite" package.json; then \
            echo "❌ FATAL: No Vite found in package.json and no vite.config" && \
            echo "This doesn't appear to be a Vite project" && \
            echo "Package.json scripts:" && \
            cat package.json | grep -A 10 '"scripts"' || echo "No scripts found" && \
            exit 1; \
        else \
            echo "✅ Vite found in package.json, proceeding..."; \
        fi; \
    else \
        echo "✅ Vite config found - confirmed Vite project"; \
    fi && \
    echo "" && \
    echo "📦 Installing dependencies for Vite project..." && \
    npm config set fund false && \
    npm config set audit-level none && \
    npm config set prefer-offline true && \
    npm config set cache /tmp/npm-cache && \
    echo "" && \
    npm install --verbose 2>&1 | tee /tmp/npm-install.log && \
    echo "" && \
    echo "✅ Dependencies installed. Checking for Vite installation..." && \
    if [ -d "node_modules/vite" ]; then \
        echo "✅ Vite installed successfully:" && \
        ls -la node_modules/vite/package.json && \
        echo "Vite version: $(cat node_modules/vite/package.json | grep '"version"' | head -1)"; \
    else \
        echo "❌ Vite not found in node_modules" && \
        echo "Installed packages:" && \
        ls node_modules/ | head -10; \
    fi && \
    echo "Total packages installed: $(ls node_modules/ | wc -l)"

# Build Vite frontend with enhanced validation and error handling
RUN echo "🏗️ Starting Vite build process..." && \
    echo "Pre-build validation:" && \
    if [ ! -f "package.json" ] || [ ! -d "src" ]; then \
        echo "❌ FATAL: Missing package.json or src directory" && \
        exit 1; \
    fi && \
    echo "✅ Required files present" && \
    echo "" && \
    echo "Checking build script..." && \
    BUILD_SCRIPT=$(cat package.json | grep -A 1 '"build"' | grep -o '"[^"]*"' | tail -1 | tr -d '"') && \
    echo "Build script command: $BUILD_SCRIPT" && \
    if echo "$BUILD_SCRIPT" | grep -q "vite"; then \
        echo "✅ Confirmed Vite build script"; \
    else \
        echo "⚠️ Build script doesn't mention Vite: $BUILD_SCRIPT"; \
        echo "Will attempt build anyway..."; \
    fi && \
    echo "" && \
    echo "Checking Vite installation..." && \
    if [ -d "node_modules/vite" ]; then \
        VITE_VERSION=$(cat node_modules/vite/package.json | grep '"version"' | cut -d'"' -f4) && \
        echo "✅ Vite $VITE_VERSION installed"; \
    else \
        echo "❌ Vite not found in node_modules" && \
        echo "Available build tools:" && \
        ls node_modules/ | grep -E "(vite|webpack|rollup|parcel)" || echo "No common build tools found" && \
        echo "Will attempt build anyway..."; \
    fi && \
    echo "" && \
    echo "🔨 Starting Vite build..." && \
    NODE_OPTIONS="--max-old-space-size=4096" \
    GENERATE_SOURCEMAP=false \
    CI=true \
    npm run build 2>&1 | tee /tmp/vite-build.log && \
    echo "" && \
    echo "🔍 Checking Vite build output..." && \
    if [ -d "dist" ] && [ -f "dist/index.html" ]; then \
        echo "✅ Vite build successful! Output in dist/ directory:" && \
        ls -la dist/ && \
        echo "" && \
        echo "Index.html preview:" && \
        head -10 dist/index.html; \
    elif [ -d "build" ] && [ -f "build/index.html" ]; then \
        echo "✅ Build successful! Output in build/ directory:" && \
        ls -la build/; \
    else \
        echo "❌ Vite build failed or produced no output" && \
        echo "" && \
        echo "Build log (last 30 lines):" && \
        tail -30 /tmp/vite-build.log 2>/dev/null || echo "No build log available" && \
        echo "" && \
        echo "Checking for any output files:" && \
        find . -name "*.html" -newer /tmp/vite-build.log 2>/dev/null || echo "No HTML files generated" && \
        echo "" && \
        echo "Available directories:" && \
        ls -la | grep ^d && \
        echo "This indicates a Vite build failure" && \
        exit 1; \
    fi

# Handle Vite build output (dist/ directory is Vite standard)
RUN echo "📁 Processing Vite build output..." && \
    if [ -d "dist" ] && [ -f "dist/index.html" ]; then \
        echo "✅ Vite build successful - dist/ directory contains:" && \
        ls -la dist/ && \
        echo "" && \
        echo "Vite assets structure:" && \
        find dist -type f | head -20 && \
        echo "" && \
        echo "Index.html size: $(wc -c < dist/index.html) bytes" && \
        if [ $(wc -c < dist/index.html) -lt 100 ]; then \
            echo "⚠️ Index.html seems very small, checking contents:" && \
            cat dist/index.html; \
        fi; \
    elif [ -d "build" ] && [ -f "build/index.html" ]; then \
        echo "✅ Build successful - build/ directory found (non-standard for Vite)" && \
        ls -la build/; \
    else \
        echo "❌ FATAL: Vite build produced no valid output" && \
        echo "" && \
        echo "Expected: dist/index.html (Vite standard)" && \
        echo "Found directories:" && \
        ls -la && \
        echo "" && \
        echo "Vite build log analysis:" && \
        if [ -f "/tmp/vite-build.log" ]; then \
            echo "Errors in build log:" && \
            grep -i "error\|failed\|cannot" /tmp/vite-build.log | tail -10 || echo "No obvious errors found" && \
            echo "" && \
            echo "Full build log (last 20 lines):" && \
            tail -20 /tmp/vite-build.log; \
        fi && \
        echo "" && \
        echo "Cannot proceed without valid Vite build output" && \
        exit 1; \
    fi

# Copy application code
WORKDIR /app
COPY . ./

# Setup Vite build for FastAPI serving
RUN echo "🔗 Setting up Vite build for FastAPI..." && \
    cd /app/frontend && \
    if [ -d "dist" ] && [ -f "dist/index.html" ]; then \
        echo "📁 Copying Vite dist/ to build/ for FastAPI compatibility..." && \
        cp -r dist build && \
        echo "✅ Vite build ready at /app/frontend/build/" && \
        echo "Build contents:" && \
        ls -la /app/frontend/build/ && \
        echo "" && \
        echo "Static assets:" && \
        find /app/frontend/build -name "*.js" -o -name "*.css" | head -10; \
    elif [ -d "build" ] && [ -f "build/index.html" ]; then \
        echo "✅ Build directory already exists and ready (unusual for Vite)" && \
        ls -la /app/frontend/build/; \
    else \
        echo "❌ FATAL: No valid Vite build output to serve" && \
        echo "Expected: /app/frontend/dist/ directory from Vite build" && \
        exit 1; \
    fi

# Final debugging and verification
RUN echo "=== FINAL VERIFICATION ===" && \
    echo "✅ Frontend build status:" && \
    ls -la /app/frontend/build/index.html && \
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
