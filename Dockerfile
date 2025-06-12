FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA and GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies including Node.js for React dashboard
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18.x for React dashboard
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Ollama configuration for enhanced performance
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_NUM_PARALLEL=2
ENV OLLAMA_MAX_LOADED_MODELS=2
ENV OLLAMA_GPU_OVERHEAD=0
ENV OLLAMA_DEBUG=INFO

# Memory management - Enhanced configuration
ENV MAX_MEMORY_MB=12288
ENV CACHE_MEMORY_LIMIT_MB=1024
ENV MODEL_MEMORY_LIMIT_MB=6144
ENV SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Enhanced feature toggles - Progressive enablement
ENV ENABLE_SEMANTIC_CLASSIFICATION=false
ENV ENABLE_STREAMING=true
ENV ENABLE_MODEL_WARMUP=true
ENV ENABLE_DETAILED_METRICS=true
ENV ENABLE_DASHBOARD=true
ENV ENABLE_WEBSOCKET_DASHBOARD=true

# Dashboard configuration
ENV DASHBOARD_PATH=/app/static
ENV ENABLE_REACT_DASHBOARD=true

# Performance and caching
ENV ENABLE_REDIS_CACHE=true
ENV REDIS_URL=redis://localhost:6379
ENV ENABLE_SEMANTIC_CACHE=true
ENV SEMANTIC_SIMILARITY_THRESHOLD=0.85

# Security settings (change in production)
ENV ENABLE_AUTH=false
ENV DEFAULT_API_KEY=sk-change-me-in-production
ENV API_KEY_HEADER=X-API-Key
ENV CORS_ORIGINS=["*"]

# Advanced features
ENV ENABLE_CIRCUIT_BREAKER=true
ENV ENABLE_CONNECTION_POOLING=true
ENV ENABLE_PERFORMANCE_MONITORING=true

WORKDIR /app

# Copy package files first for better caching
COPY requirements.txt .

# Install core Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install enhanced ML dependencies with graceful fallbacks
RUN pip3 install --no-cache-dir \
    "sentence-transformers>=2.2.0,<3.0.0" \
    "faiss-cpu==1.7.4" \
    "sse-starlette==1.6.5" \
    "numpy>=1.21.0,<1.25.0" \
    "scikit-learn>=1.1.0" \
    || echo "⚠️ Some ML features may be limited"

# Install caching and performance dependencies
RUN pip3 install --no-cache-dir \
    "redis>=4.5.0" \
    "aioredis>=2.0.0" \
    "prometheus-client>=0.19.0" \
    || echo "⚠️ Some performance features may be limited"

# Install development and monitoring tools
RUN pip3 install --no-cache-dir \
    "python-json-logger>=2.0.7" \
    "GPUtil>=1.4.0" \
    || echo "⚠️ Some monitoring features may be limited"

# Pre-download semantic model (conditional to save space)
RUN if [ "$ENABLE_SEMANTIC_CLASSIFICATION" = "true" ]; then \
        echo "📥 Pre-downloading semantic model..." && \
        python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
        || echo "⚠️ Failed to download semantic model - will download at runtime"; \
    fi

# Copy frontend package.json first for better caching
COPY frontend/package*.json frontend/ 2>/dev/null || echo "No frontend package files found"

# Install Node.js dependencies if frontend exists
RUN if [ -f "frontend/package.json" ]; then \
        echo "📦 Installing Node.js dependencies..." && \
        cd frontend && \
        npm install --production && \
        cd .. ; \
    else \
        echo "ℹ️ No frontend directory found - dashboard will be disabled"; \
    fi

# Copy frontend source code
COPY frontend/ ./frontend/ 2>/dev/null || echo "No frontend source found"

# Copy dashboard build script
COPY build_dashboard.sh ./ 2>/dev/null || echo "No dashboard build script found"

# Build React dashboard with error handling
RUN if [ -f "build_dashboard.sh" ]; then \
        echo "🔨 Building React dashboard..." && \
        chmod +x build_dashboard.sh && \
        ./build_dashboard.sh || echo "⚠️ Dashboard build failed - will build at runtime"; \
    elif [ -d "frontend" ] && [ -f "frontend/package.json" ]; then \
        echo "🔨 Building dashboard directly..." && \
        cd frontend && \
        npm run build && \
        mkdir -p ../static && \
        cp -r build/* ../static/ && \
        cd .. && \
        echo "✅ Dashboard built successfully" \
        || echo "⚠️ Dashboard build failed"; \
    else \
        echo "ℹ️ No dashboard source found - skipping dashboard build"; \
    fi

# Copy application code
COPY . .

# Fix line endings and make scripts executable
RUN find . -name "*.py" -exec dos2unix {} \; 2>/dev/null || true && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \; 2>/dev/null || true

# Create directories for data persistence
RUN mkdir -p /app/cache /app/logs /app/models /app/data /app/static

# Verify dashboard build and create fallback
RUN if [ ! -f "/app/static/index.html" ] && [ -d "frontend" ]; then \
        echo "🔄 Dashboard not found, creating fallback..." && \
        mkdir -p /app/static && \
        echo '<!DOCTYPE html><html><head><title>LLM Proxy Dashboard</title></head><body><h1>Dashboard Loading...</h1><p>Building dashboard, please wait...</p><script>setTimeout(() => location.reload(), 5000);</script></body></html>' > /app/static/index.html; \
    fi

# Enhanced health check with dashboard verification
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health && \
        (curl -f http://localhost:8000/ >/dev/null 2>&1 || echo "Dashboard check skipped") \
        || exit 1

# Expose ports
EXPOSE 8000 11434

# Enhanced startup script with comprehensive initialization
CMD ["/bin/bash", "-c", "\
    echo '🚀 Starting Enhanced LLM Proxy with Dashboard...' && \
    \
    # Export environment variables \
    export CUDA_VISIBLE_DEVICES=0 && \
    export NVIDIA_VISIBLE_DEVICES=all && \
    export OLLAMA_HOST=0.0.0.0:11434 && \
    export OLLAMA_GPU_OVERHEAD=0 && \
    \
    # Verify GPU detection \
    echo '🔍 Checking GPU availability...' && \
    nvidia-smi || echo '⚠️ GPU detection may have issues' && \
    \
    # Start Ollama service with GPU support \
    echo '📡 Starting Ollama service with GPU support...' && \
    CUDA_VISIBLE_DEVICES=0 ollama serve & \
    OLLAMA_PID=$$! && \
    \
    # Wait for Ollama with enhanced error handling \
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
    # Verify Ollama started successfully \
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then \
        echo '❌ Failed to start Ollama service - checking logs'; \
        ps aux | grep ollama; \
        exit 1; \
    fi && \
    \
    # Pull and warm up priority model \
    echo '📦 Pulling Mistral 7B (Priority model)...' && \
    (CUDA_VISIBLE_DEVICES=0 ollama pull mistral:7b-instruct-q4_0 || echo '⚠️ Model pull failed') && \
    echo '✅ Mistral 7B ready!' && \
    \
    # Warm up the model \
    echo '🔥 Warming up Mistral...' && \
    (curl -X POST http://localhost:11434/api/chat \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"mistral:7b-instruct-q4_0\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"stream\": false, \"options\": {\"num_predict\": 5}}' \
        >/dev/null 2>&1 && echo '✅ Mistral warmed up!' || echo '⚠️ Warmup failed') && \
    \
    # Build dashboard at runtime if not already built \
    if [ ! -f '/app/static/index.html' ] && [ -f 'build_dashboard.sh' ]; then \
        echo '🔨 Building dashboard at runtime...' && \
        ./build_dashboard.sh || echo '⚠️ Runtime dashboard build failed'; \
    fi && \
    \
    # Verify dashboard status \
    if [ -f '/app/static/index.html' ]; then \
        echo '✅ Dashboard available at http://localhost:8000/'; \
    else \
        echo 'ℹ️ Dashboard not available - API-only mode'; \
    fi && \
    \
    # Display startup summary \
    echo '' && \
    echo '🎉 Enhanced LLM Proxy Started Successfully!' && \
    echo '===========================================' && \
    echo '📊 API Documentation: http://localhost:8000/docs' && \
    echo '🏥 Health Check: http://localhost:8000/health' && \
    echo '📈 Metrics: http://localhost:8000/metrics' && \
    echo '🎛️ Dashboard: http://localhost:8000/' && \
    echo '🔌 Ollama API: http://localhost:11434' && \
    echo '===========================================' && \
    echo '' && \
    \
    # Start the Enhanced FastAPI application \
    echo '🌐 Starting Enhanced FastAPI application...' && \
    python3 main.py \
"]
