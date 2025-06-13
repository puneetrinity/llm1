# Dockerfile - Consolidated LLM Proxy with GPU Support
# Combines: CUDA + Ollama + React Frontend + FastAPI Backend

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA and GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

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

# Dashboard configuration - FIXED paths
ENV DASHBOARD_PATH=/app
ENV ENABLE_REACT_DASHBOARD=true

# Performance and caching
ENV ENABLE_REDIS_CACHE=false
ENV REDIS_URL=redis://localhost:6379
ENV ENABLE_SEMANTIC_CACHE=true
ENV SEMANTIC_SIMILARITY_THRESHOLD=0.85

# Security settings (change in production)
ENV ENABLE_AUTH=false
ENV DEFAULT_API_KEY=sk-change-me-in-production
ENV API_KEY_HEADER=X-API-Key

# Advanced features
ENV ENABLE_CIRCUIT_BREAKER=true
ENV ENABLE_CONNECTION_POOLING=true
ENV ENABLE_PERFORMANCE_MONITORING=true

# App configuration - FIXED for consolidated setup
ENV HOST=0.0.0.0
ENV PORT=8001
ENV LOG_LEVEL=INFO
ENV DEBUG=false

WORKDIR /app

# ============================================================================
# STAGE 1: SYSTEM DEPENDENCIES
# ============================================================================

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
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18.x for React dashboard
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# ============================================================================
# STAGE 2: PYTHON DEPENDENCIES
# ============================================================================

# Copy requirements first for better caching
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

# ============================================================================
# STAGE 3: FRONTEND BUILD (FIXED - NO SHELL OPERATORS IN COPY)
# ============================================================================

# Create frontend directory structure
RUN mkdir -p frontend/src frontend/public

# Copy frontend files with proper error handling
# First, copy package.json files if they exist
COPY frontend/package*.json frontend/

# Check if frontend package.json exists and install dependencies
RUN if [ -f "frontend/package.json" ]; then \
        echo "📦 Installing Node.js dependencies..." && \
        cd frontend && \
        npm install --production && \
        cd .. ; \
    else \
        echo "ℹ️ No frontend package.json found - creating minimal structure" && \
        echo '{"name":"minimal-frontend","version":"1.0.0","scripts":{"build":"echo No build needed"}}' > frontend/package.json; \
    fi

# Copy all frontend source code
COPY frontend/ ./frontend/

# Build React dashboard with comprehensive error handling
RUN if [ -f "frontend/src/App.tsx" ] || [ -f "frontend/src/App.js" ]; then \
        echo "🔨 Building React dashboard..." && \
        cd frontend && \
        npm run build && \
        cd .. && \
        echo "✅ Dashboard built successfully" && \
        ls -la frontend/build/; \
    else \
        echo "ℹ️ No React source found - creating fallback dashboard" && \
        mkdir -p frontend/build && \
        echo '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 LLM Proxy Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .container { background: rgba(255,255,255,0.1); padding: 40px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2); }
        h1 { margin-bottom: 20px; }
        .status { background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin: 20px 0; }
        .links { margin-top: 30px; }
        .links a { color: #fff; text-decoration: none; margin: 0 10px; padding: 10px 20px; background: rgba(255,255,255,0.2); border-radius: 5px; display: inline-block; margin: 5px; }
        .links a:hover { background: rgba(255,255,255,0.3); }
        .health-check { margin: 20px 0; }
        .refresh-btn { background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%); border: none; padding: 10px 20px; border-radius: 25px; color: white; cursor: pointer; font-weight: bold; }
    </style>
    <script>
        async function checkHealth() {
            try {
                const response = await fetch("/health");
                const data = await response.json();
                document.getElementById("health-status").innerHTML = 
                    `<span style="color: #4CAF50;">✅ ${data.status}</span> - Version: ${data.version}`;
            } catch (error) {
                document.getElementById("health-status").innerHTML = 
                    `<span style="color: #f44336;">❌ Connection Error</span>`;
            }
        }
        
        // Check health on load and every 30 seconds
        window.onload = function() {
            checkHealth();
            setInterval(checkHealth, 30000);
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>🚀 LLM Proxy API</h1>
        <div class="status">
            <strong>Status:</strong> <span id="health-status">Checking...</span>
        </div>
        <div class="health-check">
            <button class="refresh-btn" onclick="checkHealth()">🔄 Refresh Status</button>
        </div>
        <p>React dashboard not built - API endpoints available below:</p>
        <div class="links">
            <a href="/health">💚 Health Check</a>
            <a href="/docs">📚 API Documentation</a>
            <a href="/api/status">📊 Status API</a>
            <a href="/metrics">📈 Metrics</a>
        </div>
        <div style="margin-top: 30px; font-size: 14px; opacity: 0.8;">
            <p>🔧 Enhanced LLM Proxy with GPU Support</p>
            <p>🎯 Ollama API: <a href="http://localhost:11434" style="color: #fff;">localhost:11434</a></p>
        </div>
    </div>
</body>
</html>' > frontend/build/index.html; \
    fi

# ============================================================================
# STAGE 4: APPLICATION CODE
# ============================================================================

# Copy consolidated application files
COPY config.py main_master.py ./

# Copy environment template and startup scripts
COPY .env.template ./
COPY container_start.sh ./

# Copy any additional application files
COPY . .

# ============================================================================
# STAGE 5: FINALIZATION
# ============================================================================

# Fix line endings and make scripts executable
RUN find . -name "*.py" -exec dos2unix {} \; 2>/dev/null || true && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \; 2>/dev/null || true

# Create directories for data persistence
RUN mkdir -p /app/cache /app/logs /app/models /app/data

# Create non-root user for security (but keep root for GPU access)
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser \
    && chown -R appuser:appuser /app

# Verify dashboard build
RUN if [ -f "frontend/build/index.html" ]; then \
        echo "✅ Dashboard verified at frontend/build/index.html"; \
        ls -la frontend/build/; \
    else \
        echo "❌ Dashboard verification failed"; \
        ls -la frontend/; \
    fi

# Enhanced health check with dashboard verification
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports (updated for consolidated setup)
EXPOSE 8001 11434

# Pre-download semantic model (conditional to save space)
RUN if [ "$ENABLE_SEMANTIC_CLASSIFICATION" = "true" ]; then \
        echo "📥 Pre-downloading semantic model..." && \
        python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
        || echo "⚠️ Failed to download semantic model - will download at runtime"; \
    fi

# ============================================================================
# STAGE 6: STARTUP SCRIPT
# ============================================================================

# Enhanced startup script with comprehensive initialization
CMD ["/bin/bash", "-c", "\
    echo '🚀 Starting Consolidated Enhanced LLM Proxy...' && \
    \
    # Export environment variables \
    export CUDA_VISIBLE_DEVICES=0 && \
    export NVIDIA_VISIBLE_DEVICES=all && \
    export OLLAMA_HOST=0.0.0.0:11434 && \
    export OLLAMA_GPU_OVERHEAD=0 && \
    export HOST=0.0.0.0 && \
    export PORT=8001 && \
    \
    # Verify GPU detection \
    echo '🔍 Checking GPU availability...' && \
    nvidia-smi || echo '⚠️ GPU detection may have issues' && \
    \
    # Start Ollama service with GPU support \
    echo '📡 Starting Ollama service with GPU support...' && \
    CUDA_VISIBLE_DEVICES=0 ollama serve & \
    OLLAMA_PID=\$! && \
    \
    # Wait for Ollama with enhanced error handling \
    echo '⏳ Waiting for Ollama to start...' && \
    for i in {1..60}; do \
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then \
            echo '✅ Ollama is ready!'; \
            break; \
        fi; \
        echo \"   Attempt \$i/60 - waiting 5 seconds...\"; \
        sleep 5; \
    done && \
    \
    # Verify Ollama started successfully \
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then \
        echo '❌ Failed to start Ollama service - checking logs'; \
        ps aux | grep ollama; \
        echo '⚠️ Continuing without Ollama...'; \
    fi && \
    \
    # Pull and warm up priority model (in background) \
    echo '📦 Pulling Mistral 7B in background...' && \
    (CUDA_VISIBLE_DEVICES=0 ollama pull mistral:7b-instruct-q4_0 && \
     echo '✅ Mistral 7B ready!' && \
     curl -X POST http://localhost:11434/api/chat \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"mistral:7b-instruct-q4_0\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"stream\": false, \"options\": {\"num_predict\": 5}}' \
        >/dev/null 2>&1 && echo '✅ Mistral warmed up!' \
    ) & \
    \
    # Verify dashboard status \
    if [ -f 'frontend/build/index.html' ]; then \
        echo '✅ Dashboard available at http://localhost:8001/app'; \
    else \
        echo 'ℹ️ Dashboard not available - API-only mode'; \
    fi && \
    \
    # Create .env if it doesn't exist \
    if [ ! -f '.env' ]; then \
        cp .env.template .env || echo 'DEBUG=false' > .env; \
    fi && \
    \
    # Display startup summary \
    echo '' && \
    echo '🎉 Consolidated Enhanced LLM Proxy Started!' && \
    echo '==========================================' && \
    echo '🌐 Main API: http://localhost:8001' && \
    echo '📊 API Documentation: http://localhost:8001/docs' && \
    echo '🏥 Health Check: http://localhost:8001/health' && \
    echo '📈 Metrics: http://localhost:8001/metrics' && \
    echo '🎛️ Dashboard: http://localhost:8001/app' && \
    echo '🔌 Ollama API: http://localhost:11434' && \
    echo '🎯 Status API: http://localhost:8001/api/status' && \
    echo '==========================================' && \
    echo '' && \
    \
    # Start the Consolidated FastAPI application \
    echo '🌐 Starting Consolidated FastAPI application...' && \
    python3 main_master.py \
"]
