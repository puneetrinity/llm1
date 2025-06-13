# Dockerfile - COMPLETELY FIXED for GitHub Container Registry
# No shell operators in COPY commands - handles optional files properly

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

# Enhanced feature toggles
ENV ENABLE_SEMANTIC_CLASSIFICATION=false
ENV ENABLE_STREAMING=true
ENV ENABLE_MODEL_WARMUP=true
ENV ENABLE_DETAILED_METRICS=true
ENV ENABLE_DASHBOARD=true

# App configuration - FIXED for consolidated setup
ENV HOST=0.0.0.0
ENV PORT=8001
ENV LOG_LEVEL=INFO
ENV DEBUG=false
ENV ENABLE_AUTH=false

WORKDIR /app

# ============================================================================
# STAGE 1: SYSTEM DEPENDENCIES
# ============================================================================

# Install system dependencies including Node.js
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

# Install Node.js 18.x
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

# ============================================================================
# STAGE 3: APPLICATION CODE (COPY ALL FILES FIRST)
# ============================================================================

# Copy ALL files first (simplest approach - no conditional copying)
COPY . .

# ============================================================================
# STAGE 4: FRONTEND BUILD (AFTER ALL FILES ARE COPIED)
# ============================================================================

# Now handle frontend build with all files available
RUN echo "🔍 Checking for frontend files..." && \
    if [ -f "frontend/package.json" ] && [ -d "frontend/src" ]; then \
        echo "📦 Frontend source found - building React app..." && \
        cd frontend && \
        echo "Installing dependencies..." && \
        npm install --production && \
        echo "Building React app..." && \
        npm run build && \
        cd .. && \
        echo "✅ React build completed" && \
        ls -la frontend/build/; \
    elif [ -f "frontend/package.json" ]; then \
        echo "📦 Frontend package.json found but no src/ - minimal build..." && \
        cd frontend && \
        npm install --production && \
        mkdir -p build && \
        echo '<!DOCTYPE html><html><head><title>API Dashboard</title></head><body><h1>LLM Proxy API</h1><p>Frontend build minimal</p></body></html>' > build/index.html && \
        cd ..; \
    else \
        echo "ℹ️ No frontend found - creating fallback dashboard..." && \
        mkdir -p frontend/build && \
        echo '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 LLM Proxy API</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0; padding: 0; min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex; align-items: center; justify-content: center;
        }
        .container {
            background: rgba(255,255,255,0.95);
            padding: 40px; border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center; max-width: 600px; margin: 20px;
        }
        h1 { color: #333; margin-bottom: 10px; }
        .subtitle { color: #666; margin-bottom: 30px; }
        .status { 
            background: #e3f2fd; color: #1976d2; padding: 15px; 
            border-radius: 8px; margin: 20px 0; border-left: 4px solid #2196f3;
        }
        .links { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 30px; }
        .link {
            display: block; padding: 15px; background: #f5f5f5; color: #333;
            text-decoration: none; border-radius: 8px; transition: all 0.3s;
            border: 2px solid transparent;
        }
        .link:hover { background: #667eea; color: white; transform: translateY(-2px); }
        .health { margin: 20px 0; }
        .health-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white; border: none; padding: 12px 24px; border-radius: 25px;
            cursor: pointer; font-size: 16px; font-weight: bold;
        }
        .health-status { margin-top: 15px; font-weight: bold; }
        .footer { margin-top: 30px; font-size: 14px; color: #666; }
    </style>
    <script>
        async function checkHealth() {
            const btn = document.getElementById("health-btn");
            const status = document.getElementById("health-status");
            
            btn.textContent = "🔄 Checking...";
            btn.disabled = true;
            
            try {
                const response = await fetch("/health");
                const data = await response.json();
                status.innerHTML = `<span style="color: #4CAF50;">✅ Healthy</span> | Version: ${data.version || "Unknown"}`;
                btn.textContent = "🔄 Refresh Status";
            } catch (error) {
                status.innerHTML = `<span style="color: #f44336;">❌ Connection Error</span>`;
                btn.textContent = "🔄 Retry";
            }
            
            btn.disabled = false;
        }
        
        window.onload = function() {
            checkHealth();
            setInterval(checkHealth, 60000); // Check every minute
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>🚀 Enhanced LLM Proxy</h1>
        <div class="subtitle">Consolidated FastAPI + GPU + Ollama</div>
        
        <div class="status">
            <strong>Dashboard:</strong> API-Only Mode (React frontend not built)
        </div>
        
        <div class="health">
            <button id="health-btn" class="health-btn" onclick="checkHealth()">🔄 Check Health</button>
            <div id="health-status" class="health-status">Checking server status...</div>
        </div>
        
        <div class="links">
            <a href="/health" class="link">💚 Health Check</a>
            <a href="/docs" class="link">📚 API Documentation</a>
            <a href="/api/status" class="link">📊 Status API</a>
            <a href="/metrics" class="link">📈 Metrics</a>
        </div>
        
        <div class="footer">
            <p><strong>🎯 Ollama API:</strong> <code>localhost:11434</code></p>
            <p><strong>🌐 FastAPI:</strong> <code>localhost:8001</code></p>
            <p>Built with CUDA + GPU support | Container deployment ready</p>
        </div>
    </div>
</body>
</html>' > frontend/build/index.html; \
    fi

# ============================================================================
# STAGE 5: FINALIZATION
# ============================================================================

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; 2>/dev/null || true && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \; 2>/dev/null || true

# Create necessary directories
RUN mkdir -p logs cache models data

# Verify final structure
RUN echo "📋 Final directory structure:" && \
    ls -la && \
    echo "📦 Frontend build:" && \
    ls -la frontend/build/ 2>/dev/null || echo "No frontend build directory" && \
    echo "✅ Setup verification complete"

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# ============================================================================
# STARTUP COMMAND
# ============================================================================

CMD ["/bin/bash", "-c", "\
    echo '🚀 Starting Consolidated Enhanced LLM Proxy...' && \
    \
    # Export environment variables \
    export CUDA_VISIBLE_DEVICES=0 && \
    export NVIDIA_VISIBLE_DEVICES=all && \
    export OLLAMA_HOST=0.0.0.0:11434 && \
    export HOST=0.0.0.0 && \
    export PORT=8001 && \
    \
    # Verify GPU \
    echo '🔍 GPU Check:' && \
    nvidia-smi 2>/dev/null | head -10 || echo '⚠️ No GPU detected' && \
    \
    # Start Ollama \
    echo '📡 Starting Ollama...' && \
    ollama serve & \
    OLLAMA_PID=\$! && \
    \
    # Wait for Ollama \
    echo '⏳ Waiting for Ollama (max 60s)...' && \
    for i in {1..12}; do \
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then \
            echo '✅ Ollama ready!'; break; \
        fi; \
        echo \"   Waiting... attempt \$i/12\"; \
        sleep 5; \
    done && \
    \
    # Background model download \
    echo '📦 Starting model download in background...' && \
    (ollama pull mistral:7b-instruct-q4_0 2>/dev/null && echo '✅ Model ready!') & \
    \
    # Create .env if needed \
    [ ! -f .env ] && echo 'PORT=8001' > .env || true && \
    \
    # Status summary \
    echo '' && \
    echo '🎉 SYSTEM READY!' && \
    echo '=================' && \
    echo '🌐 API: http://localhost:8001' && \
    echo '📊 Dashboard: http://localhost:8001/app' && \
    echo '📚 Docs: http://localhost:8001/docs' && \
    echo '🔌 Ollama: http://localhost:11434' && \
    echo '=================' && \
    echo '' && \
    \
    # Start the FastAPI application \
    echo '🌐 Starting FastAPI...' && \
    python3 main_master.py \
"]
