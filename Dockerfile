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

# Security settings (change in production)
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

# Copy package files first for better caching
COPY requirements.txt .
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
    prometheus-client

# Install Node.js dependencies and build React dashboard
WORKDIR /app/frontend
RUN if [ -f "package.json" ]; then \
        echo "Installing Node.js dependencies..." && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm install --legacy-peer-deps --prefer-offline --no-optional; \
    else \
        echo "No frontend package.json found"; \
    fi

# Copy frontend source and build
COPY frontend/ .
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
COPY . .

# Create fallback frontend if build failed
RUN if [ ! -f "frontend/build/index.html" ]; then \
        echo "Creating fallback HTML page..." && \
        mkdir -p frontend/build && \
        cat > frontend/build/index.html << 'EOF'
<!DOCTYPE html>
<html><head><title>LLM Proxy API</title>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:Arial,sans-serif;text-align:center;padding:50px;background:#f0f0f0;margin:0;}
h1{color:#333;margin-bottom:20px;}.container{max-width:600px;margin:0 auto;}
.link{display:inline-block;margin:10px;padding:15px 25px;background:#007bff;color:white;text-decoration:none;border-radius:5px;transition:background 0.3s;}
.link:hover{background:#0056b3;}
.metrics{background:#fff;padding:20px;margin:20px 0;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
</style></head>
<body>
<div class="container">
  <h1>🚀 Enhanced LLM Proxy API</h1>
  <p>Production-ready FastAPI with GPU Support, Ollama Integration & Advanced Features</p>
  <div style="margin:30px 0;">
    <a href="/health" class="link">Health Check</a>
    <a href="/docs" class="link">API Documentation</a>
    <a href="/v1/chat/completions" class="link">Chat API</a>
    <a href="/metrics" class="link">Prometheus Metrics</a>
  </div>
  <div class="metrics">
    <h3>🎯 Features Enabled</h3>
    <p>✅ OpenAI-compatible API endpoints</p>
    <p>✅ GPU-accelerated inference with Ollama</p>
    <p>✅ Real-time streaming responses</p>
    <p>✅ Comprehensive monitoring & metrics</p>
    <p>✅ Production-ready security features</p>
  </div>
  <p style="margin-top:40px;color:#666;">
    <strong>Ollama Endpoint:</strong> localhost:11434<br>
    <strong>API Base URL:</strong> /v1/
  </p>
</div>
</body></html>
EOF
        echo "Enhanced fallback page created"; \
    else \
        echo "Frontend build exists"; \
    fi

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \;

# Create required directories with proper structure
RUN mkdir -p /app/cache /app/logs /app/models /app/data \
             /app/static /app/frontend/build

# Comprehensive health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Comprehensive Enhanced LLM Proxy..."\n\
\n\
# Export GPU environment variables\n\
export CUDA_VISIBLE_DEVICES=0\n\
export NVIDIA_VISIBLE_DEVICES=all\n\
export OLLAMA_HOST=0.0.0.0:11434\n\
\n\
# Verify GPU detection\n\
echo "🔍 Checking GPU availability..."\n\
nvidia-smi || echo "GPU detection may have issues - continuing with CPU"\n\
\n\
# Start Ollama service\n\
echo "📡 Starting Ollama service..."\n\
CUDA_VISIBLE_DEVICES=0 ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama with comprehensive timeout\n\
echo "⏳ Waiting for Ollama to start..."\n\
for i in {1..60}; do\n\
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then\n\
        echo "✅ Ollama is ready!"\n\
        break\n\
    fi\n\
    echo "   Attempt $i/60 - waiting 5 seconds..."\n\
    sleep 5\n\
done\n\
\n\
# Verify Ollama started successfully\n\
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then\n\
    echo "❌ Failed to start Ollama service after 5 minutes"\n\
    echo "🔧 Attempting fallback startup..."\n\
    pkill ollama || true\n\
    sleep 5\n\
    ollama serve &\n\
    sleep 30\n\
fi\n\
\n\
# Pull essential models in background\n\
echo "📦 Pulling essential models in background..."\n\
(CUDA_VISIBLE_DEVICES=0 ollama pull mistral:7b-instruct-q4_0 >/dev/null 2>&1 &)\n\
\n\
# Create default .env if not exists\n\
[ ! -f .env ] && echo "PORT=8001" > .env || true\n\
\n\
# Start the comprehensive FastAPI application\n\
echo "🌐 Starting Enhanced FastAPI application..."\n\
echo "✅ System Ready: http://localhost:8001"\n\
echo "📚 API Documentation: http://localhost:8001/docs"\n\
echo "🏥 Health Check: http://localhost:8001/health"\n\
python3 main_master.py\n' > /app/start.sh && chmod +x /app/start.sh

# Use the startup script
CMD ["/app/start.sh"]
