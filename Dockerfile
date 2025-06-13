# Dockerfile - SIMPLIFIED to avoid parsing errors
# CUDA + Ollama + FastAPI with minimal dashboard

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ollama configuration
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_NUM_PARALLEL=2
ENV OLLAMA_MAX_LOADED_MODELS=2
ENV OLLAMA_GPU_OVERHEAD=0

# App configuration
ENV HOST=0.0.0.0
ENV PORT=8001
ENV LOG_LEVEL=INFO
ENV DEBUG=false
ENV ENABLE_AUTH=false
ENV ENABLE_DASHBOARD=true

# Memory management
ENV MAX_MEMORY_MB=12288
ENV CACHE_MEMORY_LIMIT_MB=1024

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
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18.x
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install enhanced dependencies with fallbacks
RUN pip3 install --no-cache-dir \
    "sentence-transformers>=2.2.0,<3.0.0" \
    "faiss-cpu==1.7.4" \
    "sse-starlette==1.6.5" \
    || echo "Some ML features may be limited"

RUN pip3 install --no-cache-dir \
    "redis>=4.5.0" \
    "aioredis>=2.0.0" \
    "prometheus-client>=0.19.0" \
    || echo "Some performance features may be limited"

# Copy all application files
COPY . .

# Handle frontend build (simplified approach)
RUN mkdir -p frontend/build

# Try to build React frontend if it exists
RUN if [ -f "frontend/package.json" ] && [ -d "frontend/src" ]; then \
        echo "Building React frontend..." && \
        cd frontend && \
        npm install --legacy-peer-deps && \
        CI=true npm run build && \
        cd .. && \
        echo "React build completed"; \
    else \
        echo "Creating simple fallback dashboard..." && \
        echo '<html><head><title>LLM Proxy API</title><style>body{font-family:Arial;text-align:center;padding:50px;background:#f0f0f0}h1{color:#333}.link{display:block;margin:10px;padding:15px;background:#007bff;color:white;text-decoration:none;border-radius:5px}</style></head><body><h1>🚀 LLM Proxy API</h1><p>Enhanced FastAPI with GPU Support</p><a href="/health" class="link">Health Check</a><a href="/docs" class="link">API Documentation</a><a href="/api/status" class="link">Status API</a><a href="/metrics" class="link">Metrics</a><p><strong>Ollama:</strong> localhost:11434</p></body></html>' > frontend/build/index.html; \
    fi

# Fix permissions
RUN find . -name "*.py" -exec dos2unix {} \; 2>/dev/null || true
RUN find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \; 2>/dev/null || true

# Create directories
RUN mkdir -p logs cache models data

# Verify setup
RUN echo "Setup verification:" && \
    ls -la && \
    echo "Frontend build:" && \
    ls -la frontend/build/ 2>/dev/null || echo "No frontend build"

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Simple startup command
CMD ["/bin/bash", "-c", "\
    echo '🚀 Starting LLM Proxy...' && \
    \
    # Start Ollama \
    echo 'Starting Ollama service...' && \
    ollama serve & \
    \
    # Wait for Ollama \
    echo 'Waiting for Ollama...' && \
    for i in {1..12}; do \
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then \
            echo 'Ollama ready!'; break; \
        fi; \
        sleep 5; \
    done && \
    \
    # Download model in background \
    (ollama pull mistral:7b-instruct-q4_0 2>/dev/null) & \
    \
    # Create .env if needed \
    [ ! -f .env ] && echo 'PORT=8001' > .env || true && \
    \
    # Show access info \
    echo '' && \
    echo 'System Ready!' && \
    echo 'API: http://localhost:8001' && \
    echo 'Dashboard: http://localhost:8001/app' && \
    echo 'Docs: http://localhost:8001/docs' && \
    echo '' && \
    \
    # Start FastAPI \
    python3 main_master.py \
"]
