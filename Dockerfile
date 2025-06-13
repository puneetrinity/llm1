# Base image with CUDA
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_NUM_PARALLEL=2 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    OLLAMA_GPU_OVERHEAD=0 \
    HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO \
    DEBUG=false \
    ENABLE_AUTH=false \
    ENABLE_DASHBOARD=true \
    MAX_MEMORY_MB=12288 \
    CACHE_MEMORY_LIMIT_MB=1024

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

# Install Node.js 18.x FIRST
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install Python requirements with CURRENT versions
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir \
        sentence-transformers>=3.0.0 \
        faiss-cpu>=1.8.0 \
        sse-starlette>=1.6.5 \
        redis \
        aioredis \
        prometheus-client

# Copy all app code
COPY . .

# Create frontend directory
RUN mkdir -p frontend/build

# Build React frontend (simple approach)
RUN set +e && \
    if [ -f "frontend/package.json" ] && [ -d "frontend/src" ]; then \
        echo "Building React frontend..." && \
        cd frontend && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm install --legacy-peer-deps --silent && \
        GENERATE_SOURCEMAP=false CI=true npm run build && \
        echo "React build completed successfully!" || echo "React build failed"; \
        cd ..; \
    else \
        echo "No React frontend found"; \
    fi && \
    set -e

# Create fallback HTML if build failed
RUN if [ ! -f "frontend/build/index.html" ]; then \
        echo "Creating fallback HTML page..." && \
        echo '<!DOCTYPE html>' > frontend/build/index.html && \
        echo '<html><head><title>LLM Proxy API</title>' >> frontend/build/index.html && \
        echo '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' >> frontend/build/index.html && \
        echo '<style>body{font-family:Arial,sans-serif;text-align:center;padding:50px;background:#f0f0f0;margin:0;}' >> frontend/build/index.html && \
        echo 'h1{color:#333;margin-bottom:20px;}.container{max-width:600px;margin:0 auto;}' >> frontend/build/index.html && \
        echo '.link{display:inline-block;margin:10px;padding:15px 25px;background:#007bff;' >> frontend/build/index.html && \
        echo 'color:white;text-decoration:none;border-radius:5px;transition:background 0.3s;}' >> frontend/build/index.html && \
        echo '.link:hover{background:#0056b3;}</style></head>' >> frontend/build/index.html && \
        echo '<body><div class="container"><h1>🚀 LLM Proxy API</h1>' >> frontend/build/index.html && \
        echo '<p>Enhanced FastAPI with GPU Support & Ollama Integration</p>' >> frontend/build/index.html && \
        echo '<div style="margin:30px 0;"><a href="/health" class="link">Health Check</a>' >> frontend/build/index.html && \
        echo '<a href="/docs" class="link">API Docs</a><a href="/api/status" class="link">Status</a>' >> frontend/build/index.html && \
        echo '<a href="/metrics" class="link">Metrics</a></div>' >> frontend/build/index.html && \
        echo '<p style="margin-top:40px;color:#666;"><strong>Ollama Endpoint:</strong> localhost:11434</p>' >> frontend/build/index.html && \
        echo '</div></body></html>' >> frontend/build/index.html && \
        echo "Fallback page created"; \
    else \
        echo "Frontend build exists"; \
    fi

# Fix permissions and convert line endings
RUN find . -name "*.py" -exec dos2unix {} \; 2>/dev/null || true && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \; 2>/dev/null || true

# Create necessary directories
RUN mkdir -p logs cache models data

# Healthcheck
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001 11434

# Startup command
CMD ["/bin/bash", "-c", "\
    echo '🚀 Starting LLM Proxy...' && \
    echo '🧠 Starting Ollama service...' && \
    ollama serve & \
    echo '⏳ Waiting for Ollama to be ready...' && \
    for i in {1..12}; do \
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then \
            echo '✅ Ollama is ready!'; break; \
        fi; \
        sleep 5; \
    done && \
    echo '⬇️ Pulling default model (mistral:7b-instruct-q4_0)...' && \
    (ollama pull mistral:7b-instruct-q4_0 2>/dev/null || true) & \
    [ ! -f .env ] && echo 'PORT=8001' > .env || true && \
    echo '✅ System Ready: http://localhost:8001' && \
    python3 main_master.py \
"]
