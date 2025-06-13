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

# Install Node.js 18.x
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir \
        sentence-transformers==2.2.2 \
        faiss-cpu==1.7.4 \
        sse-starlette==1.6.5 \
        redis \
        aioredis \
        prometheus-client

# Copy all app code
COPY . .

# Create frontend directory
RUN mkdir -p frontend/build

# Build React frontend if source exists, otherwise create fallback
RUN if [ -f "frontend/package.json" ] && [ -d "frontend/src" ]; then \
        echo "📦 Building React frontend..." && \
        cd frontend && \
        npm install --legacy-peer-deps && \
        CI=true npm run build && \
        echo "✅ React build completed."; \
    else \
        echo "⚠️ React frontend not found, creating fallback page..."; \
    fi

# Create fallback HTML page if React build doesn't exist
RUN if [ ! -f "frontend/build/index.html" ]; then \
        echo '<!DOCTYPE html>' > frontend/build/index.html && \
        echo '<html><head><title>LLM Proxy API</title>' >> frontend/build/index.html && \
        echo '<style>body{font-family:Arial;text-align:center;padding:50px;background:#f0f0f0;}' >> frontend/build/index.html && \
        echo 'h1{color:#333;}.link{display:block;margin:10px;padding:15px;background:#007bff;' >> frontend/build/index.html && \
        echo 'color:white;text-decoration:none;border-radius:5px;}</style></head>' >> frontend/build/index.html && \
        echo '<body><h1>🚀 LLM Proxy API</h1><p>Enhanced FastAPI with GPU Support</p>' >> frontend/build/index.html && \
        echo '<a href="/health" class="link">Health Check</a>' >> frontend/build/index.html && \
        echo '<a href="/docs" class="link">API Documentation</a>' >> frontend/build/index.html && \
        echo '<a href="/api/status" class="link">Status API</a>' >> frontend/build/index.html && \
        echo '<a href="/metrics" class="link">Metrics</a>' >> frontend/build/index.html && \
        echo '<p><strong>Ollama:</strong> localhost:11434</p></body></html>' >> frontend/build/index.html; \
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
