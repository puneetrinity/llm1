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

# Install system dependencies (optimized single layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    dos2unix \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Node.js 20 LTS (reliable method)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && npm --version

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install Python requirements with CURRENT versions
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir \
        "sentence-transformers>=3.1.0,<4.0.0" \
        "faiss-cpu>=1.11.0" \
        "sse-starlette>=2.1.0" \
        "redis>=5.0.0" \
        "aioredis>=2.0.0" \
        "prometheus-client>=0.20.0" \
        "numpy>=1.24.0,<2.0.0" \
        "scikit-learn>=1.4.0"

# Copy package.json first (better caching)
COPY frontend/package*.json ./frontend/

# Create vite.config.js (no heredoc issues)
RUN mkdir -p frontend
COPY <<EOF frontend/vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/app/',
  build: {
    outDir: 'build',
    sourcemap: false,
    minify: 'esbuild',
    target: 'es2020'
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8001',
      '/health': 'http://localhost:8001',
      '/docs': 'http://localhost:8001'
    }
  }
})
EOF

# Install frontend dependencies
RUN cd frontend \
    && if [ -f "package.json" ]; then \
        npm ci --omit=dev --silent; \
    fi \
    && cd ..

# Copy all application code
COPY . .

# Build React frontend with VITE
RUN cd frontend \
    && if [ -f "package.json" ] && [ -d "src" ]; then \
        echo "Building with Vite..." \
        && npm run build \
        && echo "Vite build completed!" \
        && if [ -d "dist" ] && [ ! -d "build" ]; then \
            mv dist build; \
        fi; \
    else \
        echo "No frontend source found" \
        && mkdir -p build; \
    fi \
    && cd ..

# Create fallback HTML (no heredoc issues)
RUN mkdir -p frontend/build
COPY <<EOF frontend/build/index.html
<!DOCTYPE html>
<html>
<head>
    <title>LLM Proxy API</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
        body{font-family:Arial,sans-serif;text-align:center;padding:50px;background:#f0f0f0;margin:0;}
        h1{color:#333;margin-bottom:20px;}
        .container{max-width:600px;margin:0 auto;}
        .link{display:inline-block;margin:10px;padding:15px 25px;background:#007bff;color:white;text-decoration:none;border-radius:5px;transition:background 0.3s;}
        .link:hover{background:#0056b3;}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 LLM Proxy API</h1>
        <p>Enhanced FastAPI with GPU Support & Ollama Integration</p>
        <div style="margin:30px 0;">
            <a href="/health" class="link">Health Check</a>
            <a href="/docs" class="link">API Docs</a>
            <a href="/api/status" class="link">Status</a>
            <a href="/metrics" class="link">Metrics</a>
        </div>
        <p style="margin-top:40px;color:#666;">
            <strong>Ollama Endpoint:</strong> localhost:11434
        </p>
    </div>
</body>
</html>
EOF

# Fix permissions and line endings
RUN find . -name "*.py" -exec dos2unix {} \; 2>/dev/null || true
RUN find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \; 2>/dev/null || true

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
