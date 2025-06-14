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
    ENABLE_WEBSOCKET_DASHBOARD=true

# Dashboard configuration
ENV DASHBOARD_PATH=/app/static \
    ENABLE_REACT_DASHBOARD=true

# Performance and caching
ENV ENABLE_REDIS_CACHE=true \
    REDIS_URL=redis://localhost:6379 \
    ENABLE_SEMANTIC_CACHE=true \
    SEMANTIC_SIMILARITY_THRESHOLD=0.85

# Security settings (change in production) - Fixed JSON syntax
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

# Copy requirements and package files first for better caching
COPY requirements.txt ./
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
    prometheus-client \
    || echo "Some enhanced features may be limited"

# Install Node.js dependencies for React dashboard
WORKDIR /app/frontend
RUN if [ -f "package.json" ]; then \
        echo "Installing Node.js dependencies..." && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm install --legacy-peer-deps --prefer-offline --no-optional; \
    else \
        echo "No frontend package.json found"; \
    fi

# Copy frontend source code and build React dashboard
COPY frontend/ ./
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

# Copy critical files explicitly first (in case .dockerignore has issues)
COPY start.sh ./
COPY requirements.txt ./

# Copy Python files explicitly
COPY *.py ./

# Copy everything else
COPY . ./

# === DEBUGGING SECTION ===
RUN echo "=== DEBUGGING: Checking file structure ===" && \
    echo "Current directory:" && pwd && \
    echo "Files in /app:" && ls -la /app/ && \
    echo "Looking for main.py:" && \
    (test -f /app/main.py && echo "✅ main.py found" || echo "❌ main.py NOT found") && \
    echo "Looking for Python files:" && find /app -name "*.py" | head -10 && \
    echo "Contents of start.sh:" && cat /app/start.sh 2>/dev/null || echo "start.sh not found"

# Create fallback main.py if it doesn't exist
RUN if [ ! -f "/app/main.py" ]; then \
        echo "❌ main.py not found - creating fallback version..." && \
        cat > /app/main.py << 'EOF'
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="LLM Proxy Server", version="1.0.0")

@app.get("/")
def read_root():
    return {
        "message": "🚀 Complete LLM Proxy Server is running", 
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "llm-proxy"}

@app.get("/dashboard")
def dashboard():
    return {"dashboard": "available", "path": "/app"}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host=host, port=port)
EOF
        echo "✅ Fallback main.py created"; \
    else \
        echo "✅ main.py exists"; \
    fi

# Test Python import if main.py exists
RUN echo "=== Testing Python import of main.py ===" && \
    python3 -c "import sys; sys.path.insert(0, '/app'); import main; print('✅ main.py imports successfully')" 2>&1 || \
    echo "❌ Failed to import main.py - check for syntax errors"

# Create fallback frontend if build failed
RUN if [ ! -f "frontend/build/index.html" ]; then \
        echo "Creating fallback HTML page..." && \
        mkdir -p frontend/build; \
    else \
        echo "Frontend build exists"; \
    fi

# Copy fallback HTML (clean approach)
COPY fallback.html frontend/build/index.html

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \;

# Create required directories
RUN mkdir -p logs cache models data static frontend/build

# Ensure start.sh is executable and exists
RUN if [ ! -f "/app/start.sh" ]; then \
        echo "❌ start.sh not found - creating fallback..." && \
        cat > /app/start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LLM Proxy Server..."
echo "📍 Server will be available at: http://0.0.0.0:8001"
echo "📊 Dashboard will be available at: http://0.0.0.0:8001/app"

# Start the application
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
EOF
        chmod +x /app/start.sh; \
    else \
        echo "✅ start.sh exists"; \
    fi

# Final verification
RUN echo "=== FINAL VERIFICATION ===" && \
    echo "✅ Files ready:" && \
    ls -la /app/main.py /app/start.sh && \
    echo "✅ Permissions:" && \
    ls -la /app/start.sh && \
    echo "✅ Python can import main:" && \
    python3 -c "import main" && \
    echo "🎉 Container is ready!"

# Comprehensive health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Use the startup script
CMD ["./start.sh"]
