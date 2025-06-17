# Space-Optimized Dockerfile for Enhanced 4-Model LLM Proxy
# Addresses disk space issues and reduces image size

# =============================================================================
# Stage 1: Frontend Build (Minimal)
# =============================================================================
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend/package*.json ./

# Install only production dependencies to save space
RUN npm ci --only=production --legacy-peer-deps --silent && \
    npm cache clean --force

# Copy frontend source
COPY frontend/ ./

# Build with production optimizations
ENV NODE_ENV=production
ENV GENERATE_SOURCEMAP=false
RUN npm run build && \
    ls -la build/ && \
    test -f build/index.html && \
    # Clean up node_modules to save space
    rm -rf node_modules

# =============================================================================
# Stage 2: Lightweight Production Image
# =============================================================================
FROM python:3.11-slim AS production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# 4-Model System Configuration
# =============================================================================
ENV DEFAULT_MODEL="mistral:7b-instruct-q4_0" \
    ENABLE_4_MODEL_ROUTING=true \
    PHI_MODEL="phi:3.5" \
    MISTRAL_MODEL="mistral:7b-instruct-q4_0" \
    GEMMA_MODEL="gemma:7b-instruct" \
    LLAMA_MODEL="llama3:8b-instruct-q4_0"

# =============================================================================
# Optimized Configuration (Reduced Memory Footprint)
# =============================================================================
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_NUM_PARALLEL=2 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    MAX_MEMORY_MB=8192 \
    CACHE_MEMORY_LIMIT_MB=1024 \
    MODEL_MEMORY_LIMIT_MB=4096

# =============================================================================
# Enhanced Features (Optimized)
# =============================================================================
ENV ENABLE_SEMANTIC_CLASSIFICATION=false \
    ENABLE_STREAMING=true \
    ENABLE_MODEL_WARMUP=true \
    ENABLE_DETAILED_METRICS=true \
    ENABLE_DASHBOARD=true \
    ENABLE_REACT_DASHBOARD=true

# =============================================================================
# Core Application Settings
# =============================================================================
ENV HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO \
    DEBUG=false

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Node.js (minimal version for runtime only)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm cache clean --force

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies with space optimization
RUN pip install --no-cache-dir -r requirements.txt && \
    # Install only essential enhanced features
    pip install --no-cache-dir \
    sse-starlette \
    redis \
    prometheus-client \
    || echo "Some features may be limited" && \
    # Clean up pip cache
    pip cache purge || true

# Copy built frontend (minimal)
COPY --from=frontend-build /app/frontend/build ./static
RUN ls -la ./static/ && test -f ./static/index.html

# Copy only essential application files
COPY main*.py ./
COPY config*.py ./
COPY services/ ./services/
COPY *.sh ./

# Create necessary directories
RUN mkdir -p data/{cache,logs,models} logs cache models && \
    chmod 755 data logs cache models && \
    find . -name "*.sh" -type f -exec chmod +x {} \; || true

# Create optimized startup script (without Ollama download)
RUN cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Space-Optimized 4-Model LLM Proxy"
echo "=============================================="

# Wait for external Ollama service
echo "📡 Waiting for Ollama service..."
for i in {1..30}; do
    if curl -f ${OLLAMA_BASE_URL:-http://ollama:11434}/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    fi
    echo "   Waiting for Ollama... ($i/30)"
    sleep 2
done

# Start the main application
echo "🌐 Starting FastAPI application..."
if [ -f "main_master.py" ]; then
    exec python main_master.py
elif [ -f "main.py" ]; then
    exec python main.py
else
    echo "❌ No main file found!"
    exit 1
fi
EOF

RUN chmod +x start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# Expose port
EXPOSE 8001

# Use non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["./start.sh"]

# Optimize image size
LABEL maintainer="LLM Proxy Team" \
      version="2.2.0-optimized" \
      description="Space-Optimized 4-Model LLM Proxy" \
      space-optimized="true"
