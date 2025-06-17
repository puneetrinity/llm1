# Fixed LLM Proxy - No models, selective copy to avoid conflicts
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install enhanced features
RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    sse-starlette \
    redis \
    aioredis \
    prometheus-client \
    numpy \
    scikit-learn \
    || echo "Some enhanced features may be limited"

# Copy ONLY the files we need (avoid directory conflicts)
COPY main*.py ./
COPY services/ ./services/
COPY config*.py ./ 2>/dev/null || true

# Handle frontend - copy build or create minimal
COPY frontend/build/ ./static/ 2>/dev/null || \
    (mkdir -p ./static && \
     echo '<!DOCTYPE html><html><head><title>LLM Proxy</title></head><body><h1>🚀 LLM Proxy Ready</h1><p><a href="/docs">API Docs</a> | <a href="/health">Health</a></p></body></html>' > ./static/index.html)

# Create directories AFTER copying files
RUN mkdir -p \
    data/cache \
    data/logs \
    logs \
    cache \
    && chmod 755 data data/cache data/logs logs cache static

# Create basic .env
RUN cat > .env << 'EOF'
PORT=8001
HOST=0.0.0.0
DEBUG=false
LOG_LEVEL=INFO
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_STREAMING=true
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-key
MAX_MEMORY_MB=8192
EOF

# Create startup script
RUN cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LLM Proxy (no models pre-installed)"

# Start Ollama in background
ollama serve &
sleep 3

echo "✅ Ollama ready at http://localhost:11434"
echo "📥 Download models with: docker exec <container> ollama pull mistral:7b-instruct-q4_0"

# Start LLM proxy
echo "🌐 Starting LLM Proxy at http://localhost:8001"
exec python main.py
EOF

RUN chmod +x start.sh

# Expose ports
EXPOSE 8001 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["./start.sh"]
