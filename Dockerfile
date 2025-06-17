# Clean LLM Proxy Dockerfile - No models, no syntax errors
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

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
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
    scikit-learn

# Copy application files
COPY main*.py ./
COPY services/ ./services/
COPY config*.py ./

# Create static directory and handle frontend
RUN mkdir -p ./static
COPY frontend/build/ ./static/
RUN if [ ! -f "./static/index.html" ]; then \
        echo '<!DOCTYPE html>' > ./static/index.html && \
        echo '<html><head><title>LLM Proxy</title></head>' >> ./static/index.html && \
        echo '<body><h1>🚀 LLM Proxy Ready</h1>' >> ./static/index.html && \
        echo '<p><a href="/docs">API Docs</a> | <a href="/health">Health</a></p>' >> ./static/index.html && \
        echo '</body></html>' >> ./static/index.html; \
    fi

# Create directories
RUN mkdir -p data/cache data/logs logs cache && \
    chmod 755 data data/cache data/logs logs cache static

# Create .env configuration
RUN echo 'PORT=8001' > .env && \
    echo 'HOST=0.0.0.0' >> .env && \
    echo 'DEBUG=false' >> .env && \
    echo 'LOG_LEVEL=INFO' >> .env && \
    echo 'OLLAMA_BASE_URL=http://localhost:11434' >> .env && \
    echo 'ENABLE_STREAMING=true' >> .env && \
    echo 'ENABLE_SEMANTIC_CLASSIFICATION=true' >> .env && \
    echo 'ENABLE_AUTH=false' >> .env && \
    echo 'DEFAULT_API_KEY=sk-dev-key' >> .env && \
    echo 'MAX_MEMORY_MB=8192' >> .env

# Create startup script
RUN echo '#!/bin/bash' > start.sh && \
    echo 'echo "🚀 Starting LLM Proxy (models not included)"' >> start.sh && \
    echo 'ollama serve &' >> start.sh && \
    echo 'sleep 3' >> start.sh && \
    echo 'echo "✅ Ollama ready at http://localhost:11434"' >> start.sh && \
    echo 'echo "📥 Download models: docker exec <container> ollama pull mistral:7b-instruct-q4_0"' >> start.sh && \
    echo 'echo "🌐 Starting LLM Proxy at http://localhost:8001"' >> start.sh && \
    echo 'exec python main.py' >> start.sh && \
    chmod +x start.sh

# Expose ports
EXPOSE 8001 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start
CMD ["./start.sh"]
