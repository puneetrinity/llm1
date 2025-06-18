FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install sentence-transformers faiss-cpu sse-starlette redis aioredis prometheus-client numpy scikit-learn

# Copy ALL application files (includes enhanced features)
COPY main*.py ./
COPY services/ ./services/
COPY config*.py ./
COPY middleware/ ./middleware/
COPY backups/ ./backups/

# Create basic HTML page
RUN mkdir -p static && \
    echo '<!DOCTYPE html>' > static/index.html && \
    echo '<html><head><title>LLM Proxy</title></head>' >> static/index.html && \
    echo '<body><h1>LLM Proxy API</h1>' >> static/index.html && \
    echo '<p><a href="/docs">API Documentation</a></p>' >> static/index.html && \
    echo '<p><a href="/health">Health Check</a></p>' >> static/index.html && \
    echo '<p><a href="/v1/models">Available Models</a></p>' >> static/index.html && \
    echo '</body></html>' >> static/index.html

# Create environment file with enhanced features enabled
RUN echo 'PORT=8001' > .env && \
    echo 'HOST=0.0.0.0' >> .env && \
    echo 'OLLAMA_BASE_URL=http://localhost:11434' >> .env && \
    echo 'ENABLE_STREAMING=true' >> .env && \
    echo 'ENABLE_SEMANTIC_CLASSIFICATION=true' >> .env && \
    echo 'ENABLE_MODEL_WARMUP=true' >> .env && \
    echo 'ENABLE_ENHANCED_ROUTING=true' >> .env && \
    echo 'ENABLE_SEMANTIC_CACHE=true' >> .env && \
    echo 'ENABLE_AUTH=false' >> .env

# Create startup script
RUN echo '#!/bin/bash' > start.sh && \
    echo 'echo "Starting Ollama..."' >> start.sh && \
    echo 'ollama serve &' >> start.sh && \
    echo 'sleep 5' >> start.sh && \
    echo 'echo "Starting LLM Proxy with Enhanced Features..."' >> start.sh && \
    echo 'python main.py' >> start.sh && \
    chmod +x start.sh

EXPOSE 8001 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["./start.sh"]
