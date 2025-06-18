FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl gcc g++ python3-dev net-tools procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install sentence-transformers faiss-cpu sse-starlette redis aioredis prometheus-client numpy scikit-learn

# Copy application files
COPY main*.py ./
COPY services/ ./services/
COPY config*.py ./
COPY middleware/ ./middleware/
COPY backups/ ./backups/
COPY start.sh ./
COPY enhanced_setup.sh ./
COPY start_dev.sh ./
COPY download_models.sh ./
COPY test_system.sh ./
COPY frontend/ ./frontend/

# Create directories
RUN mkdir -p data/cache data/logs data/models logs cache models static tests scripts backups frontend/build /root/.ollama

# Create simple HTML
RUN echo '<!DOCTYPE html><html><head><title>Enhanced 4-Model LLM Proxy</title></head><body><h1>🚀 Enhanced 4-Model LLM Proxy</h1><p><strong>First setup:</strong> Run <code>docker exec &lt;container&gt; ./download_models.sh</code></p><p><a href="/docs">API Docs</a> | <a href="/health">Health</a> | <a href="/v1/models">Models</a></p></body></html>' > static/index.html

# Create environment file
RUN echo 'ENVIRONMENT=production' > .env && \
    echo 'HOST=0.0.0.0' >> .env && \
    echo 'PORT=8001' >> .env && \
    echo 'OLLAMA_BASE_URL=http://localhost:11434' >> .env && \
    echo 'ENABLE_4_MODEL_ROUTING=true' >> .env && \
    echo 'ENABLE_SEMANTIC_CLASSIFICATION=true' >> .env && \
    echo 'ENABLE_STREAMING=true' >> .env && \
    echo 'ENABLE_MODEL_WARMUP=true' >> .env && \
    echo 'ENABLE_DASHBOARD=true' >> .env && \
    echo 'AUTO_DOWNLOAD_MODELS=false' >> .env

# Make scripts executable
RUN chmod +x *.sh

EXPOSE 8001 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["./setup.sh"]
