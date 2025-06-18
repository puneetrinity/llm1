FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    python3-dev \
    net-tools \
    procps \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install sentence-transformers faiss-cpu sse-starlette redis aioredis prometheus-client numpy scikit-learn

# Copy ALL application files (based on your actual project structure)
COPY main*.py ./
COPY config*.py ./
COPY services/ ./services/
COPY middleware/ ./middleware/
COPY backups/ ./backups/
COPY frontend/ ./frontend/

# Copy all shell scripts
COPY *.sh ./

# Build frontend
RUN if [ -f "frontend/package.json" ]; then \
        cd frontend && \
        npm install --legacy-peer-deps && \
        npm run build && \
        mkdir -p /app/static && \
        cp -r build/* /app/static/ || true; \
    fi

# Create static directory with fallback content if frontend build failed
RUN mkdir -p static && \
    if [ ! -f "static/index.html" ]; then \
        echo '<!DOCTYPE html>' > static/index.html && \
        echo '<html><head><title>Enhanced 4-Model LLM Proxy</title>' >> static/index.html && \
        echo '<style>body{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5}.container{max-width:800px;margin:0 auto;background:white;padding:30px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1)}h1{color:#333;text-align:center}.models{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:20px;margin:30px 0}.model{padding:20px;background:#f8f9fa;border-radius:8px;text-align:center;border-left:4px solid #007bff}.links{display:flex;gap:20px;justify-content:center;flex-wrap:wrap}.links a{padding:10px 20px;background:#007bff;color:white;text-decoration:none;border-radius:5px}.notice{background:#fff3cd;border:1px solid #ffeaa7;color:#856404;padding:15px;border-radius:5px;margin:20px 0}</style>' >> static/index.html && \
        echo '</head><body><div class="container">' >> static/index.html && \
        echo '<h1>🚀 Enhanced 4-Model LLM Proxy</h1>' >> static/index.html && \
        echo '<div class="notice"><strong>📥 Setup Required:</strong> Download models with: <code>docker exec container ollama pull mistral:7b-instruct-q4_0</code></div>' >> static/index.html && \
        echo '<div class="models">' >> static/index.html && \
        echo '<div class="model"><h3>🧠 Phi-3.5</h3><p>Math, Logic, Reasoning</p></div>' >> static/index.html && \
        echo '<div class="model"><h3>⚡ Mistral 7B</h3><p>General, Quick Facts</p></div>' >> static/index.html && \
        echo '<div class="model"><h3>⚙️ Gemma 7B</h3><p>Technical, Coding</p></div>' >> static/index.html && \
        echo '<div class="model"><h3>🎨 Llama3 8B</h3><p>Creative, Writing</p></div>' >> static/index.html && \
        echo '</div>' >> static/index.html && \
        echo '<div class="links">' >> static/index.html && \
        echo '<a href="/docs">📚 API Documentation</a>' >> static/index.html && \
        echo '<a href="/health">🏥 Health Check</a>' >> static/index.html && \
        echo '<a href="/v1/models">📋 Available Models</a>' >> static/index.html && \
        echo '<a href="/metrics">📊 Metrics</a>' >> static/index.html && \
        echo '</div></div></body></html>' >> static/index.html; \
    fi

# Create directories
RUN mkdir -p data/cache data/logs data/models logs cache models tests scripts /root/.ollama

# Create enhanced environment configuration
RUN echo 'ENVIRONMENT=production' > .env && \
    echo 'HOST=0.0.0.0' >> .env && \
    echo 'PORT=8001' >> .env && \
    echo 'LOG_LEVEL=INFO' >> .env && \
    echo 'DEBUG=false' >> .env && \
    echo 'OLLAMA_BASE_URL=http://localhost:11434' >> .env && \
    echo 'OLLAMA_HOST=0.0.0.0:11434' >> .env && \
    echo 'DEFAULT_MODEL=mistral:7b-instruct-q4_0' >> .env && \
    echo 'ENABLE_4_MODEL_ROUTING=true' >> .env && \
    echo 'PHI_MODEL=phi:3.5' >> .env && \
    echo 'MISTRAL_MODEL=mistral:7b-instruct-q4_0' >> .env && \
    echo 'GEMMA_MODEL=gemma:7b-instruct' >> .env && \
    echo 'LLAMA_MODEL=llama3:8b-instruct-q4_0' >> .env && \
    echo 'ENABLE_SEMANTIC_CLASSIFICATION=true' >> .env && \
    echo 'ENABLE_STREAMING=true' >> .env && \
    echo 'ENABLE_MODEL_WARMUP=true' >> .env && \
    echo 'ENABLE_DETAILED_METRICS=true' >> .env && \
    echo 'ENABLE_DASHBOARD=true' >> .env && \
    echo 'ENABLE_REACT_DASHBOARD=true' >> .env && \
    echo 'ENABLE_ENHANCED_ROUTING=true' >> .env && \
    echo 'ENABLE_REDIS_CACHE=true' >> .env && \
    echo 'ENABLE_SEMANTIC_CACHE=true' >> .env && \
    echo 'ENABLE_CIRCUIT_BREAKER=true' >> .env && \
    echo 'ENABLE_PERFORMANCE_MONITORING=true' >> .env && \
    echo 'ENABLE_AUTH=false' >> .env && \
    echo 'DEFAULT_API_KEY=sk-enhanced-4model-proxy' >> .env && \
    echo 'MAX_MEMORY_MB=16384' >> .env && \
    echo 'CACHE_MEMORY_LIMIT_MB=2048' >> .env && \
    echo 'MODEL_MEMORY_LIMIT_MB=8192' >> .env && \
    echo 'AUTO_DOWNLOAD_MODELS=false' >> .env

# Make scripts executable
RUN find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

# Set permissions
RUN chmod 755 data logs cache models static services middleware backups

EXPOSE 8001 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use start.sh if it exists, otherwise create a simple startup
CMD if [ -f "start.sh" ]; then ./start.sh; else ollama serve & sleep 5 && python main_master.py || python main.py; fi
