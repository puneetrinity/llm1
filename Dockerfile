FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    python3-dev \
    net-tools \
    procps \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (just the binary, NO models)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install sentence-transformers faiss-cpu sse-starlette redis aioredis prometheus-client numpy scikit-learn

# Copy ALL application files for 4-model system
COPY main*.py ./
COPY services/ ./services/
COPY config*.py ./
COPY middleware/ ./middleware/
COPY backups/ ./backups/
COPY start.sh ./
COPY enhanced_setup.sh ./
COPY frontend/ ./frontend/

# Copy helper scripts (includes download_models.sh for user to run later)
COPY start_dev.sh ./
COPY download_models.sh ./
COPY test_system.sh ./

# Create directories
RUN mkdir -p data/cache data/logs data/models logs cache models static tests scripts backups frontend/build

# Create enhanced environment for 4-model system
RUN cat > .env << 'EOF'
# Enhanced 4-Model LLM Proxy Configuration
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8001
LOG_LEVEL=INFO
DEBUG=false

# 4-Model System Configuration
DEFAULT_MODEL=mistral:7b-instruct-q4_0
ENABLE_4_MODEL_ROUTING=true
PHI_MODEL=phi:3.5
MISTRAL_MODEL=mistral:7b-instruct-q4_0
GEMMA_MODEL=gemma:7b-instruct
LLAMA_MODEL=llama3:8b-instruct-q4_0

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=4

# Memory Management
MAX_MEMORY_MB=16384
CACHE_MEMORY_LIMIT_MB=2048
MODEL_MEMORY_LIMIT_MB=8192
SEMANTIC_MODEL_MAX_MEMORY_MB=1024

# Enhanced Features - ALL ENABLED
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true
ENABLE_DASHBOARD=true
ENABLE_REACT_DASHBOARD=true
ENABLE_WEBSOCKET_DASHBOARD=true
ENABLE_ENHANCED_ROUTING=true

# Performance & Caching
ENABLE_REDIS_CACHE=true
REDIS_URL=redis://localhost:6379
ENABLE_SEMANTIC_CACHE=true
SEMANTIC_SIMILARITY_THRESHOLD=0.85
ENABLE_CIRCUIT_BREAKER=true
ENABLE_CONNECTION_POOLING=true
ENABLE_PERFORMANCE_MONITORING=true

# Security & CORS
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-enhanced-4model-proxy
CORS_ORIGINS=["*"]
CORS_ALLOW_CREDENTIALS=true

# Dashboard Configuration
DASHBOARD_PATH=/app/static
ENABLE_WEBSOCKET=true

# NO AUTO-DOWNLOAD (user downloads manually after container starts)
AUTO_DOWNLOAD_MODELS=false
EOF

# Build frontend if available
RUN if [ -f "frontend/package.json" ]; then \
        cd frontend && \
        npm install --legacy-peer-deps && \
        npm run build && \
        cp -r build/* /app/static/ 2>/dev/null || true; \
    fi

# Create fallback static content
RUN if [ ! -f "static/index.html" ]; then \
        mkdir -p static && \
        cat > static/index.html << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced 4-Model LLM Proxy</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .models { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin: 30px 0; }
        .model { padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }
        .links { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
        .links a { padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
        .links a:hover { background: #0056b3; }
        .notice { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Enhanced 4-Model LLM Proxy</h1>
        
        <div class="notice">
            <strong>📥 First Time Setup:</strong> Download models with: <code>docker exec &lt;container&gt; ./download_models.sh</code>
        </div>
        
        <div class="models">
            <div class="model">
                <h3>🧠 Phi-3.5</h3>
                <p>Math, Logic, Reasoning</p>
            </div>
            <div class="model">
                <h3>⚡ Mistral 7B</h3>
                <p>General, Quick Facts</p>
            </div>
            <div class="model">
                <h3>⚙️ Gemma 7B</h3>
                <p>Technical, Coding</p>
            </div>
            <div class="model">
                <h3>🎨 Llama3 8B</h3>
                <p>Creative, Writing</p>
            </div>
        </div>
        
        <div class="links">
            <a href="/docs">📚 API Documentation</a>
            <a href="/health">🏥 Health Check</a>
            <a href="/v1/models">📋 Available Models</a>
            <a href="/metrics">📊 Metrics</a>
        </div>
    </div>
</body>
</html>
HTMLEOF
    fi

# Make scripts executable
RUN chmod +x start.sh enhanced_setup.sh start_dev.sh download_models.sh test_system.sh

# IMPORTANT: Create .ollama directory but do NOT download any models during build
RUN mkdir -p /root/.ollama

EXPOSE 8001 11434 6379

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use the sophisticated start.sh script
CMD ["./start.sh"]
