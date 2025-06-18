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

# Copy application files
COPY main*.py ./
COPY services/ ./services/

# Create basic HTML page
RUN mkdir -p static && \
    echo '<html><body><h1>LLM Proxy API</h1><p><a href="/docs">API Documentation</a></p><p><a href="/health">Health Check</a></p></body></html>' > static/index.html

# Create environment file
RUN echo 'PORT=8001' > .env && \
    echo 'HOST=0.0.0.0' >> .env && \
    echo 'OLLAMA_BASE_URL=http://localhost:11434' >> .env && \
    echo 'ENABLE_STREAMING=true' >> .env && \
    echo 'ENABLE_AUTH=false' >> .env

# Create startup script
RUN echo '#!/bin/bash' > start.sh && \
    echo 'echo "Starting Ollama..."' >> start.sh && \
    echo 'ollama serve &' >> start.sh && \
    echo 'sleep 5' >> start.sh && \
    echo 'echo "Starting LLM Proxy..."' >> start.sh && \
    echo 'python main.py' >> start.sh && \
    chmod +x start.sh

EXPOSE 8001 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["./start.sh"]
