# Dockerfile.clean - Clean Enhanced LLM Proxy Image
# Uses the working main.py file with all enhanced features

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies first
RUN apt-get update --fix-missing \
    && apt-get install -y \
        curl \
        wget \
        gnupg2 \
        software-properties-common \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install Python dependencies FIRST
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ML dependencies for full functionality
RUN pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    transformers==4.35.2 \
    torch \
    numpy \
    scikit-learn \
    faiss-cpu \
    sse-starlette \
    python-multipart \
    aiofiles

# Create directories before copying files
RUN mkdir -p logs data cache models static services middleware utils

# Copy application files in correct order
COPY config.py .
COPY main.py .

# Copy service directories
COPY services/ ./services/
COPY middleware/ ./middleware/
COPY utils/ ./utils/
COPY models/ ./models/

# Copy optional files if they exist
COPY config_enhanced.py .

# Create a production .env file with proper escaping
RUN echo "# Production Configuration" > .env \
    && echo "HOST=0.0.0.0" >> .env \
    && echo "PORT=8001" >> .env \
    && echo "DEBUG=false" >> .env \
    && echo "LOG_LEVEL=INFO" >> .env \
    && echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env \
    && echo "OLLAMA_TIMEOUT=300" >> .env \
    && echo "CORS_ORIGINS=[\"*\"]" >> .env \
    && echo "CORS_ALLOW_CREDENTIALS=true" >> .env \
    && echo "ENABLE_AUTH=false" >> .env \
    && echo "ENABLE_DASHBOARD=true" >> .env \
    && echo "ENABLE_ENHANCED_FEATURES=true" >> .env \
    && echo "ENABLE_WEBSOCKET=false" >> .env \
    && echo "MAX_MEMORY_MB=16384" >> .env \
    && echo "CACHE_MEMORY_LIMIT_MB=2048" >> .env \
    && echo "ENVIRONMENT=production" >> .env

# Create startup script
RUN echo '#!/bin/bash' > /app/start.sh \
    && echo 'set -e' >> /app/start.sh \
    && echo 'echo "🚀 Starting Enhanced LLM Proxy..."' >> /app/start.sh \
    && echo 'echo "📋 Configuration check:"' >> /app/start.sh \
    && echo 'python -c "from config import settings; print(f\"✅ Config loaded: {settings.HOST}:{settings.PORT}\")"' >> /app/start.sh \
    && echo 'echo "🔧 Testing imports..."' >> /app/start.sh \
    && echo 'python -c "import main; print(\"✅ Main imports successful\")"' >> /app/start.sh \
    && echo 'echo "🎯 Starting application..."' >> /app/start.sh \
    && echo 'python main.py' >> /app/start.sh \
    && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Use the startup script
CMD ["/app/start.sh"]
