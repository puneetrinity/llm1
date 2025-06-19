# Enhanced LLM Proxy - Fixed Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/usr/local/bin:$PATH"
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS="*"
ENV OLLAMA_KEEP_ALIVE=5m

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh && \
    mkdir -p /root/.ollama/models && \
    mkdir -p /app/data/logs && \
    mkdir -p /app/data/cache && \
    chmod -R 755 /root/.ollama

# Copy and install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir \
        sentence-transformers \
        scikit-learn \
        sse-starlette \
        python-multipart \
        aiofiles

# Copy application files
COPY main.py .
COPY config.py .
COPY config_enhanced.py .
COPY services/ ./services/
COPY utils/ ./utils/
COPY middleware/ ./middleware/
COPY models/ ./models/

# Create optimized environment file
RUN echo "# Enhanced LLM Proxy Configuration" > .env && \
    echo "HOST=0.0.0.0" >> .env && \
    echo "PORT=8001" >> .env && \
    echo "DEBUG=false" >> .env && \
    echo "LOG_LEVEL=INFO" >> .env && \
    echo "" >> .env && \
    echo "# Ollama Configuration" >> .env && \
    echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env && \
    echo "OLLAMA_TIMEOUT=300" >> .env && \
    echo "OLLAMA_MAX_LOADED_MODELS=4" >> .env && \
    echo "" >> .env && \
    echo "# Features" >> .env && \
    echo "ENABLE_AUTH=false" >> .env && \
    echo "ENABLE_DASHBOARD=true" >> .env && \
    echo "ENABLE_ENHANCED_FEATURES=true" >> .env && \
    echo "ENABLE_SEMANTIC_CLASSIFICATION=true" >> .env && \
    echo "ENABLE_MODEL_ROUTING=true" >> .env && \
    echo "ENABLE_4_MODEL_ROUTING=true" >> .env

# Copy the fixed startup script
COPY start_fixed.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Use the fixed startup script
CMD ["/app/start.sh"]
