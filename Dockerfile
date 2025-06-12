FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Environment variables for GPU and memory management
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_NUM_PARALLEL=2
ENV OLLAMA_MAX_LOADED_MODELS=2
ENV NVIDIA_VISIBLE_DEVICES=all
ENV GPU_MEMORY_FRACTION=0.9

# Memory limits for enhanced features
ENV MAX_MEMORY_MB=8192
ENV CACHE_MEMORY_LIMIT_MB=1024
ENV MODEL_MEMORY_LIMIT_MB=4096
ENV SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Feature toggles (conservative defaults)
ENV ENABLE_SEMANTIC_CLASSIFICATION=false
ENV ENABLE_STREAMING=true
ENV ENABLE_MODEL_WARMUP=true
ENV ENABLE_DETAILED_METRICS=true
ENV ENABLE_DASHBOARD=false
ENV ENABLE_WEBSOCKET_DASHBOARD=false

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions for stability
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/cache /app/logs /app/models

# Health check with appropriate timeouts
HEALTHCHECK --interval=30s --timeout=20s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 11434

# Start script
CMD ["python3", "main.py"]
