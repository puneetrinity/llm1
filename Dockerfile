# Dockerfile.clean - Clean Enhanced LLM Proxy Image
# Uses the working main.py file with all enhanced features

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update --fix-missing \
    && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Enhanced dependencies for full functionality
RUN pip install --no-cache-dir \
    sentence-transformers \
    scikit-learn \
    sse-starlette \
    python-multipart \
    aiofiles

# Copy application files (ONLY the working ones)
COPY main.py .
COPY config.py .
COPY config_enhanced.py .

# Copy working services
COPY services/ ./services/
COPY utils/ ./utils/
COPY middleware/ ./middleware/
COPY models/ ./models/

# Create directories
RUN mkdir -p logs data cache models static

# Create optimized environment file
RUN echo "# Enhanced LLM Proxy Configuration" > .env \
    && echo "HOST=0.0.0.0" >> .env \
    && echo "PORT=8001" >> .env \
    && echo "DEBUG=false" >> .env \
    && echo "LOG_LEVEL=INFO" >> .env \
    && echo "" >> .env \
    && echo "# Ollama Configuration" >> .env \
    && echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env \
    && echo "OLLAMA_TIMEOUT=300" >> .env \
    && echo "OLLAMA_MAX_LOADED_MODELS=4" >> .env \
    && echo "" >> .env \
    && echo "# Enhanced Features (ALL ENABLED)" >> .env \
    && echo "ENABLE_AUTH=false" >> .env \
    && echo "ENABLE_DASHBOARD=true" >> .env \
    && echo "ENABLE_ENHANCED_FEATURES=true" >> .env \
    && echo "ENABLE_WEBSOCKET=false" >> .env \
    && echo "ENABLE_SEMANTIC_CLASSIFICATION=true" >> .env \
    && echo "ENABLE_MODEL_ROUTING=true" >> .env \
    && echo "ENABLE_4_MODEL_ROUTING=true" >> .env \
    && echo "" >> .env \
    && echo "# Memory Management" >> .env \
    && echo "MAX_MEMORY_MB=16384" >> .env \
    && echo "CACHE_MEMORY_LIMIT_MB=2048" >> .env \
    && echo "MODEL_MEMORY_LIMIT_MB=8192" >> .env \
    && echo "" >> .env \
    && echo "# Model Configuration" >> .env \
    && echo "PHI_MODEL=phi3.5" >> .env \
    && echo "MISTRAL_MODEL=mistral:7b-instruct-q4_0" >> .env \
    && echo "GEMMA_MODEL=gemma:7b-instruct" >> .env \
    && echo "LLAMA_MODEL=llama3:8b-instruct-q4_0" >> .env \
    && echo "" >> .env \
    && echo "# CORS" >> .env \
    && echo "CORS_ORIGINS=[\"*\"]" >> .env \
    && echo "CORS_ALLOW_CREDENTIALS=true" >> .env

# Create clean startup script
RUN echo "#!/bin/bash" > start_clean.sh \
    && echo "set -e" >> start_clean.sh \
    && echo "" >> start_clean.sh \
    && echo "echo '🚀 Enhanced LLM Proxy - Clean Start'" >> start_clean.sh \
    && echo "echo '=================================='" >> start_clean.sh \
    && echo "echo 'Using: main.py (complete version)'" >> start_clean.sh \
    && echo "echo 'Models: Phi3.5 | Mistral 7B | Gemma 7B | Llama3 8B'" >> start_clean.sh \
    && echo "echo ''" >> start_clean.sh \
    && echo "echo '📡 Starting Ollama...'" >> start_clean.sh \
    && echo "ollama serve > /tmp/ollama.log 2>&1 &" >> start_clean.sh \
    && echo "sleep 15" >> start_clean.sh \
    && echo "echo '📦 Checking models...'" >> start_clean.sh \
    && echo 'if ! ollama list | grep -q "phi3.5"; then' >> start_clean.sh \
    && echo '  echo "Downloading Phi3.5..."' >> start_clean.sh \
    && echo '  ollama pull phi3.5 &' >> start_clean.sh \
    && echo 'fi' >> start_clean.sh \
    && echo 'if ! ollama list | grep -q "mistral:7b-instruct-q4_0"; then' >> start_clean.sh \
    && echo '  echo "Downloading Mistral 7B..."' >> start_clean.sh \
    && echo '  ollama pull mistral:7b-instruct-q4_0 &' >> start_clean.sh \
    && echo 'fi' >> start_clean.sh \
    && echo 'if ! ollama list | grep -q "gemma:7b-instruct"; then' >> start_clean.sh \
    && echo '  echo "Downloading Gemma 7B..."' >> start_clean.sh \
    && echo '  ollama pull gemma:7b-instruct &' >> start_clean.sh \
    && echo 'fi' >> start_clean.sh \
    && echo 'if ! ollama list | grep -q "llama3:8b-instruct-q4_0"; then' >> start_clean.sh \
    && echo '  echo "Downloading Llama3 8B..."' >> start_clean.sh \
    && echo '  ollama pull llama3:8b-instruct-q4_0 &' >> start_clean.sh \
    && echo 'fi' >> start_clean.sh \
    && echo 'wait' >> start_clean.sh \
    && echo "echo '✅ All models ready'" >> start_clean.sh \
    && echo "echo '🌐 Starting Enhanced LLM Proxy...'" >> start_clean.sh \
    && echo "python main.py" >> start_clean.sh
RUN chmod +x start_clean.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Debug: keep container running for manual inspection
CMD ["tail", "-f", "/dev/null"]
#CMD ["./start_clean.sh"]
