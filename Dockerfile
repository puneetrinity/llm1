# Complete LLM Proxy Dockerfile with Ollama, Models, and React Dashboard
# Multi-stage build for optimal image size and performance

# =====================================
# Stage 1: Frontend Build
# =====================================
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files for better caching
COPY frontend/package*.json ./

# Install dependencies (including devDependencies needed for build)
RUN npm ci --silent

# Copy frontend source code
COPY frontend/ ./

# Build the React application
RUN npm run build

# Verify build was successful
RUN ls -la build/ && test -f build/index.html

# =====================================
# Stage 2: Main Application with Ollama
# =====================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    git \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy Python requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY config*.py ./

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create necessary directories
RUN mkdir -p data logs /root/.ollama

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/app/data/models

# Create model download script
RUN cat > /app/download_models.sh << 'EOF'
#!/bin/bash
set -e

echo "🤖 Starting Ollama and downloading models..."

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
for i in {1..60}; do
  if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  echo "   Attempt $i/60 - waiting 5 seconds..."
  sleep 5
done

# Check if Ollama started successfully
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "❌ Failed to start Ollama service"
  exit 1
fi

# Download models in priority order
echo "📦 Downloading models (this may take 10-20 minutes)..."

# Priority 1: Mistral (most frequently used)
echo "🔄 Downloading Mistral 7B (4GB)..."
ollama pull mistral:7b-instruct-q4_0

# Priority 2: DeepSeek (coding specialist)
echo "🔄 Downloading DeepSeek V2 7B (4.1GB)..."
ollama pull deepseek-v2:7b-q4_0

# Priority 3: LLaMA3 (creative tasks)
echo "🔄 Downloading LLaMA3 8B (4.7GB)..."
ollama pull llama3:8b-instruct-q4_0

echo "✅ All models downloaded successfully!"

# Test each model
echo "🧪 Testing models..."
for model in "mistral:7b-instruct-q4_0" "deepseek-v2:7b-q4_0" "llama3:8b-instruct-q4_0"; do
  echo "Testing $model..."
  if ollama run $model "Hello" --timeout 30s >/dev/null 2>&1; then
    echo "✅ $model working"
  else
    echo "⚠️ $model test failed"
  fi
done

# Stop Ollama for now
kill $OLLAMA_PID 2>/dev/null && wait $OLLAMA_PID 2>/dev/null

echo "🎉 Model setup complete!"
EOF

RUN chmod +x /app/download_models.sh

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting LLM Proxy Service..."

# Start Ollama in background
echo "📡 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
  if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  echo "   Attempt $i/30 - waiting 2 seconds..."
  sleep 2
done

# Check if models exist, download if needed
echo "🔍 Checking for models..."
if ! ollama list | grep -q "mistral:7b-instruct-q4_0"; then
  echo "📦 Models not found, downloading..."
  /app/download_models.sh
fi

# Warm up the primary model
echo "🔥 Warming up primary model (Mistral)..."
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b-instruct-q4_0",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false,
    "options": {"num_predict": 5}
  }' >/dev/null 2>&1 || echo "⚠️ Warmup failed (normal on first run)"

echo "✅ Mistral warmed up and ready!"

# Start the FastAPI application
echo "🌐 Starting FastAPI application on port 8001..."
echo "📍 Dashboard: http://localhost:8001/app"
echo "📚 API Docs: http://localhost:8001/docs"
echo "🏥 Health: http://localhost:8001/health"
echo ""

# Cleanup function
cleanup() {
  echo "🛑 Shutting down services..."
  kill $OLLAMA_PID 2>/dev/null
  exit
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start FastAPI app
python main.py &
APP_PID=$!

# Wait for either process to exit
wait
EOF

RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 8001 11434

# Health check that verifies both FastAPI and Ollama
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health && curl -f http://localhost:11434/api/tags || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]
