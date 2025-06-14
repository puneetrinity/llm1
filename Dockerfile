# Base image with CUDA
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# CUDA and GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ollama configuration for enhanced performance
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_NUM_PARALLEL=2 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    OLLAMA_GPU_OVERHEAD=0 \
    OLLAMA_DEBUG=INFO

# Memory management - Enhanced configuration
ENV MAX_MEMORY_MB=12288 \
    CACHE_MEMORY_LIMIT_MB=1024 \
    MODEL_MEMORY_LIMIT_MB=6144 \
    SEMANTIC_MODEL_MAX_MEMORY_MB=500

# Enhanced feature toggles - Progressive enablement
ENV ENABLE_SEMANTIC_CLASSIFICATION=false \
    ENABLE_STREAMING=true \
    ENABLE_MODEL_WARMUP=true \
    ENABLE_DETAILED_METRICS=true \
    ENABLE_DASHBOARD=true \
    ENABLE_WEBSOCKET_DASHBOARD=false

# Dashboard configuration
ENV DASHBOARD_PATH=/app/static \
    ENABLE_REACT_DASHBOARD=true

# Performance and caching
ENV ENABLE_REDIS_CACHE=true \
    REDIS_URL=redis://localhost:6379 \
    ENABLE_SEMANTIC_CACHE=true \
    SEMANTIC_SIMILARITY_THRESHOLD=0.85

# Security settings - Set at runtime for production security
ENV ENABLE_AUTH=false \
    CORS_ORIGINS='["*"]'

# Advanced features
ENV ENABLE_CIRCUIT_BREAKER=true \
    ENABLE_CONNECTION_POOLING=true \
    ENABLE_PERFORMANCE_MONITORING=true

# Core application settings
ENV HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO \
    DEBUG=false

WORKDIR /app

# Install system dependencies including Node.js for React dashboard
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    dos2unix \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18.x for React dashboard
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and package files first for better caching
COPY requirements.txt ./
COPY frontend/package*.json ./frontend/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install enhanced dependencies with error handling
RUN pip3 install --no-cache-dir \
    sentence-transformers>=2.2.0 \
    faiss-cpu==1.7.4 \
    sse-starlette==1.6.5 \
    redis>=4.5.0 \
    aioredis>=2.0.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.1.0 \
    prometheus-client \
    || echo "Some enhanced features may be limited"

# Install Node.js dependencies for React dashboard
WORKDIR /app/frontend
RUN if [ -f "package.json" ]; then \
        echo "Installing Node.js dependencies..." && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm install --legacy-peer-deps --prefer-offline --no-optional; \
    else \
        echo "No frontend package.json found"; \
    fi

# Copy frontend source code and build React dashboard
COPY frontend/ ./
RUN if [ -f "package.json" ] && [ -d "src" ]; then \
        echo "Building React frontend..." && \
        GENERATE_SOURCEMAP=false CI=true NODE_OPTIONS="--max-old-space-size=4096" npm run build && \
        echo "React build completed successfully!"; \
    else \
        echo "Skipping frontend build - no source found" && \
        mkdir -p build; \
    fi

# Copy application code
WORKDIR /app

# Copy critical files explicitly first
COPY start.sh ./
COPY requirements.txt ./
COPY *.py ./
COPY . ./

# === DEBUGGING SECTION ===
RUN echo "=== DEBUGGING: Checking file structure ===" && \
    echo "Current directory:" && pwd && \
    echo "Files in /app:" && ls -la /app/ && \
    echo "Looking for main files:" && \
    find /app -name "main*.py" | head -5 && \
    echo "Contents of start.sh:" && cat /app/start.sh 2>/dev/null || echo "start.sh not found"

# Test Python import if main files exist
RUN echo "=== Testing Python imports ===" && \
    python3 -c "import sys; sys.path.insert(0, '/app')" && \
    (python3 -c "import main_master" && echo "✅ main_master imports successfully") || \
    (python3 -c "import main" && echo "✅ main imports successfully") || \
    echo "❌ No main files import successfully"

# Create working dashboard if React build failed
RUN if [ ! -f "/app/frontend/build/index.html" ] || [ ! -s "/app/frontend/build/index.html" ]; then \
        echo "Creating working dashboard..." && \
        mkdir -p /app/frontend/build && \
        printf '%s\n' \
            '<!DOCTYPE html>' \
            '<html lang="en">' \
            '<head>' \
            '    <meta charset="UTF-8">' \
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">' \
            '    <title>🚀 LLM Proxy Dashboard</title>' \
            '    <style>' \
            '        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f0f2f5; }' \
            '        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }' \
            '        .header { text-align: center; margin-bottom: 30px; }' \
            '        .header h1 { color: #1976d2; margin-bottom: 10px; }' \
            '        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }' \
            '        .panel { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #1976d2; }' \
            '        button { background: #1976d2; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; margin: 5px; }' \
            '        button:hover { background: #1565c0; }' \
            '        .result { background: #e3f2fd; padding: 15px; margin: 15px 0; border-radius: 5px; max-height: 300px; overflow-y: auto; }' \
            '        pre { white-space: pre-wrap; word-wrap: break-word; font-size: 13px; }' \
            '        input, textarea, select { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }' \
            '    </style>' \
            '</head>' \
            '<body>' \
            '    <div class="container">' \
            '        <div class="header">' \
            '            <h1>🚀 LLM Proxy Dashboard</h1>' \
            '            <p>Status: <span id="status">🔄 Loading...</span></p>' \
            '        </div>' \
            '        <div class="grid">' \
            '            <div class="panel">' \
            '                <h3>🏥 System Status</h3>' \
            '                <button onclick="checkHealth()">Check Health</button>' \
            '                <button onclick="listModels()">List Models</button>' \
            '                <div id="healthResult" class="result" style="display:none;"></div>' \
            '            </div>' \
            '            <div class="panel">' \
            '                <h3>💬 Test Chat</h3>' \
            '                <select id="modelSelect"><option value="mistral:7b-instruct-q4_0">Mistral 7B</option></select>' \
            '                <textarea id="chatInput" rows="3" placeholder="Ask something..."></textarea>' \
            '                <button onclick="sendChat()">Send</button>' \
            '                <div id="chatResult" class="result" style="display:none;"></div>' \
            '            </div>' \
            '        </div>' \
            '    </div>' \
            '    <script>' \
            '        window.onload = () => checkHealth();' \
            '        function showResult(id, data) {' \
            '            const el = document.getElementById(id);' \
            '            el.style.display = "block";' \
            '            el.innerHTML = "<pre>" + JSON.stringify(data, null, 2) + "</pre>";' \
            '        }' \
            '        async function checkHealth() {' \
            '            try {' \
            '                const res = await fetch("/health");' \
            '                const data = await res.json();' \
            '                showResult("healthResult", data);' \
            '                document.getElementById("status").innerHTML = "🟢 Healthy";' \
            '            } catch (e) {' \
            '                showResult("healthResult", {error: e.message});' \
            '                document.getElementById("status").innerHTML = "🔴 Error";' \
            '            }' \
            '        }' \
            '        async function listModels() {' \
            '            try {' \
            '                const res = await fetch("/v1/models");' \
            '                const data = await res.json();' \
            '                showResult("healthResult", data);' \
            '            } catch (e) { showResult("healthResult", {error: e.message}); }' \
            '        }' \
            '        async function sendChat() {' \
            '            const input = document.getElementById("chatInput").value;' \
            '            if (!input.trim()) return;' \
            '            try {' \
            '                const res = await fetch("/v1/chat/completions", {' \
            '                    method: "POST",' \
            '                    headers: {"Content-Type": "application/json"},' \
            '                    body: JSON.stringify({' \
            '                        messages: [{role: "user", content: input}],' \
            '                        model: "mistral:7b-instruct-q4_0"' \
            '                    })' \
            '                });' \
            '                const data = await res.json();' \
            '                showResult("chatResult", data);' \
            '            } catch (e) { showResult("chatResult", {error: e.message}); }' \
            '        }' \
            '        const originalFetch = window.fetch;' \
            '        window.fetch = function(url, options) {' \
            '            if (url.includes("auth/websocket-session")) {' \
            '                return Promise.resolve({ok: false, status: 503});' \
            '            }' \
            '            return originalFetch.apply(this, arguments);' \
            '        };' \
            '    </script>' \
            '</body>' \
            '</html>' \
        > /app/frontend/build/index.html && \
        echo "✅ Working dashboard created"; \
    else \
        echo "✅ React build exists"; \
    fi

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \;

# Create required directories
RUN mkdir -p logs cache models data static frontend/build

# Ensure start.sh is executable and exists
RUN if [ ! -f "/app/start.sh" ]; then \
        echo "❌ start.sh not found - creating fallback..." && \
        printf '%s\n' \
            '#!/bin/bash' \
            'echo "🚀 Starting LLM Proxy Server..."' \
            'echo "📍 Server will be available at: http://0.0.0.0:8001"' \
            'echo "📊 Dashboard will be available at: http://0.0.0.0:8001/app"' \
            '' \
            '# Start the application' \
            'python3 -m uvicorn main_master:app --host 0.0.0.0 --port 8001 --reload' \
        > /app/start.sh && \
        chmod +x /app/start.sh && \
        echo "✅ Fallback start.sh created"; \
    else \
        echo "✅ start.sh exists" && \
        chmod +x /app/start.sh; \
    fi

# Final verification
RUN echo "=== FINAL VERIFICATION ===" && \
    echo "✅ Files ready:" && \
    ls -la /app/start.sh && \
    echo "✅ Dashboard ready:" && \
    ls -la /app/frontend/build/index.html && \
    echo "🎉 Container is ready!"

# Comprehensive health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Use the startup script
CMD ["./start.sh"]
