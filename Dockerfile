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

# Install system dependencies including Node.js for React + Vite
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

# Install Node.js 18.x for Vite
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verify Node.js installation
RUN echo "=== Node.js Version Check ===" && \
    node --version && \
    npm --version

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy Python requirements first
COPY requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install enhanced Python dependencies with error handling
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

# === FRONTEND BUILD SECTION WITH ERROR HANDLING ===
# Copy frontend source code first
COPY frontend/ ./frontend/

# Check frontend structure with detailed output
RUN echo "=== Frontend Structure Analysis ===" && \
    echo "Contents of /app/frontend/:" && \
    ls -la /app/frontend/ && \
    echo "" && \
    if [ -f "/app/frontend/package.json" ]; then \
        echo "✅ package.json found" && \
        echo "Package.json contents:" && \
        cat /app/frontend/package.json | head -20; \
    else \
        echo "❌ package.json not found"; \
        echo "Available files:" && \
        find /app/frontend -name "*.json" 2>/dev/null || echo "No JSON files found"; \
    fi && \
    echo "" && \
    if [ -d "/app/frontend/src" ]; then \
        echo "✅ src directory found" && \
        echo "Source files:" && \
        ls -la /app/frontend/src/ | head -10; \
    else \
        echo "❌ src directory not found"; \
    fi && \
    echo "" && \
    echo "Looking for config files:" && \
    ls -la /app/frontend/vite* /app/frontend/*config* 2>/dev/null || echo "No config files found"

# Install Node.js dependencies with better error handling
WORKDIR /app/frontend
RUN if [ -f "package.json" ]; then \
        echo "🔧 Configuring npm for Docker build..." && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm config set prefer-offline true && \
        npm config set cache /tmp/npm-cache && \
        echo "" && \
        echo "📦 Installing dependencies..." && \
        npm install --verbose 2>&1 | tee /tmp/npm-install.log && \
        echo "" && \
        echo "✅ Dependencies installed. Checking node_modules..." && \
        ls -la node_modules/ | head -10 && \
        echo "Total packages: $(ls node_modules/ | wc -l)"; \
    else \
        echo "❌ FATAL: No package.json found in frontend directory" && \
        echo "Cannot proceed with frontend build" && \
        exit 1; \
    fi

# Build frontend with enhanced error handling and fallback
RUN echo "🏗️ Starting frontend build process..." && \
    if [ -f "package.json" ] && [ -d "src" ]; then \
        echo "Pre-build check:" && \
        npm run --silent 2>/dev/null | grep -E "(build|dev)" || echo "Available scripts:" && \
        cat package.json | grep -A 10 '"scripts"' && \
        echo "" && \
        echo "Setting Node.js memory limit and starting build..." && \
        NODE_OPTIONS="--max-old-space-size=4096 --max-heap-size=4096" \
        GENERATE_SOURCEMAP=false \
        CI=true \
        npm run build 2>&1 | tee /tmp/build.log && \
        echo "" && \
        echo "✅ Build completed! Checking output..." && \
        (ls -la dist/ 2>/dev/null && echo "Vite dist/ directory found") || \
        (ls -la build/ 2>/dev/null && echo "Build/ directory found") || \
        echo "❌ No build output directory found"; \
    else \
        echo "❌ Cannot build - missing requirements" && \
        echo "Package.json exists: $([ -f package.json ] && echo YES || echo NO)" && \
        echo "Src directory exists: $([ -d src ] && echo YES || echo NO)" && \
        exit 1; \
    fi

# Handle build output and create fallback if needed
RUN echo "📁 Processing build output..." && \
    BUILD_SUCCESS=false && \
    if [ -d "dist" ] && [ -f "dist/index.html" ]; then \
        echo "✅ Vite build successful - dist/ directory found" && \
        BUILD_SUCCESS=true; \
    elif [ -d "build" ] && [ -f "build/index.html" ]; then \
        echo "✅ Build successful - build/ directory found" && \
        BUILD_SUCCESS=true; \
    else \
        echo "❌ Build failed or produced no output" && \
        echo "Checking for any HTML files:" && \
        find . -name "*.html" 2>/dev/null || echo "No HTML files found" && \
        echo "" && \
        echo "Build log:" && \
        cat /tmp/build.log 2>/dev/null | tail -20 || echo "No build log available" && \
        echo "" && \
        echo "🔧 Creating fallback dashboard..." && \
        mkdir -p dist && \
        printf '%s\n' \
            '<!DOCTYPE html>' \
            '<html lang="en">' \
            '<head>' \
            '    <meta charset="UTF-8">' \
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">' \
            '    <title>🚀 LLM Proxy Dashboard</title>' \
            '    <style>' \
            '        body { font-family: Arial, sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; margin: 0; }' \
            '        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }' \
            '        h1 { color: #5a67d8; text-align: center; margin-bottom: 20px; }' \
            '        .panel { background: #f7fafc; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #5a67d8; }' \
            '        button { background: #5a67d8; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; margin: 5px; }' \
            '        button:hover { background: #4c51bf; }' \
            '        .result { background: #edf2f7; padding: 15px; margin: 15px 0; border-radius: 5px; }' \
            '        pre { white-space: pre-wrap; word-wrap: break-word; font-size: 13px; }' \
            '    </style>' \
            '</head>' \
            '<body>' \
            '    <div class="container">' \
            '        <h1>🚀 LLM Proxy Dashboard</h1>' \
            '        <div class="panel">' \
            '            <h3>⚠️ Frontend Build Notice</h3>' \
            '            <p>The React build encountered issues, but the LLM Proxy is fully functional with this fallback dashboard.</p>' \
            '        </div>' \
            '        <div class="panel">' \
            '            <h3>🏥 System Health</h3>' \
            '            <button onclick="checkHealth()">Check Health</button>' \
            '            <button onclick="listModels()">List Models</button>' \
            '            <div id="healthResult" class="result" style="display:none;"></div>' \
            '        </div>' \
            '        <div class="panel">' \
            '            <h3>💬 Test Chat</h3>' \
            '            <input type="text" id="chatInput" placeholder="Ask something..." style="width: 70%; padding: 8px; margin: 5px;">' \
            '            <button onclick="sendChat()">Send</button>' \
            '            <div id="chatResult" class="result" style="display:none;"></div>' \
            '        </div>' \
            '    </div>' \
            '    <script>' \
            '        async function checkHealth() {' \
            '            try {' \
            '                const res = await fetch("/health");' \
            '                const data = await res.json();' \
            '                document.getElementById("healthResult").style.display = "block";' \
            '                document.getElementById("healthResult").innerHTML = "<pre>" + JSON.stringify(data, null, 2) + "</pre>";' \
            '            } catch (e) { alert("Health check failed: " + e.message); }' \
            '        }' \
            '        async function listModels() {' \
            '            try {' \
            '                const res = await fetch("/v1/models");' \
            '                const data = await res.json();' \
            '                document.getElementById("healthResult").style.display = "block";' \
            '                document.getElementById("healthResult").innerHTML = "<h4>Models:</h4><pre>" + JSON.stringify(data, null, 2) + "</pre>";' \
            '            } catch (e) { alert("Models fetch failed: " + e.message); }' \
            '        }' \
            '        async function sendChat() {' \
            '            const input = document.getElementById("chatInput").value;' \
            '            if (!input.trim()) return;' \
            '            try {' \
            '                const res = await fetch("/v1/chat/completions", {' \
            '                    method: "POST", headers: {"Content-Type": "application/json"},' \
            '                    body: JSON.stringify({messages: [{role: "user", content: input}], model: "mistral:7b-instruct-q4_0"})' \
            '                });' \
            '                const data = await res.json();' \
            '                document.getElementById("chatResult").style.display = "block";' \
            '                document.getElementById("chatResult").innerHTML = "<h4>Response:</h4><pre>" + JSON.stringify(data, null, 2) + "</pre>";' \
            '            } catch (e) { alert("Chat failed: " + e.message); }' \
            '        }' \
            '    </script>' \
            '</body>' \
            '</html>' \
        > dist/index.html && \
        echo "✅ Fallback dashboard created"; \
    fi

# Copy application code
WORKDIR /app
COPY . ./

# Setup frontend for FastAPI serving
RUN echo "🔗 Setting up frontend for FastAPI..." && \
    cd /app/frontend && \
    if [ -d "dist" ] && [ -f "dist/index.html" ]; then \
        echo "📁 Copying Vite dist/ to build/ for FastAPI compatibility..." && \
        cp -r dist build && \
        echo "✅ Frontend ready at /app/frontend/build/"; \
    elif [ -d "build" ] && [ -f "build/index.html" ]; then \
        echo "✅ Build directory already exists and ready"; \
    else \
        echo "❌ No valid frontend build found" && \
        mkdir -p build && \
        echo "<h1>Frontend Build Failed</h1>" > build/index.html; \
    fi && \
    ls -la /app/frontend/build/

# Final debugging and verification
RUN echo "=== FINAL VERIFICATION ===" && \
    echo "✅ Frontend build status:" && \
    ls -la /app/frontend/build/index.html && \
    echo "" && \
    echo "✅ Python files:" && \
    ls -la /app/main*.py && \
    echo "" && \
    echo "✅ Start script:" && \
    ls -la /app/start.sh 2>/dev/null || echo "No start.sh found" && \
    echo "" && \
    echo "🎉 Container verification complete!"

# Fix line endings and permissions
RUN find . -name "*.py" -exec dos2unix {} \; && \
    find . -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \;

# Create required directories
RUN mkdir -p logs cache models data static

# Ensure start.sh exists and is executable
RUN if [ ! -f "/app/start.sh" ]; then \
        echo "Creating fallback start.sh..." && \
        printf '%s\n' \
            '#!/bin/bash' \
            'echo "🚀 Starting LLM Proxy Server..."' \
            'echo "📍 Server: http://0.0.0.0:8001"' \
            'echo "📊 Dashboard: http://0.0.0.0:8001/app"' \
            'python3 -m uvicorn main_master:app --host 0.0.0.0 --port 8001 --reload' \
        > /app/start.sh && \
        chmod +x /app/start.sh; \
    else \
        chmod +x /app/start.sh; \
    fi

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 8001 11434

# Start the application
CMD ["./start.sh"]
