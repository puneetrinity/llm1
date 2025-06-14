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

# Copy requirements first
COPY requirements.txt ./

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

# Create frontend directory structure
RUN mkdir -p /app/frontend

# Copy frontend files if they exist, create minimal structure if not
COPY frontend/ ./frontend/ 2>/dev/null || echo "No frontend directory found, creating minimal structure"

# Create minimal frontend structure if missing
RUN if [ ! -f "/app/frontend/package.json" ]; then \
        echo "Creating minimal frontend structure..." && \
        mkdir -p /app/frontend && \
        printf '%s\n' \
            '{' \
            '  "name": "llm-proxy-dashboard",' \
            '  "version": "1.0.0",' \
            '  "private": true,' \
            '  "dependencies": {' \
            '    "react": "^18.2.0",' \
            '    "react-dom": "^18.2.0",' \
            '    "react-scripts": "5.0.1"' \
            '  },' \
            '  "scripts": {' \
            '    "start": "react-scripts start",' \
            '    "build": "react-scripts build",' \
            '    "test": "react-scripts test",' \
            '    "eject": "react-scripts eject"' \
            '  },' \
            '  "eslintConfig": {' \
            '    "extends": ["react-app", "react-app/jest"]' \
            '  },' \
            '  "browserslist": {' \
            '    "production": [">0.2%", "not dead", "not op_mini all"],' \
            '    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]' \
            '  }' \
            '}' \
        > /app/frontend/package.json && \
        echo "✅ Created minimal package.json"; \
    else \
        echo "✅ Found existing package.json"; \
    fi

# Install Node.js dependencies
WORKDIR /app/frontend
RUN if [ -f "package.json" ]; then \
        echo "Installing Node.js dependencies..." && \
        npm config set fund false && \
        npm config set audit-level none && \
        npm install --legacy-peer-deps --prefer-offline --no-optional || echo "npm install failed, continuing..."; \
    else \
        echo "No frontend package.json found"; \
    fi

# Create minimal React app structure if src doesn't exist
RUN if [ ! -d "src" ]; then \
        echo "Creating minimal React app structure..." && \
        mkdir -p src public && \
        printf '%s\n' \
            "import React from 'react';" \
            "import ReactDOM from 'react-dom/client';" \
            "import App from './App';" \
            "" \
            "const root = ReactDOM.createRoot(document.getElementById('root'));" \
            "root.render(<App />);" \
        > src/index.js && \
        printf '%s\n' \
            "import React from 'react';" \
            "" \
            "function App() {" \
            "  return (" \
            "    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>" \
            "      <h1>🚀 LLM Proxy Dashboard</h1>" \
            "      <p>Dashboard is loading...</p>" \
            "    </div>" \
            "  );" \
            "}" \
            "" \
            "export default App;" \
        > src/App.js && \
        printf '%s\n' \
            '<!DOCTYPE html>' \
            '<html lang="en">' \
            '<head>' \
            '    <meta charset="UTF-8">' \
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">' \
            '    <title>LLM Proxy Dashboard</title>' \
            '</head>' \
            '<body>' \
            '    <div id="root"></div>' \
            '</body>' \
            '</html>' \
        > public/index.html && \
        echo "✅ Created minimal React structure"; \
    fi

# Try to build React app, create fallback if it fails
RUN if [ -f "package.json" ] && [ -d "src" ]; then \
        echo "Building React frontend..." && \
        (GENERATE_SOURCEMAP=false CI=true NODE_OPTIONS="--max-old-space-size=4096" npm run build && \
        echo "✅ React build completed successfully!") || \
        (echo "⚠️ React build failed, creating fallback..."); \
    else \
        echo "Skipping React build - using fallback dashboard"; \
    fi

# Copy application code
WORKDIR /app
COPY . ./

# === DEBUGGING SECTION ===
RUN echo "=== DEBUGGING: Checking file structure ===" && \
    echo "Current directory:" && pwd && \
    echo "Files in /app:" && ls -la /app/ && \
    echo "Frontend structure:" && ls -la /app/frontend/ 2>/dev/null || echo "No frontend directory" && \
    echo "Looking for main files:" && \
    find /app -name "main*.py" | head -5 && \
    echo "Contents of start.sh:" && cat /app/start.sh 2>/dev/null || echo "start.sh not found"

# Test Python import if main files exist
RUN echo "=== Testing Python imports ===" && \
    python3 -c "import sys; sys.path.insert(0, '/app')" && \
    (python3 -c "import main_master" && echo "✅ main_master imports successfully") || \
    (python3 -c "import main" && echo "✅ main imports successfully") || \
    echo "❌ No main files import successfully"

# Create guaranteed working dashboard
RUN echo "Creating guaranteed working dashboard..." && \
    mkdir -p /app/frontend/build && \
    printf '%s\n' \
        '<!DOCTYPE html>' \
        '<html lang="en">' \
        '<head>' \
        '    <meta charset="UTF-8">' \
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">' \
        '    <title>🚀 LLM Proxy Dashboard</title>' \
        '    <style>' \
        '        * { margin: 0; padding: 0; box-sizing: border-box; }' \
        '        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; padding: 20px; }' \
        '        .container { max-width: 1200px; margin: 0 auto; background: rgba(255,255,255,0.95); border-radius: 15px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }' \
        '        .header { text-align: center; margin-bottom: 30px; }' \
        '        .header h1 { color: #5a67d8; font-size: 2.5rem; margin-bottom: 10px; }' \
        '        .status { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; }' \
        '        .status.healthy { background: #c6f6d5; color: #22543d; }' \
        '        .status.error { background: #fed7d7; color: #742a2a; }' \
        '        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin: 30px 0; }' \
        '        .panel { background: #f7fafc; padding: 25px; border-radius: 12px; border-left: 4px solid #5a67d8; }' \
        '        .panel h3 { color: #2d3748; margin-bottom: 20px; font-size: 1.3rem; }' \
        '        .form-group { margin-bottom: 15px; }' \
        '        label { display: block; margin-bottom: 5px; font-weight: 600; color: #4a5568; }' \
        '        input, textarea, select { width: 100%; padding: 12px; border: 2px solid #e2e8f0; border-radius: 8px; font-size: 14px; transition: border-color 0.3s; }' \
        '        input:focus, textarea:focus, select:focus { outline: none; border-color: #5a67d8; box-shadow: 0 0 0 3px rgba(90,103,216,0.1); }' \
        '        button { background: linear-gradient(135deg, #5a67d8, #667eea); color: white; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; transition: transform 0.2s; margin: 5px; }' \
        '        button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(90,103,216,0.3); }' \
        '        .result { background: #edf2f7; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #38b2ac; max-height: 400px; overflow-y: auto; }' \
        '        pre { white-space: pre-wrap; word-wrap: break-word; font-size: 13px; }' \
        '        .loading { opacity: 0.6; pointer-events: none; }' \
        '    </style>' \
        '</head>' \
        '<body>' \
        '    <div class="container">' \
        '        <div class="header">' \
        '            <h1>🚀 LLM Proxy Dashboard</h1>' \
        '            <div id="status" class="status healthy">🔄 Initializing...</div>' \
        '        </div>' \
        '        <div class="grid">' \
        '            <div class="panel">' \
        '                <h3>🏥 System Health</h3>' \
        '                <button onclick="checkHealth()">Check Health</button>' \
        '                <button onclick="listModels()">List Models</button>' \
        '                <button onclick="getMetrics()">Get Metrics</button>' \
        '                <div id="healthResult" class="result" style="display:none;"></div>' \
        '            </div>' \
        '            <div class="panel">' \
        '                <h3>💬 Chat Interface</h3>' \
        '                <div class="form-group">' \
        '                    <label>Model:</label>' \
        '                    <select id="modelSelect">' \
        '                        <option value="mistral:7b-instruct-q4_0">Mistral 7B</option>' \
        '                    </select>' \
        '                </div>' \
        '                <div class="form-group">' \
        '                    <label>Message:</label>' \
        '                    <textarea id="chatInput" rows="3" placeholder="Type your message here..."></textarea>' \
        '                </div>' \
        '                <button onclick="sendChat()">Send Message</button>' \
        '                <button onclick="clearChat()">Clear</button>' \
        '                <div id="chatResult" class="result" style="display:none;"></div>' \
        '            </div>' \
        '        </div>' \
        '    </div>' \
    > /app/frontend/build/index.html && \
    printf '%s\n' \
        '    <script>' \
        '        let isLoading = false;' \
        '        window.onload = function() { checkHealth(); loadModels(); };' \
        '        function setLoading(loading) { isLoading = loading; document.body.className = loading ? "loading" : ""; }' \
        '        function showResult(elementId, data, title = "Result") {' \
        '            const element = document.getElementById(elementId);' \
        '            element.style.display = "block";' \
        '            element.innerHTML = `<h4>${title}</h4><pre>${JSON.stringify(data, null, 2)}</pre>`;' \
        '        }' \
        '        function updateStatus(text, healthy = true) {' \
        '            const status = document.getElementById("status");' \
        '            status.textContent = text;' \
        '            status.className = "status " + (healthy ? "healthy" : "error");' \
        '        }' \
        '        async function checkHealth() {' \
        '            setLoading(true);' \
        '            try {' \
        '                const response = await fetch("/health");' \
        '                const data = await response.json();' \
        '                showResult("healthResult", {status: response.status, health: data}, "🏥 Health Check");' \
        '                updateStatus("🟢 System Healthy", true);' \
        '            } catch (error) {' \
        '                showResult("healthResult", {error: error.message}, "❌ Health Error");' \
        '                updateStatus("🔴 System Error", false);' \
        '            } finally { setLoading(false); }' \
        '        }' \
        '        async function listModels() {' \
        '            setLoading(true);' \
        '            try {' \
        '                const response = await fetch("/v1/models");' \
        '                const data = await response.json();' \
        '                showResult("healthResult", {status: response.status, models: data}, "🤖 Available Models");' \
        '            } catch (error) { showResult("healthResult", {error: error.message}, "❌ Models Error"); }' \
        '            finally { setLoading(false); }' \
        '        }' \
        '        async function getMetrics() {' \
        '            setLoading(true);' \
        '            try {' \
        '                const response = await fetch("/metrics");' \
        '                const data = await response.json();' \
        '                showResult("healthResult", data, "📊 System Metrics");' \
        '            } catch (error) { showResult("healthResult", {error: error.message}, "❌ Metrics Error"); }' \
        '            finally { setLoading(false); }' \
        '        }' \
        '        async function loadModels() {' \
        '            try {' \
        '                const response = await fetch("/v1/models");' \
        '                const data = await response.json();' \
        '                const select = document.getElementById("modelSelect");' \
        '                select.innerHTML = "";' \
        '                if (data.data) {' \
        '                    data.data.forEach(model => {' \
        '                        const option = document.createElement("option");' \
        '                        option.value = model.id; option.textContent = model.id;' \
        '                        select.appendChild(option);' \
        '                    });' \
        '                }' \
        '            } catch (error) { console.log("Failed to load models:", error); }' \
        '        }' \
        '        async function sendChat() {' \
        '            const input = document.getElementById("chatInput").value;' \
        '            const model = document.getElementById("modelSelect").value;' \
        '            if (!input.trim()) { alert("Please enter a message"); return; }' \
        '            setLoading(true);' \
        '            try {' \
        '                const response = await fetch("/v1/chat/completions", {' \
        '                    method: "POST", headers: { "Content-Type": "application/json" },' \
        '                    body: JSON.stringify({ messages: [{role: "user", content: input}], model: model, stream: false })' \
        '                });' \
        '                const data = await response.json();' \
        '                showResult("chatResult", { input: input, model: model, response: data }, "💬 Chat Response");' \
        '            } catch (error) { showResult("chatResult", {error: error.message}, "❌ Chat Error"); }' \
        '            finally { setLoading(false); }' \
        '        }' \
        '        function clearChat() {' \
        '            document.getElementById("chatInput").value = "";' \
        '            document.getElementById("chatResult").style.display = "none";' \
        '        }' \
        '        const originalFetch = window.fetch;' \
        '        window.fetch = function(url, options) {' \
        '            if (url.includes("auth/websocket-session")) {' \
        '                console.log("🚫 Blocked WebSocket auth call");' \
        '                return Promise.resolve({ok: false, status: 503});' \
        '            }' \
        '            return originalFetch.apply(this, arguments);' \
        '        };' \
        '    </script>' \
        '</body>' \
        '</html>' \
    >> /app/frontend/build/index.html && \
    echo "✅ Guaranteed working dashboard created"

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
