#!/bin/bash
set -e

log() {
    mkdir -p /app/data/logs
    echo "[$(date)] $1" | tee -a /app/data/logs/setup_debug.log
}

log "Starting Enhanced LLM Proxy with Ollama..."

# Set Ollama environment variables
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS="*"
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_MODELS=/root/.ollama/models

# Create necessary directories
mkdir -p /root/.ollama/models
mkdir -p /tmp/ollama
mkdir -p /app/data/logs

# Function to check if Ollama is ready
check_ollama_ready() {
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1
}

log "Starting Ollama service..."
if command -v ollama >/dev/null 2>&1; then
    # Kill any existing Ollama processes
    pkill -f "ollama serve" || true
    sleep 2
    
    # Start Ollama with proper logging
    nohup ollama serve > /app/data/logs/ollama.log 2>&1 &
    OLLAMA_PID=$!
    log "Ollama started with PID: $OLLAMA_PID"
    
    # Wait for Ollama to be ready (with timeout)
    MAX_ATTEMPTS=60
    ATTEMPT=0
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if check_ollama_ready; then
            log "✅ Ollama is ready after $ATTEMPT seconds"
            break
        fi
        
        # Check if process is still running
        if ! kill -0 $OLLAMA_PID 2>/dev/null; then
            log "❌ Ollama process died, checking logs..."
            tail -n 50 /app/data/logs/ollama.log | tee -a /app/data/logs/setup_debug.log
            exit 1
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
        if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
            log "❌ Ollama failed to start within $MAX_ATTEMPTS seconds"
            log "Ollama logs:"
            tail -n 100 /app/data/logs/ollama.log | tee -a /app/data/logs/setup_debug.log
            exit 1
        fi
        
        log "Waiting for Ollama... attempt $ATTEMPT/$MAX_ATTEMPTS"
        sleep 1
    done
    
    # Download models in background
    log "Starting model downloads..."
    {
        for model in "phi3.5" "mistral:7b-instruct-q4_0" "gemma:7b-instruct" "llama3:8b-instruct-q4_0"; do
            if ! ollama list | grep -q "$model"; then
                log "Downloading $model..."
                ollama pull $model || log "Warning: Failed to pull $model"
            else
                log "Model $model already available"
            fi
        done
        log "All model downloads completed"
    } &
    
else
    log "❌ Ollama binary not found!"
    exit 1
fi

# Give Ollama a moment to stabilize
sleep 5

# Start the Python application
log "Starting Enhanced LLM Proxy application..."
exec python main.py 2>&1 | tee -a /app/data/logs/app.log
