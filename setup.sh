#!/bin/bash
# Fixed setup.sh - Prevents infinite loops and properly starts service
# Replace your existing setup.sh with this version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🚀 Enhanced LLM Proxy Setup (Fixed Version)"
echo "==========================================="

# CRITICAL: Check if we're in a loop
if [ -f "/workspace/.setup_in_progress" ]; then
    print_error "Setup already in progress! Breaking potential loop..."
    rm -f /workspace/.setup_in_progress
    sleep 2
fi

# CRITICAL: Check if setup already completed
if [ -f "/workspace/.setup_complete" ]; then
    print_status "Setup already completed successfully!"
    print_info "🚀 Starting LLM Proxy service directly..."
    
    cd /workspace/app
    
    # CRITICAL: Activate virtual environment first
    if [ -f "/workspace/venv/bin/activate" ]; then
        print_info "Activating virtual environment..."
        source /workspace/venv/bin/activate
        print_status "Virtual environment activated"
    else
        print_warning "No virtual environment found, installing dependencies globally..."
        pip3 install fastapi uvicorn pydantic pydantic-settings aiohttp psutil
    fi
    
    # Start the service immediately - NO MORE LOOPS!
    if [ -f "main.py" ]; then
        print_status "Starting main.py..."
        exec python3 main.py
    elif [ -f "main_enhanced.py" ]; then
        print_status "Starting enhanced version..."
        exec python3 main_enhanced.py
    else
        print_error "No main file found!"
        ls -la
        exit 1
    fi
fi

# Mark setup as in progress
touch /workspace/.setup_in_progress

print_info "Starting fresh setup process..."

# Update system packages
print_info "Updating system packages..."
apt-get update && apt-get install -y curl git python3-pip build-essential || {
    print_warning "Some packages failed to install, continuing..."
}

# Set up workspace
cd /workspace

# Clone repository if needed
if [ ! -d "/workspace/app" ]; then
    print_info "Cloning repository..."
    git clone https://github.com/your-repo/llm-proxy.git app || {
        print_error "Failed to clone repository"
        exit 1
    }
else
    print_status "Repository already exists"
fi

cd /workspace/app

# Install Python dependencies
print_info "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install fastapi uvicorn aiohttp pydantic pydantic-settings psutil || {
    print_error "Failed to install core dependencies"
    exit 1
}

# Install enhanced dependencies (optional)
print_info "Installing enhanced dependencies (with fallbacks)..."
pip3 install sentence-transformers faiss-cpu sse-starlette redis aioredis || {
    print_warning "Some enhanced features may be limited"
}

# Set up configuration
print_info "Setting up configuration..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Basic Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434

# Enhanced Features (Conservative settings)
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
MAX_MEMORY_MB=4096

# Security
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-runpod-key
EOF
    print_status "Configuration created"
else
    print_status "Configuration already exists"
fi

# Create required directories
mkdir -p data/{cache,logs,models}
touch data/{cache,logs,models}/.gitkeep

# CRITICAL: Ensure main.py exists and is executable
if [ ! -f "main.py" ]; then
    print_info "Creating basic main.py..."
    cat > main.py << 'EOF'
#!/usr/bin/env python3
# Basic LLM Proxy Main File

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create FastAPI app
app = FastAPI(
    title="LLM Proxy",
    description="Production-ready LLM routing proxy",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "message": "LLM Proxy is running successfully"
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Proxy API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/v1/chat/completions")
async def chat_completions():
    """Basic chat completions endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! LLM Proxy is running. Please configure Ollama connection for full functionality."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    )

if __name__ == "__main__":
    print("🚀 Starting LLM Proxy...")
    print("📋 Health check: http://localhost:8000/health")
    print("📚 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
EOF
fi

# Make main.py executable
chmod +x main.py

print_status "Setup completed successfully!"

# CRITICAL: Mark setup as complete BEFORE starting service
touch /workspace/.setup_complete
rm -f /workspace/.setup_in_progress

print_status "🎉 Setup finished! Starting LLM Proxy service..."

# Start the service immediately - NO RETURN TO SHELL!
print_info "🌐 Starting FastAPI server on http://0.0.0.0:8000"

# CRITICAL: Activate virtual environment before starting
if [ -f "/workspace/venv/bin/activate" ]; then
    print_info "Activating virtual environment..."
    source /workspace/venv/bin/activate
    print_status "Virtual environment activated"
fi

# Use exec to replace the shell process - this prevents loops!
exec python3 main.py
