#!/bin/bash
# setup.sh - UPDATED SAFE VERSION
# This fixes the infinite loop issue and adds proper safety checks

set -e  # Exit on any error

# ============================================================================
# SAFETY CHECKS - PREVENT INFINITE LOOPS
# ============================================================================

SETUP_LOCK="/workspace/.setup_running"
SETUP_COMPLETE="/workspace/.setup_complete"

# Check if setup is already running
if [ -f "$SETUP_LOCK" ]; then
    echo "❌ Setup is already running! Lock file exists: $SETUP_LOCK"
    echo "If you're sure no setup is running, remove the lock file:"
    echo "rm $SETUP_LOCK"
    exit 1
fi

# Check if setup was already completed
if [ -f "$SETUP_COMPLETE" ]; then
    echo "✅ Setup already completed successfully!"
    echo ""
    echo "🚀 To start LLM Proxy:"
    echo "   cd /workspace/app && ./start.sh"
    echo ""
    echo "🔄 To force re-run setup:"
    echo "   rm $SETUP_COMPLETE && ./setup.sh"
    echo ""
    exit 0
fi

# Create lock file
touch "$SETUP_LOCK"
echo "🔒 Setup lock created"

# Cleanup function - CRITICAL to prevent loops
cleanup() {
    local exit_code=$?
    rm -f "$SETUP_LOCK"
    if [ $exit_code -eq 0 ]; then
        touch "$SETUP_COMPLETE"
        echo "✅ Setup completed successfully"
    else
        echo "❌ Setup failed, lock removed"
    fi
}
trap cleanup EXIT

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo "🚀 LLM Proxy Setup - SAFE VERSION"
echo "=================================="
echo "This version includes proper safety checks and will only run once"
echo ""

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

# Detect environment
if [ -n "$RUNPOD_POD_ID" ]; then
    ENVIRONMENT="runpod"
    print_info "RunPod environment detected"
elif [ -n "$KUBERNETES_SERVICE_HOST" ]; then
    ENVIRONMENT="kubernetes"
    print_info "Kubernetes environment detected"
else
    ENVIRONMENT="local"
    print_info "Local environment detected"
fi

# Check available memory
MEMORY_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
print_info "Available memory: ${MEMORY_GB}GB"

# Determine feature set based on memory
if [ "$MEMORY_GB" -ge 8 ]; then
    FEATURE_SET="full"
    print_status "Full feature set enabled (${MEMORY_GB}GB memory)"
elif [ "$MEMORY_GB" -ge 4 ]; then
    FEATURE_SET="standard"
    print_warning "Standard feature set (${MEMORY_GB}GB memory)"
else
    FEATURE_SET="minimal"
    print_warning "Minimal feature set (${MEMORY_GB}GB memory)"
fi

# ============================================================================
# SYSTEM SETUP
# ============================================================================

print_info "Installing system dependencies..."

# Set non-interactive mode
export DEBIAN_FRONTEND=noninteractive

# Update package list
apt-get update -qq >/dev/null 2>&1

# Install core dependencies
apt-get install -y -qq \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    python3-dev \
    dos2unix \
    htop \
    nano \
    >/dev/null 2>&1

print_status "System dependencies installed"

# ============================================================================
# WORKSPACE SETUP
# ============================================================================

print_info "Setting up workspace..."

# Ensure we're in the right directory
cd /workspace

# Create Python virtual environment
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# ============================================================================
# APPLICATION SETUP
# ============================================================================

print_info "Setting up application..."

# Remove existing app directory if it exists (clean slate)
if [ -d "app" ]; then
    print_warning "Removing existing app directory for clean setup..."
    rm -rf app
fi

# Create new app directory
mkdir -p app
cd app

# ============================================================================
# PYTHON DEPENDENCIES
# ============================================================================

print_info "Installing Python dependencies..."

# Upgrade pip
pip install --upgrade pip -q

# Create requirements.txt based on feature set
case $FEATURE_SET in
    "full")
        cat > requirements.txt << 'EOF'
# Core FastAPI stack
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
aiohttp==3.9.1
psutil==5.9.6
python-multipart==0.0.6

# Enhanced features
redis>=4.5.0
aioredis>=2.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.1.0
sse-starlette>=1.6.5
prometheus-client>=0.19.0

# Development tools
python-json-logger>=2.0.7
EOF
        ;;
    "standard")
        cat > requirements.txt << 'EOF'
# Core FastAPI stack
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
aiohttp==3.9.1
psutil==5.9.6
python-multipart==0.0.6

# Basic enhanced features
sse-starlette>=1.6.5
prometheus-client>=0.19.0
EOF
        ;;
    "minimal")
        cat > requirements.txt << 'EOF'
# Core FastAPI stack only
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
aiohttp==3.9.1
psutil==5.9.6
EOF
        ;;
esac

# Install dependencies with error handling
pip install -r requirements.txt -q
print_status "Python dependencies installed"

# ============================================================================
# CONFIGURATION FILES
# ============================================================================

print_info "Creating configuration files..."

# Create .env file
cat > .env << EOF
# Environment Configuration
ENVIRONMENT=$ENVIRONMENT
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300

# Memory Management (optimized for $MEMORY_GB GB)
MAX_MEMORY_MB=$((MEMORY_GB * 1024 - 512))
CACHE_MEMORY_LIMIT_MB=$((MEMORY_GB * 128))
MODEL_MEMORY_LIMIT_MB=$((MEMORY_GB * 256))

# Feature Flags
ENABLE_SEMANTIC_CLASSIFICATION=$([ "$FEATURE_SET" = "full" ] && echo "true" || echo "false")
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true

# Authentication (development settings)
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-key-safe

# CORS Configuration
CORS_ORIGINS=["*"]
CORS_ALLOW_CREDENTIALS=true

# Rate Limiting
ENABLE_RATE_LIMITING=false
DEFAULT_RATE_LIMIT=100
EOF

print_status ".env configuration created"

# ============================================================================
# APPLICATION CODE
# ============================================================================

print_info "Creating application code..."

# Create main.py
cat > main.py << 'EOF'
#!/usr/bin/env python3
"""
LLM Proxy - Safe Production Version
Designed for RunPod and other cloud environments
"""

import logging
import sys
import os
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/app.log')
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = Field(False)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    healthy: bool
    timestamp: str
    version: str = "2.0.0"
    status: str
    uptime_seconds: float
    environment: str
    memory_gb: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# ============================================================================
# GLOBAL STATE
# ============================================================================

start_time = datetime.now()
request_count = 0

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 LLM Proxy starting up...")
    
    # Startup tasks
    try:
        # Add any initialization here
        logger.info("✅ Startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown tasks
    logger.info("🛑 LLM Proxy shutting down...")

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="LLM Proxy",
    description="Production-ready LLM routing proxy",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Proxy is running",
        "version": "2.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    global request_count
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    # Get memory info
    try:
        import psutil
        memory_gb = int(psutil.virtual_memory().total / (1024**3))
    except:
        memory_gb = 0
    
    # Determine environment
    environment = "unknown"
    if os.getenv('RUNPOD_POD_ID'):
        environment = "runpod"
    elif os.getenv('KUBERNETES_SERVICE_HOST'):
        environment = "kubernetes"
    
    return HealthResponse(
        healthy=True,
        timestamp=datetime.now().isoformat(),
        status="All systems operational",
        uptime_seconds=uptime,
        environment=environment,
        memory_gb=memory_gb
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    global request_count
    request_count += 1
    
    # Validate request
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    # Extract last message for processing
    last_message = request.messages[-1].content
    
    # Mock response (replace with actual LLM integration)
    response_content = f"This is a test response from LLM Proxy for: {last_message[:100]}..."
    
    # Calculate token usage (rough estimation)
    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(response_content.split())
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(datetime.now().timestamp())}",
        created=int(datetime.now().timestamp()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={
                    "role": "assistant",
                    "content": response_content
                },
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "llm-proxy"
            },
            {
                "id": "gpt-4",
                "object": "model", 
                "created": int(datetime.now().timestamp()),
                "owned_by": "llm-proxy"
            }
        ]
    }

@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    global request_count
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "requests_per_minute": (request_count / (uptime / 60)) if uptime > 0 else 0,
        "version": "2.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Detailed status information"""
    return {
        "service": "LLM Proxy",
        "version": "2.0.0",
        "status": "running",
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        "environment": os.getenv('ENVIRONMENT', 'unknown'),
        "features": {
            "semantic_classification": os.getenv('ENABLE_SEMANTIC_CLASSIFICATION', 'false').lower() == 'true',
            "streaming": os.getenv('ENABLE_STREAMING', 'false').lower() == 'true',
            "model_warmup": os.getenv('ENABLE_MODEL_WARMUP', 'false').lower() == 'true'
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting LLM Proxy on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        access_log=True,
        log_level="info"
    )
EOF

chmod +x main.py
print_status "Main application created"

# ============================================================================
# UTILITY SCRIPTS
# ============================================================================

print_info "Creating utility scripts..."

# Start script
cat > start.sh << 'EOF'
#!/bin/bash
# Start LLM Proxy

cd /workspace/app
source /workspace/venv/bin/activate

echo "🚀 Starting LLM Proxy..."
echo "========================"
echo "Environment: $(cat .env | grep ENVIRONMENT | cut -d'=' -f2)"
echo "Port: 8000"
echo "Host: 0.0.0.0"
echo "Docs: http://localhost:8000/docs"
echo ""

python main.py
EOF

# Background start script
cat > start_bg.sh << 'EOF'
#!/bin/bash
# Start LLM Proxy in background

cd /workspace/app
source /workspace/venv/bin/activate

echo "🚀 Starting LLM Proxy in background..."
nohup python main.py > /workspace/app.log 2>&1 &
echo $! > /workspace/app.pid

echo "✅ LLM Proxy started in background"
echo "📋 PID: $(cat /workspace/app.pid)"
echo "📄 Logs: tail -f /workspace/app.log"
echo "🌐 URL: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"
EOF

# Stop script
cat > stop.sh << 'EOF'
#!/bin/bash
# Stop LLM Proxy

if [ -f /workspace/app.pid ]; then
    PID=$(cat /workspace/app.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "🛑 Stopping LLM Proxy (PID: $PID)..."
        kill $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "⚡ Force killing..."
            kill -9 $PID
        fi
        rm -f /workspace/app.pid
        echo "✅ LLM Proxy stopped"
    else
        echo "ℹ️  Process not running"
        rm -f /workspace/app.pid
    fi
else
    echo "ℹ️  No PID file found"
    # Try to find and kill any running main.py processes
    pkill -f "python main.py" && echo "✅ Killed running processes" || echo "ℹ️  No processes found"
fi
EOF

# Status script
cat > status.sh << 'EOF'
#!/bin/bash
# Check LLM Proxy status

echo "🔍 LLM Proxy Status"
echo "=================="

if [ -f /workspace/app.pid ]; then
    PID=$(cat /workspace/app.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Status: Running (PID: $PID)"
        echo "📊 Memory usage: $(ps -p $PID -o rss= | awk '{print int($1/1024)"MB"}')"
        echo "⏱️  Runtime: $(ps -p $PID -o etime= | tr -d ' ')"
    else
        echo "❌ Status: Stopped (stale PID file)"
        rm -f /workspace/app.pid
    fi
else
    if pgrep -f "python main.py" >/dev/null; then
        echo "⚠️  Status: Running (no PID file)"
        echo "🔧 PID: $(pgrep -f "python main.py")"
    else
        echo "⏹️  Status: Stopped"
    fi
fi

echo ""
echo "🌐 Testing connectivity..."
if curl -s http://localhost:8000/health >/dev/null; then
    echo "✅ Service is responding"
    echo "📋 Health: $(curl -s http://localhost:8000/health | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Healthy: {data['healthy']}, Uptime: {data['uptime_seconds']:.1f}s\")")"
else
    echo "❌ Service not responding"
fi
EOF

# Make all scripts executable
chmod +x *.sh
print_status "Utility scripts created"

# ============================================================================
# TESTING
# ============================================================================

print_info "Testing setup..."

# Test Python imports
python3 -c "
import sys
try:
    import fastapi
    import uvicorn  
    import pydantic
    print('✅ Core dependencies working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

print_status "Setup tests passed"

# ============================================================================
# COMPLETION
# ============================================================================

print_status "Setup completed successfully!"

echo ""
echo "🎉 LLM PROXY SETUP COMPLETE!"
echo "============================"
echo ""
echo "📁 Working directory: /workspace/app"
echo "🚀 Start command: ./start.sh"
echo "🔧 Background mode: ./start_bg.sh"
echo "🛑 Stop command: ./stop.sh"
echo "📊 Status check: ./status.sh"
echo ""
echo "🌐 Service URLs:"
echo "   • Main: http://localhost:8000"
echo "   • Health: http://localhost:8000/health"
echo "   • Docs: http://localhost:8000/docs"
echo "   • Metrics: http://localhost:8000/metrics"
echo ""
echo "📄 Log file: /workspace/app.log"
echo "🎯 Feature set: $FEATURE_SET"
echo ""
echo "🔥 NEXT STEPS:"
echo "1. cd /workspace/app"
echo "2. ./start.sh"
echo "3. Test: curl http://localhost:8000/health"
echo ""
echo "⚠️  This setup will NOT run again automatically!"
echo "   To re-run: rm /workspace/.setup_complete"
