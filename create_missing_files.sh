#!/bin/bash
# create_missing_files.sh - Create Missing Files for React Integration

set -e

echo "🔧 CREATING Missing Files for React Integration"
echo "==============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Step 1: Check and show current directory contents
echo -e "\n${BLUE}📁 Current Directory Contents:${NC}"
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la *.py 2>/dev/null || echo "No Python files found"

# Step 2: Create main_with_react.py if it doesn't exist
if [ ! -f "main_with_react.py" ]; then
    echo -e "\n${BLUE}📄 Creating main_with_react.py${NC}"
    
    cat > main_with_react.py << 'EOF'
# main_with_react.py - FastAPI with React Integration (SIMPLIFIED)

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Try to import from main_fixed.py if it exists
try:
    from main_fixed import (
        settings, FixedOllamaClient, FixedLLMRouter, FixedMetrics,
        ChatCompletionRequest, CompletionRequest, ChatCompletionResponse, HealthResponse,
        Message, get_current_user
    )
    print("✅ Imported from main_fixed.py")
except ImportError:
    print("⚠️ main_fixed.py not found, using basic configuration")
    
    # Basic fallback configuration
    from pydantic_settings import BaseSettings
    from pydantic import BaseModel
    from typing import List
    
    class BasicSettings(BaseSettings):
        model_config = {"extra": "ignore"}
        DEBUG: bool = False
        HOST: str = "0.0.0.0"
        PORT: int = 8001
        LOG_LEVEL: str = "INFO"
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        OLLAMA_TIMEOUT: int = 300
        DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"
        ENABLE_AUTH: bool = False
        DEFAULT_API_KEY: str = "sk-dev-key"
        API_KEY_HEADER: str = "X-API-Key"
        CORS_ORIGINS: list = ["*"]
        CORS_ALLOW_CREDENTIALS: bool = True
    
    settings = BasicSettings()
    
    # Basic models
    class Message(BaseModel):
        role: str
        content: str
    
    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[Message]
        temperature: float = 0.7
        max_tokens: Optional[int] = None
        stream: bool = False
        top_p: float = 1.0
    
    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        timestamp: str
        version: str = "2.2.0"
        services: List[Dict[str, Any]] = []
    
    # Basic Ollama client
    import aiohttp
    
    class FixedOllamaClient:
        def __init__(self, base_url: str, timeout: int = 300):
            self.base_url = base_url.rstrip('/')
            self.timeout = timeout
            self.session = None
            self.stats = {'total_requests': 0, 'successful_requests': 0}
        
        async def initialize(self):
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        async def health_check(self) -> bool:
            try:
                if not self.session:
                    await self.initialize()
                async with self.session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
            except:
                return False
        
        async def list_models(self):
            try:
                if not self.session:
                    await self.initialize()
                async with self.session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('models', [])
                    return []
            except:
                return []
        
        def get_stats(self):
            return self.stats
        
        async def cleanup(self):
            if self.session:
                await self.session.close()
    
    # Basic router
    class FixedLLMRouter:
        def __init__(self, ollama_client):
            self.ollama_client = ollama_client
            self.available_models = {"gpt-3.5-turbo": {"priority": 1}}
        
        async def initialize(self):
            pass
        
        async def route_request(self, request) -> str:
            return "gpt-3.5-turbo"
        
        async def get_available_models(self):
            return [{"id": "gpt-3.5-turbo", "object": "model", "created": int(datetime.now().timestamp())}]
    
    # Basic metrics
    class FixedMetrics:
        def __init__(self):
            self.start_time = datetime.now()
            self.request_counts = {}
        
        async def get_all_metrics(self):
            return {
                "timestamp": datetime.now().isoformat(),
                "version": "2.2.0-basic",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "requests": {"total": sum(self.request_counts.values())},
                "status": "basic"
            }
        
        def track_request(self, endpoint: str, response_time: float = 0):
            self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
    
    # Basic auth
    async def get_current_user(request: Request):
        return {"user_id": "anonymous", "permissions": ["read", "write"]}

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Global service instances
ollama_client = None
llm_router = None
metrics_collector = None

async def initialize_services():
    """Initialize all services"""
    global ollama_client, llm_router, metrics_collector
    
    try:
        logging.info("🚀 Initializing services...")
        
        # Initialize metrics
        metrics_collector = FixedMetrics()
        
        # Initialize Ollama client
        ollama_client = FixedOllamaClient(settings.OLLAMA_BASE_URL, settings.OLLAMA_TIMEOUT)
        await ollama_client.initialize()
        
        # Initialize router
        llm_router = FixedLLMRouter(ollama_client)
        await llm_router.initialize()
        
        logging.info("✅ All services initialized")
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")

# Create FastAPI app
app = FastAPI(
    title="Enhanced LLM Proxy with React Dashboard",
    description="FastAPI backend with integrated React frontend",
    version="2.2.0-react",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount React build directory if it exists
react_build_dir = Path(__file__).parent / "frontend" / "build"
if react_build_dir.exists():
    # Mount static files
    app.mount("/app/static", StaticFiles(directory=react_build_dir / "static"), name="static")
    
    # Serve React app at /app
    @app.get("/app/{path:path}")
    async def serve_react_app(path: str = ""):
        """Serve React app with proper SPA routing"""
        
        # If it's a file request, try to serve it
        if path and "." in path:
            file_path = react_build_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
        
        # For all other routes, serve index.html (SPA routing)
        index_path = react_build_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            raise HTTPException(status_code=404, detail="React app not built")
    
    logging.info(f"✅ React app mounted at /app from {react_build_dir}")
else:
    # Fallback route when React app is not built
    @app.get("/app")
    @app.get("/app/{path:path}")
    async def react_not_built():
        return JSONResponse({
            "message": "React dashboard not built yet",
            "instructions": [
                "cd frontend",
                "npm install",
                "npm run build"
            ],
            "build_path": str(react_build_dir),
            "status": "not_built"
        })
    
    logging.warning(f"⚠️ React build directory not found: {react_build_dir}")

# Startup event
@app.on_event("startup")
async def startup_event():
    await initialize_services()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    if ollama_client:
        await ollama_client.cleanup()

# Basic API Routes
@app.get("/health")
async def health_check():
    try:
        services_status = []
        
        if ollama_client:
            try:
                ollama_healthy = await ollama_client.health_check()
                services_status.append({
                    "name": "ollama",
                    "status": "healthy" if ollama_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat()
                })
            except Exception as e:
                services_status.append({
                    "name": "ollama",
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "error": str(e)
                })
        
        # Add React app status
        services_status.append({
            "name": "react_dashboard",
            "status": "healthy" if react_build_dir.exists() else "not_built",
            "last_check": datetime.now().isoformat(),
            "build_path": str(react_build_dir)
        })
        
        overall_healthy = all(s["status"] in ["healthy", "not_built"] for s in services_status)
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0-react",
            "services": services_status
        }
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/models")
async def list_available_models():
    try:
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        models = await llm_router.get_available_models()
        return {"object": "list", "data": models}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        if metrics_collector:
            return await metrics_collector.get_all_metrics()
        else:
            return {
                "status": "basic_metrics",
                "timestamp": datetime.now().isoformat(),
                "message": "Basic metrics only"
            }
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/admin/status")
async def get_admin_status():
    return {
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0-react",
        "services": {
            "ollama_client": ollama_client is not None,
            "llm_router": llm_router is not None,
            "metrics_collector": metrics_collector is not None,
            "react_dashboard": react_build_dir.exists()
        },
        "enhanced_capabilities": {
            "streaming": False,
            "model_warmup": False,
            "semantic_classification": False,
            "react_dashboard": True
        },
        "configuration": {
            "dashboard_path": "/app",
            "react_build_exists": react_build_dir.exists()
        }
    }

@app.get("/")
async def root():
    return {
        "name": "Enhanced LLM Proxy with React Dashboard",
        "version": "2.2.0-react",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "models": "/models",
            "dashboard": "/app",
            "docs": "/docs"
        },
        "dashboard": {
            "url": "/app",
            "built": react_build_dir.exists(),
            "build_instructions": [
                "cd frontend",
                "npm install", 
                "npm run build"
            ] if not react_build_dir.exists() else None
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main_with_react:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
EOF
    
    print_status "main_with_react.py created"
else
    print_status "main_with_react.py already exists"
fi

# Step 3: Check for main_fixed.py
if [ ! -f "main_fixed.py" ]; then
    echo -e "\n${BLUE}📄 Creating basic main_fixed.py${NC}"
    
    cat > main_fixed.py << 'EOF'
# main_fixed.py - Basic Fixed Version

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Settings(BaseSettings):
    model_config = {"extra": "ignore"}
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

settings = Settings()

app = FastAPI(title="Basic LLM Proxy", version="2.2.0-basic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "healthy": True,
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0-basic"
    }

@app.get("/")
async def root():
    return {
        "message": "Basic LLM Proxy",
        "version": "2.2.0-basic",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("main_fixed:app", host=settings.HOST, port=settings.PORT)
EOF
    
    print_status "main_fixed.py created"
fi

# Step 4: Create updated build script that checks for files
echo -e "\n${BLUE}🚀 Creating Updated Build Script${NC}"

cat > build_and_start_simple.sh << 'EOF'
#!/bin/bash
# build_and_start_simple.sh - Simplified Build and Start

set -e

echo "🚀 Building React Dashboard and Starting FastAPI"
echo "================================================"

# Get current directory
CURRENT_DIR=$(pwd)
echo "📁 Current directory: $CURRENT_DIR"

# Check for Python files
echo "🐍 Available Python files:"
ls -la *.py 2>/dev/null || echo "No Python files found"

# Load port configuration
if [ -f .env.port ]; then
    source .env.port
    echo "🔧 Using port from .env.port: $PORT"
else
    PORT=8001
    echo "🔧 Using default port: $PORT"
fi

# Step 1: Build React app if frontend directory exists
if [ -d "frontend" ]; then
    echo ""
    echo "📦 Building React Dashboard..."
    cd frontend
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing React dependencies..."
        npm install --legacy-peer-deps || npm install
    fi
    
    # Build the app
    echo "Building React app..."
    if npm run build; then
        echo "✅ React build successful"
    else
        echo "⚠️ React build failed, but continuing..."
    fi
    
    cd ..
else
    echo "⚠️ Frontend directory not found, skipping React build"
fi

# Step 2: Determine which Python file to use
PYTHON_FILE=""

if [ -f "main_with_react.py" ]; then
    PYTHON_FILE="main_with_react.py"
    echo "✅ Using main_with_react.py"
elif [ -f "main_fixed.py" ]; then
    PYTHON_FILE="main_fixed.py"
    echo "✅ Using main_fixed.py"
elif [ -f "main.py" ]; then
    PYTHON_FILE="main.py"
    echo "✅ Using main.py"
else
    echo "❌ No suitable Python main file found!"
    echo ""
    echo "Available files:"
    ls -la *.py 2>/dev/null || echo "No Python files"
    echo ""
    echo "Please ensure you have one of these files:"
    echo "  • main_with_react.py (recommended)"
    echo "  • main_fixed.py"
    echo "  • main.py"
    exit 1
fi

# Step 3: Start the FastAPI application
echo ""
echo "🌐 Starting FastAPI server..."
echo "📊 Server info:"
echo "  • File: $PYTHON_FILE"
echo "  • Host: 0.0.0.0"
echo "  • Port: $PORT"
echo ""
echo "🎯 Access points:"
echo "  • Main API: http://localhost:$PORT"
echo "  • Health Check: http://localhost:$PORT/health"
echo "  • API Documentation: http://localhost:$PORT/docs"
if [ -d "frontend/build" ]; then
    echo "  • React Dashboard: http://localhost:$PORT/app"
else
    echo "  • React Dashboard: Not built yet"
fi
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
export PORT=$PORT
python "$PYTHON_FILE"
EOF

chmod +x build_and_start_simple.sh

print_status "Simplified build script created"

# Step 5: Create a quick test script
echo -e "\n${BLUE}🧪 Creating Quick Test Script${NC}"

cat > quick_test.sh << 'EOF'
#!/bin/bash
# quick_test.sh - Quick Test of the Setup

set -e

# Load port
if [ -f .env.port ]; then
    source .env.port
else
    PORT=8001
fi

BASE_URL="http://localhost:$PORT"

echo "🧪 Quick Test of Setup"
echo "====================="
echo "Testing: $BASE_URL"
echo ""

# Wait a moment for server to start
sleep 2

# Test health endpoint
echo "1. Testing health endpoint..."
if curl -s -f "$BASE_URL/health" >/dev/null 2>&1; then
    echo "✅ Health endpoint working"
    curl -s "$BASE_URL/health" | grep -o '"status":"[^"]*"' || echo "Response received"
else
    echo "❌ Health endpoint failed"
fi

echo ""
echo "2. Testing root endpoint..."
if curl -s -f "$BASE_URL/" >/dev/null 2>&1; then
    echo "✅ Root endpoint working"
else
    echo "❌ Root endpoint failed"
fi

echo ""
echo "3. Testing dashboard..."
if [ -d "frontend/build" ]; then
    if curl -s -f "$BASE_URL/app" >/dev/null 2>&1; then
        echo "✅ Dashboard accessible"
    else
        echo "⚠️ Dashboard not accessible"
    fi
else
    echo "⚠️ Dashboard not built yet"
fi

echo ""
echo "🎉 Quick test complete!"
echo ""
echo "🌐 Access your service at:"
echo "  $BASE_URL"
EOF

chmod +x quick_test.sh

print_status "Quick test script created"

# Step 6: Show current status and next steps
echo -e "\n${BLUE}📋 Current Status${NC}"

echo ""
echo "📁 Files in current directory:"
ls -la *.py *.sh 2>/dev/null || echo "No files found"

echo ""
echo "✅ Created files:"
echo "  • main_with_react.py - FastAPI with React integration"
echo "  • build_and_start_simple.sh - Simplified build script"
echo "  • quick_test.sh - Quick test script"

if [ -f "main_fixed.py" ]; then
    echo "  • main_fixed.py - Basic FastAPI version"
fi

echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Build and start everything:"
echo "   ./build_and_start_simple.sh"
echo ""
echo "2. Test in another terminal:"
echo "   ./quick_test.sh"
echo ""
echo "3. Access your service:"
if [ -f .env.port ]; then
    source .env.port
    echo "   http://localhost:${PORT:-8001}"
else
    echo "   http://localhost:8001"
fi

print_status "Setup complete! Ready to run: ./build_and_start_simple.sh"
