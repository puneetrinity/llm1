# main.py - Enhanced LLM Proxy with 4 Key Features
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, Query, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import sys
import json
import secrets
import aiohttp
import asyncio
import psutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Configuration - Create a simple config if config.py doesn't exist
try:
    from config import settings
except ImportError:
    from pydantic_settings import BaseSettings
    
    class Settings(BaseSettings):
        # Server Settings
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        DEBUG: bool = False
        LOG_LEVEL: str = "INFO"
        
        # Ollama Settings - UPDATE THIS TO YOUR RUNPOD URL
        OLLAMA_BASE_URL: str = "https://your-pod-id-11434.proxy.runpod.net"
        OLLAMA_TIMEOUT: int = 300
        
        # Authentication
        ENABLE_AUTH: bool = True
        DEFAULT_API_KEY: str = "sk-your-secure-api-key-here"
        API_KEY_HEADER: str = "X-API-Key"
        
        # Features
        ENABLE_DASHBOARD: bool = True
        ENABLE_ENHANCED_FEATURES: bool = True
        ENABLE_WEBSOCKET: bool = True
        
        # CORS
        CORS_ORIGINS: List[str] = ["*"]
        CORS_ALLOW_CREDENTIALS: bool = True
        
        # Memory
        MAX_MEMORY_MB: int = 8192
        CACHE_MEMORY_LIMIT_MB: int = 1024
        
        class Config:
            env_file = ".env"
            case_sensitive = True
    
    settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Pydantic Models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0

class HealthResponse(BaseModel):
    status: str
    healthy: bool
    timestamp: str
    version: str
    services: Dict[str, Any]

class StatusResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    features: Dict[str, bool]
    timestamp: str

# Simple metrics storage
class SimpleMetrics:
    def __init__(self):
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.model_requests = {}
    
    def track_request(self, endpoint: str = "/"):
        self.request_count += 1
    
    def track_error(self):
        self.error_count += 1
    
    def track_response_time(self, time_ms: float):
        self.response_times.append(time_ms)
        if len(self.response_times) > 100:  # Keep only last 100
            self.response_times.pop(0)
    
    def track_model_request(self, model: str):
        self.model_requests[model] = self.model_requests.get(model, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = datetime.now() - self.start_time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "uptime_seconds": int(uptime.total_seconds()),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "avg_response_time_ms": round(avg_response_time, 2),
            "model_usage": self.model_requests,
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }

# Global service state
services_state = {
    "ollama_connected": False,
    "dashboard_available": False,
    "initialization_complete": False,
    "available_models": []
}

# Initialize simple metrics
simple_metrics = SimpleMetrics()

# WebSocket session storage
websocket_sessions = {}

# Ollama Client for RunPod
class OllamaClient:
    """Ollama client for RunPod connection"""
    
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info(f"✅ Ollama client initialized for {self.base_url}")
    
    async def health_check(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[Dict]:
        """List available models"""
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('models', [])
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def generate_completion(self, model: str, messages: List[Dict], **kwargs):
        """Generate completion using Ollama"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.session:
                await self.initialize()
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "top_p": kwargs.get('top_p', 1.0),
                    "num_predict": kwargs.get('max_tokens', 150)
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate", 
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
            
            # Update stats
            processing_time = asyncio.get_event_loop().time() - start_time
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            
            # Track metrics
            simple_metrics.track_model_request(model)
            simple_metrics.track_response_time(processing_time * 1000)  # Convert to ms
            
            # Return OpenAI-compatible response
            return {
                "id": f"chatcmpl-{int(start_time)}",
                "object": "chat.completion",
                "created": int(start_time),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.get('response', '')
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(result.get('response', '').split()),
                    "total_tokens": len(prompt.split()) + len(result.get('response', '').split())
                },
                "processing_time": processing_time,
                "selected_model": model
            }
            
        except Exception as e:
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            simple_metrics.track_error()
            logger.error(f"Generation failed: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to prompt format"""
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"Human: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return self.stats.copy()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

# Model Router for 3 Models
class ModelRouter:
    """Routes requests to the best model based on content"""
    
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        
        # Configuration for your 3 models
        self.model_config = {
            'mistral:7b-instruct-q4_0': {
                'priority': 1,
                'good_for': ['factual', 'general', 'math', 'analysis'],
                'description': 'General purpose model, good for factual questions'
            },
            'deepseek-v2:7b-q4_0': {
                'priority': 2,
                'good_for': ['coding', 'technical', 'programming', 'debug'],
                'description': 'Specialized for coding and technical tasks'
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 3,
                'good_for': ['creative', 'storytelling', 'writing', 'chat'],
                'description': 'Best for creative writing and conversational tasks'
            }
        }
        
        self.available_models = {}
        self.default_model = 'mistral:7b-instruct-q4_0'
    
    async def initialize(self):
        """Initialize router and check available models"""
        try:
            # Get models from Ollama
            models = await self.ollama_client.list_models()
            available_model_names = {model.get('name', '') for model in models}
            
            # Filter to only configured models that are available
            self.available_models = {
                name: config for name, config in self.model_config.items()
                if name in available_model_names
            }
            
            if not self.available_models:
                # Fallback: use any available model
                if models:
                    fallback_model = models[0].get('name', self.default_model)
                    self.available_models[fallback_model] = {
                        'priority': 1,
                        'good_for': ['general'],
                        'description': 'Fallback model'
                    }
                else:
                    # Emergency fallback
                    self.available_models = {self.default_model: self.model_config[self.default_model]}
            
            logger.info(f"✅ Router initialized with models: {list(self.available_models.keys())}")
            services_state["available_models"] = list(self.available_models.keys())
            
        except Exception as e:
            logger.error(f"Router initialization failed: {e}")
            # Emergency fallback
            self.available_models = {self.default_model: self.model_config[self.default_model]}
    
    def select_model(self, request: ChatCompletionRequest) -> str:
        """Select the best model for the request"""
        # If specific model requested and available, use it
        if request.model in self.available_models:
            return request.model
        
        # Analyze content to choose best model
        text_content = " ".join([msg.content for msg in request.messages])
        text_lower = text_content.lower()
        
        # Model selection logic
        if any(word in text_lower for word in ['code', 'function', 'program', 'debug', 'script']):
            # Prefer DeepSeek for coding
            if 'deepseek-v2:7b-q4_0' in self.available_models:
                return 'deepseek-v2:7b-q4_0'
        
        elif any(word in text_lower for word in ['story', 'creative', 'write', 'poem', 'chat']):
            # Prefer Llama3 for creative tasks
            if 'llama3:8b-instruct-q4_0' in self.available_models:
                return 'llama3:8b-instruct-q4_0'
        
        # Default to Mistral for factual/general queries
        if 'mistral:7b-instruct-q4_0' in self.available_models:
            return 'mistral:7b-instruct-q4_0'
        
        # Fallback to first available model
        return list(self.available_models.keys())[0]
    
    async def process_completion(self, request: ChatCompletionRequest):
        """Process completion request"""
        # Select model
        selected_model = self.select_model(request)
        
        # Convert messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate completion
        result = await self.ollama_client.generate_completion(
            model=selected_model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )
        
        return result
    
    async def get_available_models(self):
        """Get available models with metadata"""
        models = []
        for model_name, config in self.available_models.items():
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "ollama",
                "description": config.get('description', ''),
                "capabilities": config.get('good_for', [])
            })
        return models

# Global instances
ollama_client = None
model_router = None

async def initialize_services():
    """Initialize all services"""
    global services_state, ollama_client, model_router
    
    try:
        logger.info("🚀 Initializing services...")
        
        # Initialize Ollama client
        ollama_client = OllamaClient(settings.OLLAMA_BASE_URL, settings.OLLAMA_TIMEOUT)
        await ollama_client.initialize()
        
        # Check Ollama connection
        services_state["ollama_connected"] = await ollama_client.health_check()
        
        # Initialize model router
        model_router = ModelRouter(ollama_client)
        await model_router.initialize()
        
        # Check dashboard
        react_build_dir = Path(__file__).parent / "frontend" / "build"
        services_state["dashboard_available"] = (
            react_build_dir.exists() and 
            (react_build_dir / "index.html").exists()
        )
        
        # If no React dashboard, check for static dashboard
        if not services_state["dashboard_available"]:
            static_dir = Path(__file__).parent / "static" / "dashboard"
            services_state["dashboard_available"] = (
                static_dir.exists() and 
                (static_dir / "index.html").exists()
            )
        
        services_state["initialization_complete"] = True
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        services_state["initialization_complete"] = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🌟 Starting Enhanced LLM Proxy...")
    await initialize_services()
    
    # Log startup summary
    logger.info("=" * 60)
    logger.info(f"🎯 Server: {settings.HOST}:{settings.PORT}")
    logger.info(f"🔗 Ollama: {settings.OLLAMA_BASE_URL}")
    logger.info(f"🐛 Debug Mode: {settings.DEBUG}")
    logger.info(f"📊 Services: {services_state}")
    logger.info(f"🤖 Models: {services_state.get('available_models', [])}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    if ollama_client:
        await ollama_client.cleanup()

# ENHANCEMENT 3: Debug-mode docs/redoc handling (for security)
docs_url = "/docs" if settings.DEBUG else None
redoc_url = "/redoc" if settings.DEBUG else None

# Create FastAPI app
app = FastAPI(
    title="Enhanced LLM Proxy",
    description="OpenAI-compatible API with 3-model routing, authentication, and enhanced features",
    version="3.1.0-enhanced",
    lifespan=lifespan,
    docs_url=docs_url,
    redoc_url=redoc_url
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Dashboard static files setup
dashboard_dir = None
if settings.ENABLE_DASHBOARD:
    # Try React dashboard first
    react_build_dir = Path(__file__).parent / "frontend" / "build"
    static_dir = Path(__file__).parent / "static" / "dashboard"
    
    if react_build_dir.exists() and (react_build_dir / "index.html").exists():
        dashboard_dir = react_build_dir
        logger.info(f"📊 Using React dashboard from {dashboard_dir}")
    elif static_dir.exists() and (static_dir / "index.html").exists():
        dashboard_dir = static_dir
        logger.info(f"📊 Using static dashboard from {dashboard_dir}")
    
    if dashboard_dir:
        # Mount static files
        app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = asyncio.get_event_loop().time()
    simple_metrics.track_request(request.url.path)
    
    response = await call_next(request)
    
    process_time = (asyncio.get_event_loop().time() - start_time) * 1000
    simple_metrics.track_response_time(process_time)
    
    return response

# Authentication
async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Authentication dependency"""
    if not settings.ENABLE_AUTH:
        return {"user_id": "anonymous", "permissions": ["read", "write"]}
    
    api_key = request.headers.get(settings.API_KEY_HEADER)
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "API key required",
                "message": f"Please provide API key in {settings.API_KEY_HEADER} header"
            }
        )
    
    if api_key != settings.DEFAULT_API_KEY:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is not valid"
            }
        )
    
    return {"user_id": "authenticated", "permissions": ["read", "write"]}

# WebSocket session management
@app.post("/auth/websocket-session")
async def create_websocket_session(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a session token for WebSocket authentication"""
    try:
        session_token = secrets.token_urlsafe(32)
        websocket_sessions[session_token] = {
            "user_id": current_user["user_id"],
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24)
        }
        
        return {
            "session_token": session_token,
            "expires_in": 24 * 3600,
            "websocket_url": f"ws://{settings.HOST}:{settings.PORT}/ws/dashboard?session={session_token}"
        }
    except Exception as e:
        logger.error(f"Failed to create WebSocket session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket, session: str = Query(None)):
    """WebSocket endpoint for dashboard"""
    
    # Authentication
    if settings.ENABLE_AUTH:
        if not session or session not in websocket_sessions:
            await websocket.close(code=1008, reason="Invalid session")
            return
        
        session_data = websocket_sessions[session]
        if session_data["expires_at"] < datetime.now():
            del websocket_sessions[session]
            await websocket.close(code=1008, reason="Session expired")
            return
    
    await websocket.accept()
    logger.info("🔌 WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
            elif message.get("type") == "request_update":
                # Send system status
                status_data = {
                    "timestamp": datetime.now().isoformat(),
                    "services": services_state,
                    "models": services_state.get("available_models", []),
                    "stats": ollama_client.get_stats() if ollama_client else {},
                    "metrics": simple_metrics.get_stats()
                }
                await websocket.send_text(json.dumps({
                    "type": "dashboard_update",
                    "data": status_data
                }))
            
            elif message.get("type") == "chat":
                # Handle chat message
                if model_router:
                    try:
                        # Create chat completion request
                        chat_request = ChatCompletionRequest(
                            model=message.get("model", "mistral:7b-instruct-q4_0"),
                            messages=[Message(role="user", content=message.get("content", ""))]
                        )
                        
                        # Get response
                        result = await model_router.process_completion(chat_request)
                        
                        await websocket.send_text(json.dumps({
                            "type": "chat_response",
                            "data": result
                        }))
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))
                        
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# API Endpoints

# ENHANCEMENT 4: Explicit root endpoint (self-documenting API)
@app.get("/")
async def root():
    """Enhanced root endpoint with comprehensive API documentation"""
    return {
        "service": "Enhanced LLM Proxy",
        "version": "3.1.0-enhanced",
        "timestamp": datetime.now().isoformat(),
        "status": "online" if services_state["initialization_complete"] else "initializing",
        "description": "OpenAI-compatible API with 3-model routing, authentication, and enhanced features",
        
        # Core API endpoints
        "endpoints": {
            "chat_completions": {
                "url": "/v1/chat/completions",
                "method": "POST",
                "description": "OpenAI-compatible chat completions",
                "auth_required": settings.ENABLE_AUTH
            },
            "models": {
                "url": "/v1/models",
                "method": "GET",
                "description": "List available models",
                "auth_required": False
            },
            "health": {
                "url": "/health",
                "method": "GET",
                "description": "Service health check",
                "auth_required": False
            },
            "metrics": {
                "url": "/metrics",
                "method": "GET",
                "description": "Simple system metrics",
                "auth_required": False
            },
            "dashboard": {
                "url": "/app",
                "method": "GET",
                "description": "Web dashboard interface",
                "auth_required": False
            }
        },
        
        # Authentication info
        "authentication": {
            "enabled": settings.ENABLE_AUTH,
            "header": settings.API_KEY_HEADER if settings.ENABLE_AUTH else None,
            "websocket_auth": "/auth/websocket-session" if settings.ENABLE_WEBSOCKET else None
        },
        
        # Feature capabilities
        "features": {
            "model_routing": True,
            "streaming": True,
            "websocket": settings.ENABLE_WEBSOCKET,
            "dashboard": settings.ENABLE_DASHBOARD,
            "metrics": True,
            "debug_mode": settings.DEBUG
        },
        
        # Available models
        "models": services_state.get("available_models", []),
        
        # Service status
        "services": {
            "ollama": services_state.get("ollama_connected", False),
            "dashboard": services_state.get("dashboard_available", False),
            "initialization": services_state.get("initialization_complete", False)
        },
        
        # Documentation links
        "documentation": {
            "api_docs": "/docs" if settings.DEBUG else "Not available in production",
            "redoc": "/redoc" if settings.DEBUG else "Not available in production",
            "openapi_spec": "/openapi.json"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if services_state["initialization_complete"] else "initializing",
        healthy=services_state["initialization_complete"],
        timestamp=datetime.now().isoformat(),
        version="3.1.0-enhanced",
        services=services_state
    )

# ENHANCEMENT 2: Simple /metrics endpoint (system stats)
@app.get("/metrics")
async def metrics():
    """Simple metrics endpoint for monitoring and dashboard"""
    try:
        # Get basic system metrics
        stats = simple_metrics.get_stats()
        
        # Add service-specific metrics
        ollama_stats = ollama_client.get_stats() if ollama_client else {}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "enhanced-llm-proxy",
            "version": "3.1.0-enhanced",
            
            # Performance metrics
            "performance": {
                "uptime_seconds": stats["uptime_seconds"],
                "total_requests": stats["total_requests"],
                "requests_per_minute": stats["total_requests"] / max(1, stats["uptime_seconds"] / 60),
                "error_rate": stats["error_rate"],
                "avg_response_time_ms": stats["avg_response_time_ms"]
            },
            
            # System resources
            "system": stats["system"],
            
            # Model usage
            "models": {
                "available": services_state.get("available_models", []),
                "usage_counts": stats["model_usage"]
            },
            
            # Service health
            "services": {
                "ollama_connected": services_state.get("ollama_connected", False),
                "ollama_requests": ollama_stats.get("total_requests", 0),
                "ollama_success_rate": (
                    ollama_stats.get("successful_requests", 0) / 
                    max(1, ollama_stats.get("total_requests", 1))
                ),
                "dashboard_available": services_state.get("dashboard_available", False),
                "websocket_sessions": len(websocket_sessions)
            },
            
            # Feature flags
            "features": {
                "authentication": settings.ENABLE_AUTH,
                "debug_mode": settings.DEBUG,
                "dashboard": settings.ENABLE_DASHBOARD,
                "websocket": settings.ENABLE_WEBSOCKET
            }
        }
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return {
            "error": "Failed to generate metrics",
            "timestamp": datetime.now().isoformat(),
            "message": str(e)
        }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    try:
        if not model_router:
            raise HTTPException(status_code=503, detail="Model router not available")
        
        models = await model_router.get_available_models()
        return {"object": "list", "data": models}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """OpenAI-compatible chat completions endpoint"""
    
    try:
        if not model_router:
            raise HTTPException(status_code=503, detail="Model router not available")
        
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        # Process the completion
        result = await model_router.process_completion(request)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def api_status():
    """Detailed status for dashboard"""
    return StatusResponse(
        status="online" if services_state["initialization_complete"] else "starting",
        services=services_state,
        features={
            "authentication": settings.ENABLE_AUTH,
            "dashboard": settings.ENABLE_DASHBOARD,
            "websocket": settings.ENABLE_WEBSOCKET,
            "debug_mode": settings.DEBUG
        },
        timestamp=datetime.now().isoformat()
    )

# ENHANCEMENT 1: Catch-all SPA route (client-side routing)
if dashboard_dir and settings.ENABLE_DASHBOARD:
    @app.get("/app/{path:path}")
    async def serve_dashboard_spa(path: str = ""):
        """Serve dashboard with SPA routing support"""
        try:
            # Handle static files first
            if path and "." in path:
                file_path = dashboard_dir / path
                if file_path.exists() and file_path.is_file():
                    return FileResponse(file_path)
            
            # For all other routes, serve index.html (SPA routing)
            index_path = dashboard_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            else:
                return JSONResponse({
                    "error": "Dashboard not found",
                    "message": "React app not built yet",
                    "instructions": [
                        "cd frontend",
                        "npm install",
                        "npm run build"
                    ]
                })
                
        except Exception as e:
            logger.error(f"Dashboard serving error: {e}")
            return JSONResponse({
                "error": "Dashboard error",
                "message": str(e)
            })
    
    # Catch-all route for SPA (must be last)
    @app.get("/{path:path}")
    async def catch_all_spa(path: str):
        """Catch-all route for SPA client-side routing"""
        
        # Skip API routes and system paths
        api_prefixes = [
            "v1/", "api/", "health", "metrics", "models", 
            "docs", "redoc", "openapi.json", "ws/", 
            "auth/", "favicon.ico", "static/", "admin/"
        ]
        
        # Don't interfere with API routes
        if any(path.startswith(prefix) for prefix in api_prefixes):
            raise HTTPException(status_code=404, detail=f"Endpoint not found: {path}")
        
        # Serve SPA for all other routes
        index_path = dashboard_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            return JSONResponse({
                "error": "Dashboard not available",
                "message": "Please build the React dashboard first"
            })

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    simple_metrics.track_error()
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "status_code": 500,
                "detail": "Internal server error",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

if __name__ == "__main__":
    logger.info(f"🚀 Starting Enhanced LLM Proxy")
    logger.info(f"📍 Server: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📊 Dashboard: http://{settings.HOST}:{settings.PORT}/app")
    logger.info(f"🔌 WebSocket: ws://{settings.HOST}:{settings.PORT}/ws/dashboard")
    logger.info(f"📈 Metrics: http://{settings.HOST}:{settings.PORT}/metrics")
    logger.info(f"🐛 Debug Mode: {settings.DEBUG}")
    
    if settings.DEBUG:
        logger.info(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info(f"📖 ReDoc: http://{settings.HOST}:{settings.PORT}/redoc")
    else:
        logger.info("🔒 API documentation disabled in production mode")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
