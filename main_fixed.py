# main_fixed.py - FIXED Async/Await and JSON Serialization Issues
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import logging
import sys
import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# FIXED: JSON serialization for datetime objects
def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# FIXED: Safe configuration loading
try:
    from config_enhanced import get_settings
    settings = get_settings()
    logging.info("✅ Enhanced configuration loaded")
except ImportError as e:
    logging.warning(f"Enhanced config not available: {e}")
    from pydantic_settings import BaseSettings
    
    class ComprehensiveSettings(BaseSettings):
        model_config = {"extra": "ignore"}
        
        DEBUG: bool = False
        HOST: str = "0.0.0.0"
        PORT: int = 8001  # FIXED: Changed default port to avoid conflicts
        LOG_LEVEL: str = "INFO"
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        OLLAMA_TIMEOUT: int = 300
        DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"
        ENABLE_AUTH: bool = False
        DEFAULT_API_KEY: str = "sk-dev-key-change-in-production"
        API_KEY_HEADER: str = "X-API-Key"
        CORS_ORIGINS: list = ["*"]
        CORS_ALLOW_CREDENTIALS: bool = True
        MAX_MEMORY_MB: int = 8192
        CACHE_MEMORY_LIMIT_MB: int = 1024
        MODEL_MEMORY_LIMIT_MB: int = 4096
        ENABLE_SEMANTIC_CLASSIFICATION: bool = False
        ENABLE_STREAMING: bool = True
        ENABLE_MODEL_WARMUP: bool = True
        ENABLE_DETAILED_METRICS: bool = True
        ENABLE_DASHBOARD: bool = True
        DASHBOARD_PATH: str = "/app"
    
    settings = ComprehensiveSettings()
    logging.info("✅ Comprehensive configuration loaded")

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# FIXED: Import models with fallback
try:
    from models.requests import ChatCompletionRequest, CompletionRequest
    from models.responses import ChatCompletionResponse, HealthResponse
    logging.info("✅ Custom models loaded")
except ImportError:
    logging.warning("Custom models not available - using basic models")
    from pydantic import BaseModel
    from typing import List
    
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
        intent: Optional[str] = None
        priority: str = "normal"
    
    class CompletionRequest(BaseModel):
        model: str
        prompt: str
        temperature: float = 0.7
        max_tokens: Optional[int] = None
        stream: bool = False
        top_p: float = 1.0
    
    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[Dict[str, Any]]
        usage: Dict[str, Any]
        cache_hit: bool = False
        processing_time: Optional[float] = None
        selected_model: Optional[str] = None
    
    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        timestamp: str
        version: str = "2.2.0"
        services: List[Dict[str, Any]] = []

# Global service instances
ollama_client = None
llm_router = None
metrics_collector = None
health_checker = None
websocket_dashboard = None
enhanced_capabilities = {}

# FIXED: Ollama client with proper async/await handling
import aiohttp

class FixedOllamaClient:
    """FIXED Ollama client with proper async/await handling"""
    
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
    
    async def initialize(self):
        """Initialize with proper session management"""
        try:
            # Use basic aiohttp session with proper timeout
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logging.info("✅ Fixed Ollama client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    async def health_check(self) -> bool:
        """FIXED health check with proper async/await"""
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logging.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self):
        """FIXED list models with proper async/await"""
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('models', [])
                return []
        except Exception as e:
            logging.error(f"Failed to list models: {e}")
            return []
    
    async def generate_completion(self, model: str, messages: List[Dict], **kwargs):
        """FIXED generate completion with proper async/await"""
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
            
            # FIXED: Proper async request handling
            async with self.session.post(
                f"{self.base_url}/api/generate", 
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
            
            # Update stats
            processing_time = asyncio.get_event_loop().time() - start_time
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            
            # FIXED: Proper response format with JSON-serializable datetime
            return {
                "id": f"completion-{int(start_time)}",
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
            logging.error(f"Generation failed: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to prompt format"""
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
            elif role == 'system':
                prompt += f"System: {content}\n"
        prompt += "Assistant: "
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics - JSON serializable"""
        return self.stats.copy()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

# FIXED: Router with proper async handling
class FixedLLMRouter:
    """FIXED LLM router with proper async/await handling"""
    
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self.default_model = settings.DEFAULT_MODEL
        self.model_config = {
            'mistral:7b-instruct-q4_0': {
                'priority': 1,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'good_for': ['factual', 'math', 'general']
            },
            'deepseek-v2:7b-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00015,
                'max_context': 4096,
                'good_for': ['analysis', 'coding', 'resume']
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'good_for': ['creative', 'interview', 'storytelling']
            }
        }
        self.available_models = {}
    
    async def initialize(self):
        """FIXED initialization with proper async/await"""
        try:
            # Get available models with proper await
            models = await self.ollama_client.list_models()
            available_model_names = {model.get('name', '') for model in models}
            
            self.available_models = {
                name: config for name, config in self.model_config.items()
                if name in available_model_names
            }
            
            if not self.available_models:
                # Fallback to any available model
                if models:
                    fallback_model = models[0].get('name', self.default_model)
                    self.available_models[fallback_model] = {
                        'priority': 1,
                        'cost_per_token': 0.0001,
                        'max_context': 4096,
                        'good_for': ['general']
                    }
                else:
                    # Ultimate fallback
                    self.available_models = {self.default_model: {
                        'priority': 1, 'cost_per_token': 0.0001, 'max_context': 4096, 'good_for': ['general']
                    }}
            
            logging.info(f"✅ Fixed router initialized with models: {list(self.available_models.keys())}")
            
        except Exception as e:
            logging.error(f"Router initialization failed: {e}")
            # Emergency fallback
            self.available_models = {self.default_model: {
                'priority': 1, 'cost_per_token': 0.0001, 'max_context': 4096, 'good_for': ['general']
            }}
    
    async def route_request(self, request) -> str:
        """Route request with simple logic"""
        # Use explicit model if valid
        if hasattr(request, 'model') and request.model in self.available_models:
            return request.model
        
        # Default to first available model
        if self.available_models:
            return list(self.available_models.keys())[0]
        
        return self.default_model
    
    async def process_chat_completion(self, request, model: str):
        """FIXED process chat completion with proper async/await"""
        try:
            messages = []
            if hasattr(request, 'messages'):
                for msg in request.messages:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        messages.append({"role": msg.role, "content": msg.content})
                    elif isinstance(msg, dict):
                        messages.append(msg)
            
            # FIXED: Proper await
            return await self.ollama_client.generate_completion(
                model=model,
                messages=messages,
                temperature=getattr(request, 'temperature', 0.7),
                max_tokens=getattr(request, 'max_tokens', 150),
                top_p=getattr(request, 'top_p', 1.0)
            )
        except Exception as e:
            logging.error(f"Error processing chat completion: {e}")
            raise
    
    async def get_available_models(self):
        """Get available models with JSON-serializable format"""
        models = []
        for model_name, config in self.available_models.items():
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(datetime.now().timestamp()),  # FIXED: Use timestamp instead of datetime
                "owned_by": "ollama",
                "cost_per_token": config.get('cost_per_token', 0.0001),
                "max_context": config.get('max_context', 4096),
                "capabilities": config.get('good_for', [])
            })
        return models

# FIXED: Metrics collector with JSON-serializable responses
class FixedMetrics:
    """FIXED metrics collector with proper JSON serialization"""
    
    def __init__(self):
        self.start_time = datetime.now()  # Keep as datetime internally
        self.request_counts = {}
        self.response_times = []
        self.model_usage = {}
        self.errors = {}
    
    async def get_all_metrics(self):
        """FIXED: Get metrics with JSON-serializable format"""
        uptime = datetime.now() - self.start_time
        total_requests = sum(self.request_counts.values())
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "timestamp": datetime.now().isoformat(),  # FIXED: Convert to ISO string
            "version": "2.2.0",
            "uptime_seconds": uptime.total_seconds(),
            "requests": {
                "total": total_requests,
                "by_endpoint": self.request_counts,
                "avg_response_time": avg_response_time
            },
            "models": self.model_usage,
            "errors": self.errors,
            "enhanced_features": enhanced_capabilities,
            "status": "fixed"
        }
    
    def track_request(self, endpoint: str, response_time: float = 0):
        """Track request"""
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
        if response_time > 0:
            self.response_times.append(response_time)
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-500:]
    
    def track_model_usage(self, model: str):
        """Track model usage"""
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
    
    def track_error(self, error_type: str):
        """Track error"""
        self.errors[error_type] = self.errors.get(error_type, 0) + 1

# FIXED: Initialize services with proper error handling
async def initialize_fixed_services():
    """FIXED: Initialize services with proper async/await handling"""
    global ollama_client, llm_router, metrics_collector, enhanced_capabilities
    
    try:
        logging.info("🚀 Initializing FIXED services...")
        
        # Initialize metrics
        metrics_collector = FixedMetrics()
        logging.info("✅ Fixed metrics collector initialized")
        
        # Initialize Ollama client with proper async
        ollama_client = FixedOllamaClient(
            settings.OLLAMA_BASE_URL, 
            settings.OLLAMA_TIMEOUT
        )
        await ollama_client.initialize()  # FIXED: Proper await
        logging.info("✅ Fixed Ollama client initialized")
        
        # Initialize router with proper async
        llm_router = FixedLLMRouter(ollama_client)
        await llm_router.initialize()  # FIXED: Proper await
        logging.info("✅ Fixed LLM router initialized")
        
        # Set enhanced capabilities
        enhanced_capabilities = {
            "streaming": settings.ENABLE_STREAMING,
            "model_warmup": settings.ENABLE_MODEL_WARMUP,
            "semantic_classification": False,  # Disabled for now
            "enhanced_ollama": True,
            "enhanced_router": True
        }
        
        logging.info("✅ All FIXED services initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        logging.error(traceback.format_exc())
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FIXED: Application lifespan with proper async handling"""
    
    # Startup
    logging.info("🚀 Starting FIXED Enhanced LLM Proxy...")
    
    try:
        await initialize_fixed_services()  # FIXED: Proper await
        logging.info("✅ FIXED services initialized")
        log_startup_summary()
    except Exception as e:
        logging.error(f"❌ Failed to start services: {e}")
        # Don't fail completely - allow partial startup
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down FIXED Enhanced LLM Proxy...")
    
    try:
        if ollama_client:
            await ollama_client.cleanup()  # FIXED: Proper await
        logging.info("✅ Services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

def log_startup_summary():
    """Log startup information with fixed formatting"""
    logging.info("=" * 60)
    logging.info("🚀 FIXED ENHANCED LLM PROXY - STARTUP SUMMARY")
    logging.info("=" * 60)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Host: {settings.HOST}:{settings.PORT}")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"🎯 Core Services:")
    logging.info(f"   • Ollama Client: {'✅' if ollama_client else '❌'}")
    logging.info(f"   • LLM Router: {'✅' if llm_router else '❌'}")
    logging.info(f"   • Metrics: {'✅' if metrics_collector else '❌'}")
    logging.info(f"🚀 Enhanced Features:")
    for feature, enabled in enhanced_capabilities.items():
        status = "✅ Enabled" if enabled else "⏸️ Disabled"
        logging.info(f"   • {feature.replace('_', ' ').title()}: {status}")
    logging.info("=" * 60)

# Create FastAPI app
app = FastAPI(
    title="FIXED Enhanced LLM Proxy",
    description="FIXED version with proper async/await and JSON serialization",
    version="2.2.0-fixed",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Authentication dependency
async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get current user from request"""
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

# FIXED: Main API Routes with proper async/await
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """FIXED: Chat completions endpoint with proper async/await"""
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        # Track request
        if metrics_collector:
            metrics_collector.track_request("/v1/chat/completions")
        
        # Route request - FIXED: Proper await
        selected_model = await llm_router.route_request(request)
        
        # Process request - FIXED: Proper await
        response = await llm_router.process_chat_completion(request, selected_model)
        
        # Track metrics
        if metrics_collector:
            processing_time = asyncio.get_event_loop().time() - start_time
            metrics_collector.track_request("/v1/chat/completions", processing_time)
            metrics_collector.track_model_usage(selected_model)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        if metrics_collector:
            metrics_collector.track_error("chat_completion_error")
        logging.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """FIXED: Health check with JSON-serializable responses"""
    
    try:
        services_status = []
        
        # Check Ollama - FIXED: Proper await
        if ollama_client:
            try:
                ollama_healthy = await ollama_client.health_check()
                ollama_stats = ollama_client.get_stats()
                services_status.append({
                    "name": "ollama",
                    "status": "healthy" if ollama_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat(),  # FIXED: JSON serializable
                    "stats": ollama_stats
                })
            except Exception as e:
                services_status.append({
                    "name": "ollama",
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),  # FIXED: JSON serializable
                    "error": str(e)
                })
        
        # Check Router
        if llm_router:
            services_status.append({
                "name": "llm_router",
                "status": "healthy",
                "last_check": datetime.now().isoformat(),  # FIXED: JSON serializable
                "available_models": len(llm_router.available_models)
            })
        
        overall_healthy = all(s["status"] == "healthy" for s in services_status)
        
        health_response = HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            healthy=overall_healthy,
            timestamp=datetime.now().isoformat(),  # FIXED: JSON serializable
            version="2.2.0-fixed",
            services=services_status
        )
        
        if not overall_healthy:
            return JSONResponse(
                status_code=503,
                content=health_response.dict()
            )
        
        return health_response
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/models")
async def list_available_models():
    """FIXED: List models with proper async/await"""
    
    try:
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        # FIXED: Proper await
        models = await llm_router.get_available_models()
        return {"object": "list", "data": models}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """FIXED: Get metrics with JSON-serializable format"""
    try:
        if metrics_collector:
            # FIXED: Proper await and JSON serialization
            return await metrics_collector.get_all_metrics()
        else:
            return {
                "status": "basic_metrics",
                "timestamp": datetime.now().isoformat(),  # FIXED: JSON serializable
                "message": "Enhanced metrics not available",
                "basic_stats": {
                    "requests_handled": "unknown",
                    "uptime": "unknown"
                }
            }
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return {
            "error": str(e), 
            "timestamp": datetime.now().isoformat()  # FIXED: JSON serializable
        }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "name": "FIXED Enhanced LLM Proxy",
        "version": "2.2.0-fixed",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),  # FIXED: JSON serializable
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics", 
            "models": "/models",
            "chat_completions": "/v1/chat/completions",
            "docs": "/docs"
        },
        "features": enhanced_capabilities,
        "fixes_applied": [
            "Proper async/await handling",
            "JSON serializable datetime objects",
            "Port conflict resolution",
            "Error handling improvements"
        ]
    }

# FIXED: Error handlers with JSON serialization
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """FIXED: HTTP exception handler with JSON serialization"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),  # FIXED: JSON serializable
            "path": str(request.url.path),
            "method": request.method
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """FIXED: General exception handler with proper logging"""
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Track error in metrics
    if metrics_collector:
        metrics_collector.track_error("unhandled_exception")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "type": type(exc).__name__,
            "timestamp": datetime.now().isoformat(),  # FIXED: JSON serializable
            "path": str(request.url.path),
            "method": request.method
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main_fixed:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
