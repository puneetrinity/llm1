# main.py - Complete Production-Ready LLM Proxy with Optimized Router
import os
import sys
import json
import uuid
import time
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# FastAPI and HTTP
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
import uvicorn

# Add this line with your other imports - OPTIMIZED ROUTER
from services.enhanced_router import EnhancedLLMRouter as OptimizedModelRouter

# Configuration with graceful fallbacks
try:
    from config_enhanced import get_settings
    settings = get_settings()
    logging.info("✅ Enhanced configuration loaded")
except ImportError:
    try:
        from config import Settings
        settings = Settings()
        logging.info("✅ Basic configuration loaded")
    except ImportError:
        # Ultimate fallback configuration
        from pydantic_settings import BaseSettings
        
        class FallbackSettings(BaseSettings):
            model_config = {"extra": "ignore"}
            
            # Basic settings
            DEBUG: bool = False
            HOST: str = "0.0.0.0"
            PORT: int = 8000
            LOG_LEVEL: str = "INFO"
            
            # Ollama settings
            OLLAMA_BASE_URL: str = "http://localhost:11434"
            OLLAMA_TIMEOUT: int = 300
            
            # Security settings
            ENABLE_AUTH: bool = False
            DEFAULT_API_KEY: str = "sk-dev-key"
            API_KEY_HEADER: str = "X-API-Key"
            
            # Feature flags
            ENABLE_RATE_LIMITING: bool = False
            ENABLE_STREAMING: bool = True
            ENABLE_MODEL_WARMUP: bool = True
            ENABLE_DETAILED_METRICS: bool = True
            ENABLE_SEMANTIC_CLASSIFICATION: bool = False
            
            # Resource limits
            MAX_MEMORY_MB: int = 8192
            DEFAULT_RATE_LIMIT: int = 100
            
            # CORS
            CORS_ORIGINS: list = ["*"]
            CORS_ALLOW_CREDENTIALS: bool = True
            
            # Models
            DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"
            MAX_TOKENS: int = 2048
            DEFAULT_TEMPERATURE: float = 0.7
        
        settings = FallbackSettings()
        logging.info("✅ Fallback configuration loaded")

# Configure logging
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Model imports with graceful fallbacks
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
        echo: bool = False
        user: Optional[str] = None
    
    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[Dict[str, Any]]
        usage: Dict[str, Any]
    
    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        timestamp: str
        version: str = "2.2.0"
        services: List[Dict[str, Any]] = []

# Global service instances
ollama_client = None
llm_router = None
auth_service = None
metrics_collector = None
performance_monitor = None
memory_manager = None
warmup_service = None
streaming_service = None
enhanced_capabilities = {}

# Service initialization with comprehensive fallbacks
async def initialize_core_services():
    """Initialize all services with graceful fallbacks"""
    global ollama_client, llm_router, auth_service, metrics_collector
    global performance_monitor, memory_manager, warmup_service, streaming_service
    global enhanced_capabilities
    
    try:
        # Initialize memory manager first
        try:
            from utils.memory_manager import get_memory_manager
            memory_manager = get_memory_manager(settings.MAX_MEMORY_MB)
            await memory_manager.start_monitoring()
            logging.info("✅ Memory manager initialized")
        except Exception as e:
            logging.warning(f"Memory manager initialization failed: {e}")
        
        # Initialize metrics collector
        try:
            from utils.metrics import MetricsCollector
            metrics_collector = MetricsCollector()
            logging.info("✅ Metrics collector initialized")
        except Exception as e:
            logging.warning(f"Metrics collector initialization failed: {e}")
        
        # Initialize authentication service
        try:
            from services.auth import AuthService
            auth_service = AuthService(settings)
            logging.info("✅ Authentication service initialized")
        except Exception as e:
            logging.warning(f"Auth service initialization failed: {e}")
        
        # Initialize Ollama client with enhanced features
        try:
            from services.enhanced_imports import setup_enhanced_imports
            enhanced_imports = setup_enhanced_imports()
            enhanced_capabilities = enhanced_imports['capabilities']
            
            # Try enhanced client first
            if enhanced_imports.get('EnhancedOllamaClient'):
                ollama_client = enhanced_imports['EnhancedOllamaClient'](
                    settings.OLLAMA_BASE_URL, 
                    getattr(settings, 'OLLAMA_TIMEOUT', 300)
                )
                await ollama_client.initialize()
                logging.info("✅ Enhanced Ollama client initialized")
            else:
                raise ImportError("Enhanced client not available")
                
        except Exception as e:
            logging.warning(f"Enhanced Ollama client failed: {e}")
            # Fallback to basic client
            try:
                from services.ollama_client import OllamaClient
                ollama_client = OllamaClient(settings.OLLAMA_BASE_URL)
                await ollama_client.initialize()
                enhanced_capabilities = {"streaming": False, "enhanced": False}
                logging.info("✅ Basic Ollama client initialized")
            except Exception as e2:
                logging.error(f"All Ollama client options failed: {e2}")
                # Create minimal mock client for testing
                class MockOllamaClient:
                    def __init__(self):
                        self.base_url = settings.OLLAMA_BASE_URL
                    
                    async def initialize(self):
                        pass
                    
                    async def health_check(self):
                        return False
                    
                    async def list_models(self):
                        return []
                    
                    async def generate_completion(self, *args, **kwargs):
                        return {
                            "id": f"mock-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": "mock-model",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": "Mock response - Ollama not available"},
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                        }
                
                ollama_client = MockOllamaClient()
                await ollama_client.initialize()
                enhanced_capabilities = {"mock": True}
                logging.warning("⚠️ Using mock Ollama client - service may not work properly")
        
        # Initialize optimized router for your model fleet - REPLACED ROUTER INITIALIZATION
        try:
            llm_router = OptimizedModelRouter(ollama_client)
            await llm_router.initialize()
            logging.info("✅ Optimized model router initialized")
            
            # Log available models and their specialties
            if hasattr(llm_router, 'available_models'):
                for model, config in llm_router.available_models.items():
                    specialties = ', '.join(config.get('good_for', ['general']))
                    logging.info(f"   📍 {model}: {specialties}")
            
        except Exception as e:
            logging.error(f"Optimized router initialization failed: {e}")
            # Keep your existing fallback logic here
            try:
                from services.router import LLMRouter
                llm_router = LLMRouter(ollama_client)
                await llm_router.initialize()
                logging.info("✅ Basic router fallback initialized")
            except Exception as e2:
                logging.error(f"All router options failed: {e2}")
                # Create basic router fallback
                class BasicRouter:
                    def __init__(self, client):
                        self.ollama_client = client
                        self.default_model = getattr(settings, 'DEFAULT_MODEL', 'mistral:7b-instruct-q4_0')
                    
                    async def initialize(self):
                        pass
                    
                    async def route_request(self, request):
                        return getattr(request, 'model', self.default_model)
                    
                    async def process_chat_completion(self, request, model):
                        messages = []
                        for msg in request.messages:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                messages.append({"role": msg.role, "content": msg.content})
                        
                        return await self.ollama_client.generate_completion(
                            model=model,
                            messages=messages,
                            temperature=getattr(request, 'temperature', 0.7),
                            max_tokens=getattr(request, 'max_tokens', 150)
                        )
                    
                    async def get_available_models(self):
                        try:
                            models = await self.ollama_client.list_models()
                            return [{"id": model.get("name", "unknown"), "object": "model"} for model in models]
                        except:
                            return [{"id": self.default_model, "object": "model"}]
                
                llm_router = BasicRouter(ollama_client)
                await llm_router.initialize()
                logging.info("✅ Basic router fallback initialized")
        
        # Initialize optional enhanced services
        try:
            if enhanced_capabilities.get('model_warmup') and settings.ENABLE_MODEL_WARMUP:
                from services.model_warmup import ModelWarmupService
                warmup_service = ModelWarmupService(ollama_client, llm_router)
                await warmup_service.start_warmup_service()
                logging.info("✅ Model warmup service initialized")
        except Exception as e:
            logging.warning(f"Warmup service initialization failed: {e}")
        
        try:
            if enhanced_capabilities.get('streaming') and settings.ENABLE_STREAMING:
                from services.streaming import StreamingService
                streaming_service = StreamingService(ollama_client)
                logging.info("✅ Streaming service initialized")
        except Exception as e:
            logging.warning(f"Streaming service initialization failed: {e}")
        
        try:
            if enhanced_capabilities.get('performance_monitor'):
                from utils.performance_monitor import PerformanceMonitor
                performance_monitor = PerformanceMonitor()
                await performance_monitor.start_monitoring()
                logging.info("✅ Performance monitor initialized")
        except Exception as e:
            logging.warning(f"Performance monitor initialization failed: {e}")
        
        logging.info("🎯 Core services initialization complete")
        
    except Exception as e:
        logging.error(f"Critical error in service initialization: {e}")
        logging.error(traceback.format_exc())

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper startup/shutdown"""
    
    # Startup
    logging.info("🚀 Starting LLM Proxy Service...")
    app.state.start_time = datetime.now()
    
    try:
        await initialize_core_services()
        logging.info("✅ All services initialized successfully")
        log_startup_summary()
    except Exception as e:
        logging.error(f"❌ Service initialization failed: {e}")
        # Continue anyway to allow health checks to report issues
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down LLM Proxy Service...")
    
    try:
        if performance_monitor:
            await performance_monitor.stop_monitoring()
        if memory_manager:
            await memory_manager.stop_monitoring()
        if warmup_service:
            await warmup_service.stop_warmup_service()
        if ollama_client and hasattr(ollama_client, 'cleanup'):
            await ollama_client.cleanup()
        logging.info("✅ Services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

def log_startup_summary():
    """Log comprehensive startup information"""
    logging.info("=" * 60)
    logging.info("🚀 LLM Proxy - Startup Summary")
    logging.info("=" * 60)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Host: {settings.HOST}:{settings.PORT}")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"   • Auth Enabled: {settings.ENABLE_AUTH}")
    logging.info(f"   • Rate Limiting: {settings.ENABLE_RATE_LIMITING}")
    logging.info(f"🎯 Services:")
    logging.info(f"   • Ollama Client: {'✅' if ollama_client else '❌'}")
    logging.info(f"   • LLM Router: {'✅ Optimized' if llm_router else '❌'}")
    logging.info(f"   • Auth Service: {'✅' if auth_service else '❌'}")
    logging.info(f"   • Metrics: {'✅' if metrics_collector else '❌'}")
    logging.info(f"   • Memory Manager: {'✅' if memory_manager else '❌'}")
    logging.info(f"   • Warmup Service: {'✅' if warmup_service else '❌'}")
    logging.info(f"   • Streaming: {'✅' if streaming_service else '❌'}")
    logging.info(f"🚀 Enhanced Features: {list(enhanced_capabilities.keys())}")
    logging.info("=" * 60)

# Create FastAPI application
app = FastAPI(
    title="LLM Proxy",
    description="Production-ready LLM routing proxy with optimized model routing",
    version="2.2.0",
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
    allow_headers=["*"]
)

# Add enhanced middleware if available
try:
    if settings.ENABLE_AUTH and auth_service:
        from middleware.auth import AuthMiddleware
        app.add_middleware(AuthMiddleware, auth_service=auth_service)
        logging.info("✅ Authentication middleware added")
except Exception as e:
    logging.warning(f"Auth middleware failed: {e}")

try:
    if settings.ENABLE_RATE_LIMITING:
        from middleware.rate_limit import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware, default_limit=settings.DEFAULT_RATE_LIMIT)
        logging.info("✅ Rate limiting middleware added")
except Exception as e:
    logging.warning(f"Rate limiting middleware failed: {e}")

try:
    from middleware.logging import LoggingMiddleware
    app.add_middleware(LoggingMiddleware, enable_detailed_logging=settings.ENABLE_DETAILED_METRICS)
    logging.info("✅ Logging middleware added")
except Exception as e:
    logging.warning(f"Logging middleware failed: {e}")

# Authentication dependency
security = HTTPBearer(auto_error=False)

async def get_current_user(request: Request, token: Optional[str] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current user from request"""
    if not settings.ENABLE_AUTH or not auth_service:
        return {"user_id": "anonymous", "permissions": ["read", "write"]}
    
    if hasattr(request.state, 'user'):
        return request.state.user
    
    if token and token.credentials:
        user_info = auth_service.validate_api_key(token.credentials)
        if user_info:
            request.state.user = user_info
            return user_info
    
    if settings.ENABLE_AUTH:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return {"user_id": "anonymous", "permissions": ["read", "write"]}

# Helper function to track metrics
def track_request_metrics(endpoint: str, duration: float, success: bool):
    """Track request metrics if available"""
    try:
        if metrics_collector:
            metrics_collector.track_request(endpoint)
            metrics_collector.track_response_time(duration)
            if not success:
                metrics_collector.track_error("request_error")
        
        if performance_monitor:
            performance_monitor.record_request(duration, success)
    except Exception as e:
        logging.debug(f"Metrics tracking failed: {e}")

# Main API Endpoints
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Chat completions endpoint (OpenAI compatible)"""
    
    start_time = time.time()
    success = False
    
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        # Handle streaming
        if request.stream and streaming_service:
            try:
                selected_model = await llm_router.route_request(request)
                
                async def stream_generator():
                    async for chunk in streaming_service.stream_chat_completion(
                        request.dict(), selected_model
                    ):
                        yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            except Exception as e:
                logging.error(f"Streaming failed: {e}")
                # Fall back to non-streaming
        
        # Route request to appropriate model
        selected_model = await llm_router.route_request(request)
        
        # Process request
        response = await llm_router.process_chat_completion(request, selected_model)
        
        # Add routing metadata to response (NEW ENHANCEMENT)
        duration = time.time() - start_time
        if isinstance(response, dict) and hasattr(llm_router, 'classify_intent'):
            text_content = ' '.join(msg.content for msg in request.messages if hasattr(msg, 'content'))
            intent = llm_router.classify_intent(text_content)
            
            response["routing_metadata"] = {
                "selected_model": selected_model,
                "intent_classification": intent,
                "processing_time": duration,
                "routing_strategy": "optimized_intent_based"
            }
        
        # Track metrics
        success = True
        track_request_metrics("/v1/chat/completions", duration, success)
        
        # Track model usage if metrics available
        if metrics_collector and response:
            try:
                usage = response.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                if total_tokens > 0:
                    metrics_collector.track_model_usage(selected_model, total_tokens, duration)
            except Exception as e:
                logging.debug(f"Model usage tracking failed: {e}")
        
        return response
        
    except HTTPException:
        duration = time.time() - start_time
        track_request_metrics("/v1/chat/completions", duration, False)
        raise
    except Exception as e:
        duration = time.time() - start_time
        track_request_metrics("/v1/chat/completions", duration, False)
        logging.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Text completions endpoint (OpenAI compatible)"""
    
    start_time = time.time()
    success = False
    
    try:
        # Convert to chat format
        messages = [Message(role="user", content=request.prompt)]
        
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream
        )
        
        # Handle streaming
        if request.stream and streaming_service:
            try:
                selected_model = await llm_router.route_request(chat_request)
                
                async def completion_stream_generator():
                    async for chunk in streaming_service.stream_chat_completion(
                        chat_request.dict(), selected_model
                    ):
                        # Convert chat completion chunk to text completion format
                        if chunk.startswith("data: "):
                            try:
                                chunk_data = json.loads(chunk[6:])
                                if "choices" in chunk_data:
                                    # Convert to text completion format
                                    for choice in chunk_data["choices"]:
                                        if "delta" in choice and "content" in choice["delta"]:
                                            choice["text"] = choice["delta"]["content"]
                                            del choice["delta"]
                                        elif "message" in choice:
                                            choice["text"] = choice["message"].get("content", "")
                                            del choice["message"]
                                    
                                    chunk_data["object"] = "text_completion"
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                            except json.JSONDecodeError:
                                yield chunk
                        else:
                            yield chunk
                
                return StreamingResponse(
                    completion_stream_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            except Exception as e:
                logging.error(f"Completion streaming failed: {e}")
                # Fall back to non-streaming
        
        # Non-streaming completion
        response = await chat_completions(chat_request, http_request, current_user)
        
        # Convert to completion format
        if isinstance(response, dict) and "choices" in response:
            completion_response = {
                "id": response["id"].replace("chatcmpl-", "cmpl-"),
                "object": "text_completion",
                "created": response["created"],
                "model": response["model"],
                "choices": [],
                "usage": response.get("usage", {})
            }
            
            for choice in response["choices"]:
                completion_choice = {
                    "text": choice.get("message", {}).get("content", ""),
                    "index": choice.get("index", 0),
                    "finish_reason": choice.get("finish_reason", "stop")
                }
                
                if request.echo:
                    completion_choice["text"] = request.prompt + completion_choice["text"]
                
                completion_response["choices"].append(completion_choice)
            
            success = True
            duration = time.time() - start_time
            track_request_metrics("/v1/completions", duration, success)
            
            return completion_response
        
        success = True
        duration = time.time() - start_time
        track_request_metrics("/v1/completions", duration, success)
        return response
        
    except HTTPException:
        duration = time.time() - start_time
        track_request_metrics("/v1/completions", duration, False)
        raise
    except Exception as e:
        duration = time.time() - start_time
        track_request_metrics("/v1/completions", duration, False)
        logging.error(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    
    try:
        services_status = []
        overall_healthy = True
        
        # Check Ollama
        if ollama_client:
            try:
                ollama_healthy = await ollama_client.health_check()
                services_status.append({
                    "name": "ollama",
                    "status": "healthy" if ollama_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "url": getattr(ollama_client, 'base_url', 'unknown')
                })
                if not ollama_healthy:
                    overall_healthy = False
            except Exception as e:
                services_status.append({
                    "name": "ollama",
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "error": str(e)
                })
                overall_healthy = False
        else:
            services_status.append({
                "name": "ollama",
                "status": "unavailable",
                "last_check": datetime.now().isoformat()
            })
            overall_healthy = False
        
        # Check memory if available
        if memory_manager:
            try:
                memory_status = memory_manager.get_status_summary()
                health_status = memory_status["health"]["status"]
                services_status.append({
                    "name": "memory",
                    "status": health_status,
                    "last_check": datetime.now().isoformat(),
                    "utilization": f"{memory_status['health']['utilization_percent']:.1f}%"
                })
                if health_status not in ["healthy", "good"]:
                    overall_healthy = False
            except Exception as e:
                services_status.append({
                    "name": "memory",
                    "status": "unknown",
                    "error": str(e)
                })
        
        health_response = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "services": services_status,
            "uptime_seconds": (datetime.now() - app.state.start_time).total_seconds() if hasattr(app.state, 'start_time') else 0
        }
        
        if not overall_healthy:
            return JSONResponse(status_code=503, content=health_response)
        
        return health_response
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@app.get("/models")
async def list_available_models():
    """List available models (OpenAI compatible)"""
    
    try:
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        models = await llm_router.get_available_models()
        return {"object": "list", "data": models}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        # Return fallback model list
        return {
            "object": "list",
            "data": [{
                "id": settings.DEFAULT_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama"
            }]
        }

# Admin and monitoring endpoints
@app.get("/admin/status")
async def admin_status(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Comprehensive system status for administrators"""
    
    # Check admin permissions
    if settings.ENABLE_AUTH and current_user:
        permissions = current_user.get("permissions", [])
        if "admin" not in permissions and "read" not in permissions:
            raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "uptime_seconds": (datetime.now() - app.state.start_time).total_seconds() if hasattr(app.state, 'start_time') else 0,
            "configuration": {
                "auth_enabled": settings.ENABLE_AUTH,
                "rate_limiting_enabled": settings.ENABLE_RATE_LIMITING,
                "streaming_enabled": settings.ENABLE_STREAMING,
                "model_warmup_enabled": settings.ENABLE_MODEL_WARMUP,
                "semantic_classification_enabled": settings.ENABLE_SEMANTIC_CLASSIFICATION
            },
            "services": {
                "ollama_client": ollama_client is not None,
                "llm_router": llm_router is not None,
                "auth_service": auth_service is not None,
                "metrics_collector": metrics_collector is not None,
                "performance_monitor": performance_monitor is not None,
                "memory_manager": memory_manager is not None,
                "warmup_service": warmup_service is not None,
                "streaming_service": streaming_service is not None
            },
            "enhanced_capabilities": enhanced_capabilities,
            "ollama": {
                "url": settings.OLLAMA_BASE_URL,
                "timeout": getattr(settings, 'OLLAMA_TIMEOUT', 300)
            }
        }
        
        # Add memory status if available
        if memory_manager:
            try:
                memory_status = memory_manager.get_status_summary()
                status_data["memory"] = memory_status
            except Exception as e:
                status_data["memory"] = {"error": str(e)}
        
        # Add performance stats if available
        if performance_monitor:
            try:
                perf_stats = await performance_monitor.get_current_performance_summary()
                status_data["performance"] = perf_stats
            except Exception as e:
                status_data["performance"] = {"error": str(e)}
        
        return status_data
        
    except Exception as e:
        logging.error(f"Admin status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW ADMIN ENDPOINTS - OPTIMIZED ROUTING
@app.get("/admin/routing/stats")
async def admin_routing_stats(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Get intelligent routing statistics"""
    
    if settings.ENABLE_AUTH and current_user:
        permissions = current_user.get("permissions", [])
        if "admin" not in permissions and "read" not in permissions:
            raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        if not llm_router or not hasattr(llm_router, 'get_classification_stats'):
            raise HTTPException(status_code=503, detail="Optimized routing not available")
        
        routing_stats = llm_router.get_classification_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "routing_strategy": "optimized_intent_based",
            "model_fleet": routing_stats
        }
        
    except Exception as e:
        logging.error(f"Routing stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/routing/test")
async def admin_test_routing(
    test_request: dict,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Test the routing logic with sample text"""
    
    if settings.ENABLE_AUTH and current_user:
        permissions = current_user.get("permissions", [])
        if "admin" not in permissions:
            raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        if not llm_router or not hasattr(llm_router, 'classify_intent'):
            raise HTTPException(status_code=503, detail="Optimized routing not available")
        
        text = test_request.get('text', '')
        if not text:
            raise HTTPException(status_code=400, detail="Text field required")
        
        # Test intent classification
        intent = llm_router.classify_intent(text)
        
        # Create mock request for routing test
        class MockRequest:
            def __init__(self, text):
                self.messages = [type('MockMessage', (), {'role': 'user', 'content': text})()]
                self.model = "test"
        
        mock_request = MockRequest(text)
        selected_model = await llm_router.route_request(mock_request)
        
        return {
            "test_text": text,
            "classified_intent": intent,
            "selected_model": selected_model,
            "routing_reason": f"Optimized for {intent} tasks",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Routing test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics"""
    
    try:
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "basic_metrics": {}
        }
        
        # Basic metrics if available
        if metrics_collector:
            try:
                basic_metrics = await metrics_collector.get_all_metrics()
                metrics_data["basic_metrics"] = basic_metrics
            except Exception as e:
                metrics_data["basic_metrics"] = {"error": str(e)}
        
        # Performance metrics if available
        if performance_monitor:
            try:
                perf_metrics = await performance_monitor.get_current_performance_summary()
                metrics_data["performance_metrics"] = perf_metrics
            except Exception as e:
                metrics_data["performance_metrics"] = {"error": str(e)}
        
        # Enhanced features metrics
        metrics_data["features"] = {
            "enhanced_capabilities": enhanced_capabilities,
            "services_active": {
                "ollama": ollama_client is not None,
                "router": llm_router is not None,
                "auth": auth_service is not None,
                "metrics": metrics_collector is not None,
                "warmup": warmup_service is not None,
                "streaming": streaming_service is not None
            }
        }
        
        return metrics_data
        
    except Exception as e:
        logging.error(f"Metrics error: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "basic_metrics": {
                "message": "Metrics collection failed"
            }
        }

@app.post("/admin/warmup/{model}")
async def admin_warmup_model(
    model: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Manually warm up a specific model"""
    
    # Check admin permissions
    if settings.ENABLE_AUTH and current_user:
        permissions = current_user.get("permissions", [])
        if "admin" not in permissions and "write" not in permissions:
            raise HTTPException(status_code=403, detail="Admin access required")
    
    if not warmup_service:
        raise HTTPException(status_code=503, detail="Warmup service not available")
    
    try:
        success = await warmup_service.warmup_model(model)
        return {
            "model": model,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "message": f"Model {model} warmup {'succeeded' if success else 'failed'}"
        }
    except Exception as e:
        logging.error(f"Model warmup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/warmup/stats")
async def admin_warmup_stats(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Get model warmup statistics"""
    
    if not warmup_service:
        raise HTTPException(status_code=503, detail="Warmup service not available")
    
    try:
        stats = warmup_service.get_warmup_stats()
        return {
            "timestamp": datetime.now().isoformat(),
            "warmup_stats": stats
        }
    except Exception as e:
        logging.error(f"Warmup stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper formatting"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code
            },
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    
    error_id = str(uuid.uuid4())[:8]
    
    logging.error(f"Unhandled exception [{error_id}]: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "error_id": error_id
            },
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "LLM Proxy",
        "version": "2.2.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/models",
            "health": "/health",
            "metrics": "/metrics",
            "admin_routing_stats": "/admin/routing/stats",
            "admin_routing_test": "/admin/routing/test",
            "docs": "/docs"
        },
        "features": enhanced_capabilities,
        "routing": "optimized_intent_based"
    }

# Main execution
if __name__ == "__main__":
    try:
        # Validate configuration
        if hasattr(settings, 'ENABLE_AUTH') and settings.ENABLE_AUTH:
            if len(settings.DEFAULT_API_KEY) < 16:
                logging.warning("⚠️ API key is too short for production use")
        
        if settings.MAX_MEMORY_MB < 2048:
            logging.warning("⚠️ Memory limit is very low - may cause issues")
        
        logging.info(f"🚀 Starting LLM Proxy with Optimized Routing on {settings.HOST}:{settings.PORT}")
        
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=settings.ENABLE_DETAILED_METRICS
        )
        
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        sys.exit(1)
