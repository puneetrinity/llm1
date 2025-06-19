#!/usr/bin/env python3
"""
Enhanced LLM Proxy - Production-Ready Main Application
Version: 4.0.0
Features: 4-Model Routing, Semantic Classification, Caching, Streaming, WebSockets
"""

import os
import sys
import asyncio
import logging
import json
import time
import psutil
import aiohttp
import uvicorn
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from collections import deque

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import configuration
try:
    from config_enhanced import Settings
except ImportError:
    from config import Settings

# Import models
from models.requests import ChatCompletionRequest, CompletionRequest
from models.responses import (
    ChatCompletionResponse, 
    ChatCompletionStreamResponse,
    CompletionResponse,
    ModelResponse,
    ModelListResponse
)

# Import services
from services.ollama_client import OllamaClient
from services.cache_service import CacheService
from services.circuit_breaker import CircuitBreaker
from middleware.auth import AuthMiddleware
from middleware.rate_limiter import RateLimiter
from utils.helpers import format_openai_response, handle_streaming_response

# Optional enhanced services
try:
    from services.optimized_router import EnhancedLLMRouter
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    logging.warning("Enhanced router not available - using basic routing")

try:
    from services.model_warmup import ModelWarmupService
    WARMUP_AVAILABLE = True
except ImportError:
    WARMUP_AVAILABLE = False
    logging.warning("Model warmup service not available")

try:
    from services.semantic_classifier import SemanticIntentClassifier
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.warning("Semantic classifier not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/logs/app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Global service instances
ollama_client: Optional[OllamaClient] = None
cache_service: Optional[CacheService] = None
circuit_breaker: Optional[CircuitBreaker] = None
router: Optional[Any] = None
warmup_service: Optional[Any] = None
auth_middleware: Optional[AuthMiddleware] = None
rate_limiter: Optional[RateLimiter] = None

# Application state
app_state = {
    "initialized": False,
    "start_time": datetime.now(),
    "services": {
        "ollama": False,
        "cache": False,
        "router": False,
        "warmup": False,
        "auth": False,
        "rate_limiter": False
    },
    "models": {
        "available": [],
        "loaded": [],
        "downloading": []
    },
    "metrics": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "model_usage": {},
        "response_times": deque(maxlen=1000)
    }
}

# WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}

# Response models
class HealthResponse(BaseModel):
    status: str
    healthy: bool
    timestamp: str
    version: str
    services: Dict[str, Any]
    uptime_seconds: int

class MetricsResponse(BaseModel):
    timestamp: str
    uptime_seconds: int
    performance: Dict[str, Any]
    system: Dict[str, Any]
    models: Dict[str, Any]
    cache: Optional[Dict[str, Any]]
    services: Dict[str, Any]

# Initialization service with retry logic
class InitializationService:
    def __init__(self):
        self.max_retries = 60
        self.retry_delay = 1.0
        
    async def wait_for_ollama(self) -> bool:
        """Wait for Ollama to be ready"""
        logger.info(f"Waiting for Ollama at {settings.OLLAMA_BASE_URL}...")
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{settings.OLLAMA_BASE_URL}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            models = data.get('models', [])
                            logger.info(f"✅ Ollama ready with {len(models)} models")
                            return True
            except Exception as e:
                if attempt % 10 == 0:  # Log every 10 attempts
                    logger.debug(f"Ollama not ready (attempt {attempt + 1}/{self.max_retries})")
            
            await asyncio.sleep(self.retry_delay)
        
        logger.error("❌ Ollama failed to start")
        return False

# Initialize all services
async def initialize_services():
    """Initialize all application services"""
    global ollama_client, cache_service, circuit_breaker, router
    global warmup_service, auth_middleware, rate_limiter
    
    init_service = InitializationService()
    
    try:
        # Step 1: Wait for Ollama
        if not await init_service.wait_for_ollama():
            raise Exception("Ollama service not available")
        
        # Step 2: Initialize Ollama client
        ollama_client = OllamaClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT
        )
        await ollama_client.initialize()
        app_state["services"]["ollama"] = True
        
        # Step 3: Get available models
        models = await ollama_client.list_models()
        app_state["models"]["available"] = [m.get('name', '') for m in models]
        logger.info(f"Available models: {app_state['models']['available']}")
        
        # Step 4: Initialize cache service
        if settings.ENABLE_CACHE:
            cache_service = CacheService(
                ttl=settings.CACHE_TTL,
                max_size=settings.CACHE_MAX_SIZE
            )
            app_state["services"]["cache"] = True
            logger.info("✅ Cache service initialized")
        
        # Step 5: Initialize circuit breaker
        circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception
        )
        logger.info("✅ Circuit breaker initialized")
        
        # Step 6: Initialize router
        if ROUTER_AVAILABLE and settings.ENABLE_MODEL_ROUTING:
            router = EnhancedLLMRouter(ollama_client)
            await router.initialize()
            app_state["services"]["router"] = True
            logger.info("✅ Enhanced router initialized")
        
        # Step 7: Initialize model warmup
        if WARMUP_AVAILABLE and settings.ENABLE_MODEL_WARMUP and router:
            warmup_service = ModelWarmupService(ollama_client, router)
            asyncio.create_task(warmup_service.start_warmup_service())
            app_state["services"]["warmup"] = True
            logger.info("✅ Model warmup service started")
        
        # Step 8: Initialize auth middleware
        if settings.ENABLE_AUTH:
            auth_middleware = AuthMiddleware(
                api_keys=settings.API_KEYS,
                enable_auth=True
            )
            app_state["services"]["auth"] = True
            logger.info("✅ Authentication enabled")
        
        # Step 9: Initialize rate limiter
        if settings.ENABLE_RATE_LIMITING:
            rate_limiter = RateLimiter(
                requests_per_minute=settings.RATE_LIMIT_PER_MINUTE
            )
            app_state["services"]["rate_limiter"] = True
            logger.info("✅ Rate limiting enabled")
        
        app_state["initialized"] = True
        logger.info("🎉 All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        app_state["initialized"] = False
        raise

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("="*60)
    logger.info("🚀 Starting Enhanced LLM Proxy v4.0.0")
    logger.info(f"🔧 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🌐 Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"🔗 Ollama: {settings.OLLAMA_BASE_URL}")
    logger.info("="*60)
    
    # Initialize services
    await initialize_services()
    
    # Log enabled features
    features = []
    if settings.ENABLE_AUTH: features.append("Authentication")
    if settings.ENABLE_CACHE: features.append("Caching")
    if settings.ENABLE_MODEL_ROUTING: features.append("Smart Routing")
    if settings.ENABLE_STREAMING: features.append("Streaming")
    if settings.ENABLE_WEBSOCKET: features.append("WebSocket")
    if settings.ENABLE_SEMANTIC_CLASSIFICATION: features.append("Semantic AI")
    
    logger.info(f"✨ Enabled features: {', '.join(features)}")
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced LLM Proxy...")
    
    if warmup_service:
        await warmup_service.stop_warmup_service()
    
    if router and hasattr(router, 'cleanup'):
        await router.cleanup()
    
    if ollama_client:
        await ollama_client.cleanup()
    
    logger.info("👋 Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Enhanced LLM Proxy",
    description="Production-ready OpenAI-compatible API with 4-model routing, caching, and AI features",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
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

# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track all requests for metrics"""
    start_time = time.time()
    
    # Track request
    app_state["metrics"]["total_requests"] += 1
    
    try:
        response = await call_next(request)
        
        # Track successful request
        if response.status_code < 400:
            app_state["metrics"]["successful_requests"] += 1
        else:
            app_state["metrics"]["failed_requests"] += 1
        
        # Track response time
        response_time = (time.time() - start_time) * 1000
        app_state["metrics"]["response_times"].append(response_time)
        
        # Add custom headers
        response.headers["X-Process-Time"] = f"{response_time:.2f}ms"
        response.headers["X-LLM-Proxy-Version"] = "4.0.0"
        
        return response
        
    except Exception as e:
        app_state["metrics"]["failed_requests"] += 1
        raise

# Authentication dependency
async def verify_api_key(request: Request):
    """Verify API key if auth is enabled"""
    if not settings.ENABLE_AUTH:
        return True
    
    if not auth_middleware:
        return True
    
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        api_key = request.headers.get("X-API-Key", "")
    
    if not auth_middleware.validate_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# Rate limiting dependency
async def check_rate_limit(request: Request):
    """Check rate limit if enabled"""
    if not settings.ENABLE_RATE_LIMITING or not rate_limiter:
        return True
    
    client_id = request.client.host if request.client else "unknown"
    
    if not await rate_limiter.check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return True

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - app_state["start_time"]).total_seconds()
    
    return HealthResponse(
        status="healthy" if app_state["initialized"] else "initializing",
        healthy=app_state["initialized"],
        timestamp=datetime.now().isoformat(),
        version="4.0.0",
        services=app_state["services"],
        uptime_seconds=int(uptime)
    )

# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics"""
    uptime = (datetime.now() - app_state["start_time"]).total_seconds()
    
    # Calculate average response time
    response_times = list(app_state["metrics"]["response_times"])
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Get system metrics
    system_metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "network_connections": len(psutil.net_connections())
    }
    
    # Get cache metrics if available
    cache_metrics = None
    if cache_service:
        cache_metrics = {
            "size": cache_service.cache_size(),
            "hits": app_state["metrics"]["cache_hits"],
            "misses": app_state["metrics"]["cache_misses"],
            "hit_rate": app_state["metrics"]["cache_hits"] / max(1, 
                app_state["metrics"]["cache_hits"] + app_state["metrics"]["cache_misses"])
        }
    
    return MetricsResponse(
        timestamp=datetime.now().isoformat(),
        uptime_seconds=int(uptime),
        performance={
            "total_requests": app_state["metrics"]["total_requests"],
            "successful_requests": app_state["metrics"]["successful_requests"],
            "failed_requests": app_state["metrics"]["failed_requests"],
            "avg_response_time_ms": round(avg_response_time, 2),
            "requests_per_second": app_state["metrics"]["total_requests"] / max(1, uptime)
        },
        system=system_metrics,
        models={
            "available": app_state["models"]["available"],
            "loaded": app_state["models"]["loaded"],
            "usage": app_state["metrics"]["model_usage"]
        },
        cache=cache_metrics,
        services=app_state["services"]
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Enhanced LLM Proxy",
        "version": "4.0.0",
        "status": "operational" if app_state["initialized"] else "initializing",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "docs": "/docs" if settings.DEBUG else None,
            "dashboard": "/dashboard" if settings.ENABLE_DASHBOARD else None
        },
        "features": {
            "authentication": settings.ENABLE_AUTH,
            "caching": settings.ENABLE_CACHE,
            "streaming": settings.ENABLE_STREAMING,
            "websocket": settings.ENABLE_WEBSOCKET,
            "model_routing": settings.ENABLE_MODEL_ROUTING,
            "semantic_classification": settings.ENABLE_SEMANTIC_CLASSIFICATION
        }
    }

# Models endpoint
@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    _authenticated: bool = Depends(verify_api_key)
):
    """List available models"""
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    models = await ollama_client.list_models()
    
    # Format models for OpenAI compatibility
    formatted_models = []
    for model in models:
        formatted_models.append(
            ModelResponse(
                id=model.get('name', 'unknown'),
                object="model",
                created=int(datetime.now().timestamp()),
                owned_by="ollama"
            )
        )
    
    return ModelListResponse(
        object="list",
        data=formatted_models
    )

# Chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _authenticated: bool = Depends(verify_api_key),
    _rate_limited: bool = Depends(check_rate_limit)
):
    """OpenAI-compatible chat completions endpoint"""
    if not app_state["initialized"]:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    start_time = time.time()
    
    try:
        # Check cache if enabled
        if cache_service and not request.stream:
            cache_key = cache_service.generate_cache_key(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature
            )
            
            cached_response = await cache_service.get(cache_key)
            if cached_response:
                app_state["metrics"]["cache_hits"] += 1
                logger.info(f"Cache hit for model: {request.model}")
                return cached_response
            else:
                app_state["metrics"]["cache_misses"] += 1
        
        # Route to appropriate model
        selected_model = request.model
        if router and settings.ENABLE_MODEL_ROUTING:
            routing_result = await router.route_request(request)
            selected_model = routing_result.get('model', request.model)
            logger.info(f"Routed to model: {selected_model} (reason: {routing_result.get('reason', 'default')})")
        
        # Track model usage
        app_state["metrics"]["model_usage"][selected_model] = \
            app_state["metrics"]["model_usage"].get(selected_model, 0) + 1
        
        # Generate completion
        if request.stream and settings.ENABLE_STREAMING:
            # Streaming response
            return StreamingResponse(
                handle_streaming_response(
                    ollama_client,
                    selected_model,
                    request.messages,
                    request.temperature,
                    request.max_tokens
                ),
                media_type="text/event-stream"
            )
        else:
            # Regular response
            response = await circuit_breaker.call(
                ollama_client.generate_completion,
                model=selected_model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            )
            
            # Format response
            formatted_response = format_openai_response(
                response,
                model=request.model,  # Use original model name for compatibility
                is_chat=True
            )
            
            # Cache response if enabled
            if cache_service and not request.stream:
                await cache_service.set(cache_key, formatted_response)
            
            # Track response time
            response_time = (time.time() - start_time) * 1000
            logger.info(f"Request completed in {response_time:.2f}ms")
            
            return formatted_response
            
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Completions endpoint (legacy)
@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    _authenticated: bool = Depends(verify_api_key),
    _rate_limited: bool = Depends(check_rate_limit)
):
    """OpenAI-compatible completions endpoint (legacy)"""
    if not app_state["initialized"]:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    # Convert to chat format
    messages = [{"role": "user", "content": request.prompt}]
    
    # Generate completion
    response = await circuit_breaker.call(
        ollama_client.generate_completion,
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p
    )
    
    # Format as legacy completion
    return CompletionResponse(
        id=f"cmpl-{int(time.time())}",
        object="text_completion",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "text": response.get("response", ""),
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        }
    )

# WebSocket endpoint
if settings.ENABLE_WEBSOCKET:
    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        api_key: Optional[str] = None
    ):
        """WebSocket endpoint for real-time communication"""
        # Verify API key if auth is enabled
        if settings.ENABLE_AUTH and auth_middleware:
            if not auth_middleware.validate_key(api_key):
                await websocket.close(code=1008, reason="Invalid API key")
                return
        
        await websocket.accept()
        client_id = f"ws-{int(time.time())}"
        websocket_connections[client_id] = websocket
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_json()
                
                # Process message
                if data.get("type") == "chat":
                    messages = data.get("messages", [])
                    model = data.get("model", "mistral:7b-instruct-q4_0")
                    
                    # Generate response
                    response = await ollama_client.generate_completion(
                        model=model,
                        messages=messages,
                        stream=True
                    )
                    
                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "data": response
                    })
                    
        except WebSocketDisconnect:
            del websocket_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")

# Admin endpoints
@app.post("/admin/cache/clear")
async def clear_cache(
    _authenticated: bool = Depends(verify_api_key)
):
    """Clear the cache"""
    if not cache_service:
        raise HTTPException(status_code=404, detail="Cache service not enabled")
    
    await cache_service.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.post("/admin/models/load/{model_name}")
async def load_model(
    model_name: str,
    _authenticated: bool = Depends(verify_api_key)
):
    """Load a specific model"""
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Pull model if not available
        await ollama_client.pull_model(model_name)
        app_state["models"]["loaded"].append(model_name)
        return {"status": "success", "message": f"Model {model_name} loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/router/stats")
async def get_router_stats(
    _authenticated: bool = Depends(verify_api_key)
):
    """Get router statistics"""
    if not router:
        raise HTTPException(status_code=404, detail="Router not enabled")
    
    return router.get_stats()

# Dashboard endpoints
if settings.ENABLE_DASHBOARD:
    # Check for dashboard files
    dashboard_path = None
    react_build = Path(__file__).parent / "frontend" / "build"
    static_dashboard = Path(__file__).parent / "static" / "dashboard"
    
    if react_build.exists() and (react_build / "index.html").exists():
        dashboard_path = react_build
        app.mount("/static", StaticFiles(directory=react_build), name="static")
    elif static_dashboard.exists() and (static_dashboard / "index.html").exists():
        dashboard_path = static_dashboard
        app.mount("/static", StaticFiles(directory=static_dashboard), name="static")
    
    if dashboard_path:
        @app.get("/dashboard")
        async def dashboard():
            """Serve dashboard"""
            return FileResponse(dashboard_path / "index.html")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": 500
            }
        }
    )

# Main entry point
if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS
    )
