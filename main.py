# main.py - Unified LLM Proxy with Graceful Degradation
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import sys
import traceback

# Import configuration
from config_enhanced import EnhancedSettings

# Import models
from models.requests import ChatCompletionRequest, CompletionRequest
from models.responses import ChatCompletionResponse, HealthResponse

# Core services (always available)
from services.ollama_client import OllamaClient
from services.router import LLMRouter
from services.auth import AuthService
from utils.metrics import MetricsCollector
from utils.health import HealthChecker

# Enhanced services with graceful fallback
ENHANCED_FEATURES = {
    "enhanced_ollama_client": False,
    "enhanced_router": False,
    "streaming": False,
    "model_warmup": False,
    "semantic_classification": False
}

# Try to import enhanced services
try:
    from services.enhanced_ollama_client import EnhancedOllamaClient
    ENHANCED_FEATURES["enhanced_ollama_client"] = True
    logging.info("✅ Enhanced Ollama client available")
except ImportError as e:
    logging.info(f"ℹ️  Enhanced Ollama client not available: {e}")
    EnhancedOllamaClient = OllamaClient  # Fallback to basic

try:
    from services.enhanced_router import EnhancedLLMRouter
    ENHANCED_FEATURES["enhanced_router"] = True
    ENHANCED_FEATURES["semantic_classification"] = True
    logging.info("✅ Enhanced router with semantic classification available")
except ImportError as e:
    logging.info(f"ℹ️  Enhanced router not available: {e}")
    EnhancedLLMRouter = LLMRouter  # Fallback to basic

try:
    from services.streaming import StreamingService
    ENHANCED_FEATURES["streaming"] = True
    logging.info("✅ Streaming service available")
except ImportError as e:
    logging.info(f"ℹ️  Streaming service not available: {e}")
    StreamingService = None

try:
    from services.model_warmup import ModelWarmupService
    ENHANCED_FEATURES["model_warmup"] = True
    logging.info("✅ Model warmup service available")
except ImportError as e:
    logging.info(f"ℹ️  Model warmup service not available: {e}")
    ModelWarmupService = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/app.log') if sys.platform != 'win32' else logging.StreamHandler()
    ]
)

# Global instances with enhanced fallbacks
settings = EnhancedSettings()
ollama_client = EnhancedOllamaClient(settings.OLLAMA_BASE_URL, settings.OLLAMA_TIMEOUT)
llm_router = EnhancedLLMRouter(ollama_client)
metrics = MetricsCollector()
health_checker = HealthChecker()
auth_service = AuthService(settings)

# Optional enhanced services
streaming_service = None
warmup_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Unified lifespan management with enhanced feature detection"""
    # Startup
    logging.info("🚀 Starting LLM Proxy Service...")
    
    # Log available features
    enabled_features = [k for k, v in ENHANCED_FEATURES.items() if v]
    disabled_features = [k for k, v in ENHANCED_FEATURES.items() if not v]
    
    if enabled_features:
        logging.info(f"✅ Enhanced features available: {', '.join(enabled_features)}")
    if disabled_features:
        logging.info(f"ℹ️  Basic fallbacks for: {', '.join(disabled_features)}")
    
    startup_errors = []
    
    try:
        # Initialize core services
        await ollama_client.initialize()
        await llm_router.initialize()
        logging.info("✅ Core services initialized")
        
        # Initialize enhanced services if available
        await initialize_enhanced_services()
        
        # Start monitoring
        await health_checker.start_monitoring()
        logging.info("✅ Health monitoring started")
        
        if startup_errors:
            logging.warning(f"⚠️  Some features unavailable: {startup_errors}")
        else:
            logging.info("✅ All available services started successfully")
        
    except Exception as e:
        logging.error(f"❌ Failed to start services: {e}")
        logging.error(traceback.format_exc())
        startup_errors.append(str(e))
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down LLM Proxy Service...")
    
    try:
        await cleanup_services()
        logging.info("✅ All services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

async def initialize_enhanced_services():
    """Initialize enhanced services with fallback handling"""
    global streaming_service, warmup_service
    
    # Initialize streaming if available and enabled
    if StreamingService and ENHANCED_FEATURES["streaming"] and settings.ENABLE_STREAMING:
        try:
            streaming_service = StreamingService(ollama_client)
            logging.info("✅ Streaming service initialized")
        except Exception as e:
            logging.warning(f"⚠️  Failed to initialize streaming: {e}")
            ENHANCED_FEATURES["streaming"] = False
    
    # Initialize warmup if available and enabled
    if ModelWarmupService and ENHANCED_FEATURES["model_warmup"] and settings.ENABLE_MODEL_WARMUP:
        try:
            warmup_service = ModelWarmupService(ollama_client, llm_router)
            await warmup_service.start_warmup_service()
            logging.info("✅ Model warmup service initialized")
        except Exception as e:
            logging.warning(f"⚠️  Failed to initialize warmup: {e}")
            ENHANCED_FEATURES["model_warmup"] = False

async def cleanup_services():
    """Cleanup all services gracefully"""
    global streaming_service, warmup_service
    
    try:
        await llm_router.cleanup()
    except Exception as e:
        logging.error(f"Error cleaning up router: {e}")
    
    try:
        if warmup_service:
            await warmup_service.stop_warmup_service()
    except Exception as e:
        logging.error(f"Error cleaning up warmup service: {e}")
    
    try:
        await health_checker.stop_monitoring()
    except Exception as e:
        logging.error(f"Error cleaning up health checker: {e}")
    
    try:
        await ollama_client.cleanup()
    except Exception as e:
        logging.error(f"Error cleaning up ollama client: {e}")

# Create FastAPI app
app = FastAPI(
    title="LLM Proxy",
    description="Intelligent LLM routing proxy with automatic feature detection and graceful degradation",
    version="2.1.0",
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

# Main API Routes
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request
):
    """OpenAI-compatible chat completions with automatic feature detection"""
    
    request_start_time = asyncio.get_event_loop().time()
    request_id = f"req_{int(request_start_time * 1000000) % 1000000}"
    
    try:
        # Track request
        metrics.track_request("chat_completions", "anonymous")
        
        logging.info(f"[{request_id}] Processing chat completion - Model: {request.model}, Stream: {request.stream}")
        
        # Handle streaming if available and requested
        if request.stream and streaming_service and ENHANCED_FEATURES["streaming"]:
            return await handle_streaming_request(request, request_id)
        elif request.stream and not ENHANCED_FEATURES["streaming"]:
            logging.warning(f"[{request_id}] Streaming requested but not available - falling back to non-streaming")
            request.stream = False  # Force non-streaming
        
        # Route to appropriate model
        selected_model = await llm_router.route_request(request)
        logging.info(f"[{request_id}] Routed to model: {selected_model}")
        
        # Record model usage for warmup if available
        if warmup_service and ENHANCED_FEATURES["model_warmup"]:
            warmup_service.record_model_usage(selected_model)
        
        # Process request
        response = await llm_router.process_chat_completion(request, selected_model)
        
        # Calculate response time
        response_time = asyncio.get_event_loop().time() - request_start_time
        
        # Track usage and performance
        metrics.track_model_usage(
            selected_model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response_time
        )
        
        metrics.track_response_time(response_time)
        
        # Add metadata to response
        response.cache_hit = False
        response.processing_time = response_time
        response.selected_model = selected_model
        
        logging.info(f"[{request_id}] Request completed - Duration: {response_time:.3f}s")
        
        return response
        
    except Exception as e:
        error_response_time = asyncio.get_event_loop().time() - request_start_time
        metrics.track_error(type(e).__name__)
        
        logging.error(f"[{request_id}] Error in chat_completions: {str(e)}")
        logging.error(traceback.format_exc())
        
        raise HTTPException(status_code=500, detail=str(e))

async def handle_streaming_request(request: ChatCompletionRequest, request_id: str) -> StreamingResponse:
    """Handle streaming requests when streaming is available"""
    
    try:
        selected_model = await llm_router.route_request(request)
        logging.info(f"[{request_id}] Streaming request routed to: {selected_model}")
        
        if warmup_service and ENHANCED_FEATURES["model_warmup"]:
            warmup_service.record_model_usage(selected_model)
        
        request_data = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens
        }
        
        return StreamingResponse(
            streaming_service.stream_chat_completion(request_data, selected_model),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
                "X-Selected-Model": selected_model
            }
        )
        
    except Exception as e:
        logging.error(f"[{request_id}] Error in streaming request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.post("/v1/completions")
async def completions(request: CompletionRequest, http_request: Request):
    """OpenAI-compatible completions endpoint"""
    
    try:
        metrics.track_request("completions", "anonymous")
        
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream
        )
        
        response = await chat_completions(chat_request, http_request)
        
        if not request.stream:
            return {
                "id": response.id,
                "object": "text_completion",
                "created": response.created,
                "model": response.model,
                "choices": [{
                    "text": response.choices[0].message.content,
                    "index": 0,
                    "finish_reason": response.choices[0].finish_reason
                }],
                "usage": response.usage.dict()
            }
        
        return response
        
    except Exception as e:
        logging.error(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with feature detection"""
    
    try:
        health_status = await health_checker.get_health_status()
        
        # Add service-specific health checks
        services_status = []
        
        # Check Ollama connectivity
        ollama_healthy = await ollama_client.health_check()
        services_status.append({
            "name": "ollama",
            "status": "healthy" if ollama_healthy else "unhealthy",
            "last_check": datetime.now(),
            "details": {"url": ollama_client.base_url}
        })
        
        # Check available models
        try:
            models = await ollama_client.list_models()
            services_status.append({
                "name": "models",
                "status": "healthy" if models else "unhealthy",
                "last_check": datetime.now(),
                "details": {"count": len(models)}
            })
        except Exception as e:
            services_status.append({
                "name": "models",
                "status": "unhealthy",
                "last_check": datetime.now(),
                "details": {"error": str(e)}
            })
        
        # Enhanced health status
        enhanced_status = {
            **health_status,
            "services": health_status.get("services", []) + services_status,
            "features": ENHANCED_FEATURES,
            "performance": {
                "total_requests": sum(metrics.request_counts.values()),
                "avg_response_time": sum(metrics.response_times) / max(1, len(metrics.response_times)) if metrics.response_times else 0
            }
        }
        
        if not enhanced_status["healthy"]:
            raise HTTPException(status_code=503, detail="Service unhealthy")
        
        return HealthResponse(**enhanced_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive metrics with feature information"""
    
    try:
        basic_metrics = await metrics.get_all_metrics()
        
        enhanced_metrics = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "features": ENHANCED_FEATURES,
            "metrics": basic_metrics,
            "service_status": {
                "ollama_healthy": await ollama_client.health_check()
            }
        }
        
        # Add enhanced metrics if available
        if warmup_service and ENHANCED_FEATURES["model_warmup"]:
            enhanced_metrics["warmup_stats"] = warmup_service.get_warmup_stats()
        
        if hasattr(llm_router, 'get_classification_stats') and ENHANCED_FEATURES["semantic_classification"]:
            enhanced_metrics["classification_stats"] = llm_router.get_classification_stats()
        
        return enhanced_metrics
        
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_available_models():
    """List all available models with metadata"""
    
    try:
        models = await llm_router.get_available_models()
        return {"object": "list", "data": models}
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/status")
async def get_admin_status():
    """Get admin status with feature information"""
    
    try:
        return {
            "service": "LLM Proxy",
            "version": "2.1.0",
            "timestamp": datetime.now().isoformat(),
            "features": ENHANCED_FEATURES,
            "configuration": {
                "ollama_url": settings.OLLAMA_BASE_URL,
                "enable_auth": settings.ENABLE_AUTH,
                "enable_streaming": getattr(settings, 'ENABLE_STREAMING', False),
                "enable_model_warmup": getattr(settings, 'ENABLE_MODEL_WARMUP', False),
                "enable_semantic_classification": getattr(settings, 'ENABLE_SEMANTIC_CLASSIFICATION', False)
            }
        }
    except Exception as e:
        logging.error(f"Error getting admin status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/warmup/{model}")
async def manual_warmup(model: str):
    """Manually trigger model warmup (if available)"""
    
    try:
        if not warmup_service or not ENHANCED_FEATURES["model_warmup"]:
            raise HTTPException(status_code=503, detail="Model warmup service not available")
        
        await warmup_service.warmup_model(model)
        return {
            "status": "success",
            "message": f"Model {model} warmed up successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error warming up model {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "features": ENHANCED_FEATURES
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat(),
            "features": ENHANCED_FEATURES
        }
    )

# Startup event logging
@app.on_event("startup")
async def log_startup_info():
    """Log startup information with feature detection"""
    
    logging.info("=" * 70)
    logging.info("🚀 LLM Proxy - Startup Information")
    logging.info("=" * 70)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Version: 2.1.0 (Unified with Auto-Detection)")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"🎯 Available Features:")
    for feature, available in ENHANCED_FEATURES.items():
        status = "✅ Available" if available else "⏸️  Fallback"
        logging.info(f"   • {feature.replace('_', ' ').title()}: {status}")
    logging.info(f"🔗 Endpoints:")
    logging.info(f"   • Health Check: /health")
    logging.info(f"   • Chat Completions: /v1/chat/completions")
    logging.info(f"   • Completions: /v1/completions")
    logging.info(f"   • Models: /models")
    logging.info(f"   • Metrics: /metrics")
    logging.info(f"   • Admin Status: /admin/status")
    logging.info(f"   • API Docs: /docs")
    logging.info("=" * 70)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
