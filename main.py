# main_enhanced.py - Fixed and Simplified Enhanced FastAPI Application
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

# Import services with error handling
try:
    from services.enhanced_router import EnhancedLLMRouter
    from services.enhanced_ollama_client import EnhancedOllamaClient
    from services.streaming import StreamingService
    from services.model_warmup import ModelWarmupService
    from services.auth import AuthService
    ENHANCED_SERVICES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced services not available: {e}")
    # Fallback to basic services
    from services.router import LLMRouter as EnhancedLLMRouter
    from services.ollama_client import OllamaClient as EnhancedOllamaClient
    ENHANCED_SERVICES_AVAILABLE = False

# Import utilities
from utils.metrics import MetricsCollector
from utils.health import HealthChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/app.log') if sys.platform != 'win32' else logging.StreamHandler()
    ]
)

# Global instances with error handling
settings = EnhancedSettings()
ollama_client = EnhancedOllamaClient()
llm_router = EnhancedLLMRouter(ollama_client)
metrics = MetricsCollector()
health_checker = HealthChecker()

# Optional services
streaming_service = None
warmup_service = None
auth_service = None

@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Enhanced lifespan management with proper error handling"""
    # Startup
    logging.info("🚀 Starting Enhanced LLM Proxy Service...")
    
    startup_errors = []
    
    try:
        # Initialize core services
        await ollama_client.initialize()
        await llm_router.initialize()
        logging.info("✅ Core services initialized")
        
        # Initialize optional services
        await initialize_optional_services()
        
        # Start monitoring
        await health_checker.start_monitoring()
        logging.info("✅ Health monitoring started")
        
        logging.info("✅ Enhanced LLM Proxy Service started successfully")
        
    except Exception as e:
        logging.error(f"❌ Failed to start services: {e}")
        logging.error(traceback.format_exc())
        startup_errors.append(str(e))
        # Continue with limited functionality
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Enhanced LLM Proxy Service...")
    
    try:
        await cleanup_services()
        logging.info("✅ All services shut down gracefully")
        
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

async def initialize_optional_services():
    """Initialize optional services with fallbacks"""
    global streaming_service, warmup_service, auth_service
    
    try:
        if ENHANCED_SERVICES_AVAILABLE and settings.ENABLE_STREAMING:
            streaming_service = StreamingService(ollama_client)
            logging.info("✅ Streaming service initialized")
    except Exception as e:
        logging.warning(f"Failed to initialize streaming service: {e}")
    
    try:
        if ENHANCED_SERVICES_AVAILABLE and settings.ENABLE_MODEL_WARMUP:
            warmup_service = ModelWarmupService(ollama_client, llm_router)
            await warmup_service.start_warmup_service()
            logging.info("✅ Model warmup service initialized")
    except Exception as e:
        logging.warning(f"Failed to initialize warmup service: {e}")
    
    try:
        auth_service = AuthService(settings)
        logging.info("✅ Authentication service initialized")
    except Exception as e:
        logging.warning(f"Failed to initialize auth service: {e}")

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

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Enhanced LLM Proxy",
    description="Intelligent routing proxy for multiple LLM models with semantic classification, streaming, and cost optimization",
    version="2.0.0",
    lifespan=enhanced_lifespan,
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
async def enhanced_chat_completions(
    request: ChatCompletionRequest,
    http_request: Request
):
    """Enhanced OpenAI-compatible chat completions with streaming support"""
    
    request_start_time = asyncio.get_event_loop().time()
    request_id = f"req_{int(request_start_time * 1000000) % 1000000}"  # Simple ID
    
    try:
        # Track request
        metrics.track_request("chat_completions", "anonymous")
        
        logging.info(f"[{request_id}] Processing chat completion request - Model: {request.model}, Stream: {request.stream}")
        
        # Handle streaming requests
        if request.stream and streaming_service:
            return await handle_streaming_request(request, request_id)
        
        # Route to appropriate model using enhanced router
        selected_model = await llm_router.route_request(request)
        logging.info(f"[{request_id}] Routed to model: {selected_model}")
        
        # Record model usage for warmup service
        if warmup_service:
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
        
        logging.info(f"[{request_id}] Request completed successfully - Duration: {response_time:.3f}s")
        
        return response
        
    except Exception as e:
        # Calculate error response time
        error_response_time = asyncio.get_event_loop().time() - request_start_time
        
        # Track error
        metrics.track_error(type(e).__name__)
        
        logging.error(f"[{request_id}] Error in chat_completions: {str(e)}")
        logging.error(traceback.format_exc())
        
        raise HTTPException(status_code=500, detail=str(e))

async def handle_streaming_request(
    request: ChatCompletionRequest,
    request_id: str
) -> StreamingResponse:
    """Handle streaming chat completion requests"""
    
    try:
        # Route to appropriate model
        selected_model = await llm_router.route_request(request)
        logging.info(f"[{request_id}] Streaming request routed to: {selected_model}")
        
        # Record model usage
        if warmup_service:
            warmup_service.record_model_usage(selected_model)
        
        # Prepare request data for streaming
        request_data = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens
        }
        
        # Create streaming response
        return StreamingResponse(
            streaming_service.stream_chat_completion(request_data, selected_model),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering
                "X-Request-ID": request_id,
                "X-Selected-Model": selected_model
            }
        )
        
    except Exception as e:
        logging.error(f"[{request_id}] Error in streaming request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.post("/v1/completions")
async def enhanced_completions(
    request: CompletionRequest,
    http_request: Request
):
    """Enhanced OpenAI-compatible completions endpoint"""
    
    try:
        metrics.track_request("completions", "anonymous")
        
        # Convert to chat format for unified processing
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream
        )
        
        # Use chat completions logic
        response = await enhanced_chat_completions(chat_request, http_request)
        
        # Convert back to completions format if not streaming
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
    """Comprehensive health check endpoint"""
    
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
        
        # Add enhanced service status
        enhanced_status = {
            **health_status,
            "services": health_status.get("services", []) + services_status,
            "enhanced_features": {
                "streaming": {
                    "enabled": streaming_service is not None
                },
                "model_warmup": {
                    "enabled": warmup_service is not None,
                    "active_models": len(warmup_service.model_last_used) if warmup_service else 0
                },
                "semantic_classification": {
                    "enabled": hasattr(llm_router, 'semantic_classifier') and 
                              llm_router.semantic_classifier is not None
                }
            },
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
async def get_comprehensive_metrics():
    """Get comprehensive system metrics"""
    
    try:
        basic_metrics = await metrics.get_all_metrics()
        
        enhanced_metrics = {
            "timestamp": datetime.now().isoformat(),
            "basic_metrics": basic_metrics,
            "service_status": {
                "ollama_healthy": await ollama_client.health_check(),
                "enhanced_services_available": ENHANCED_SERVICES_AVAILABLE,
                "streaming_enabled": streaming_service is not None,
                "warmup_enabled": warmup_service is not None
            }
        }
        
        # Add warmup stats if available
        if warmup_service:
            enhanced_metrics["warmup_stats"] = warmup_service.get_warmup_stats()
        
        # Add classification stats if available
        if hasattr(llm_router, 'get_classification_stats'):
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
        
        return {
            "object": "list",
            "data": models
        }
        
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoints (basic versions)
@app.get("/admin/status")
async def get_admin_status():
    """Get admin status information"""
    
    try:
        return {
            "service": "Enhanced LLM Proxy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "enhanced_services_available": ENHANCED_SERVICES_AVAILABLE,
            "features": {
                "streaming": streaming_service is not None,
                "warmup": warmup_service is not None,
                "auth": auth_service is not None,
                "semantic_classification": hasattr(llm_router, 'semantic_classifier')
            }
        }
        
    except Exception as e:
        logging.error(f"Error getting admin status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/warmup/{model}")
async def manual_warmup(model: str):
    """Manually trigger model warmup"""
    
    try:
        if not warmup_service:
            raise HTTPException(status_code=503, detail="Warmup service not available")
        
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
    """Custom HTTP exception handler"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors"""
    
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event logging
@app.on_event("startup")
async def log_startup_info():
    """Log startup information"""
    
    logging.info("=" * 60)
    logging.info("🚀 Enhanced LLM Proxy - Startup Information")
    logging.info("=" * 60)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Enhanced Services Available: {ENHANCED_SERVICES_AVAILABLE}")
    logging.info(f"   • Streaming Support: {streaming_service is not None}")
    logging.info(f"   • Model Warmup: {warmup_service is not None}")
    logging.info(f"   • Authentication: {auth_service is not None}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"🔗 Endpoints:")
    logging.info(f"   • Health Check: /health")
    logging.info(f"   • Chat Completions: /v1/chat/completions")
    logging.info(f"   • Completions: /v1/completions")
    logging.info(f"   • Models: /models")
    logging.info(f"   • Metrics: /metrics")
    logging.info(f"   • Admin Status: /admin/status")
    logging.info(f"   • API Docs: /docs")
    logging.info("=" * 60)

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
