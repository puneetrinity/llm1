# main.py - Complete LLM Proxy with All Enhancements
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import asyncio
import logging
import sys
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any

# Import configuration with security
from config_enhanced import EnhancedSettings
try:
    from security.config import SecuritySettings, setup_security_middleware, validate_production_config
    SECURITY_AVAILABLE = True
except ImportError:
    logging.warning("Security module not available - using basic configuration")
    SecuritySettings = None
    SECURITY_AVAILABLE = False

# Import models
from models.requests import ChatCompletionRequest, CompletionRequest
from models.responses import ChatCompletionResponse, HealthResponse

# Import enhanced systems
from utils.memory_manager import get_memory_manager, MemoryManager
from utils.error_handler import (
    handle_errors, ErrorContext, LLMProxyError, 
    OllamaConnectionError, ModelNotFoundError, ValidationError,
    request_context, model_context, ollama_context
)
from services.enhanced_imports import setup_enhanced_imports, import_manager
from utils.metrics import MetricsCollector
from utils.health import HealthChecker

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/app/logs/app.log') if sys.platform != 'win32' else logging.StreamHandler()
    ]
)

# Global settings and memory manager
settings = EnhancedSettings()
if SECURITY_AVAILABLE:
    security_settings = SecuritySettings()
    # Validate production configuration
    if security_settings.ENVIRONMENT == "production":
        config_issues = validate_production_config(security_settings)
        if config_issues:
            for issue in config_issues:
                logging.error(f"🚨 Security Issue: {issue}")
            if any("CRITICAL" in issue for issue in config_issues):
                raise RuntimeError("Critical security issues detected. Cannot start in production mode.")

memory_manager = get_memory_manager(settings.MAX_MEMORY_MB)

# Global service instances
ollama_client = None
llm_router = None
metrics = MetricsCollector()
health_checker = HealthChecker()
auth_service = None

# Enhanced service instances (set during initialization)
streaming_service = None
warmup_service = None
enhanced_capabilities = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Complete lifespan management with all enhancements"""
    
    # Startup
    logging.info("🚀 Starting Complete LLM Proxy Service...")
    
    try:
        # Initialize memory management
        await memory_manager.start_monitoring()
        logging.info("✅ Memory manager started")
        
        # Setup enhanced imports with fallbacks
        await setup_all_services()
        
        # Start health monitoring
        await health_checker.start_monitoring()
        logging.info("✅ Health monitoring started")
        
        # Log startup summary
        import_manager.log_startup_summary()
        log_memory_allocation()
        
        if SECURITY_AVAILABLE and security_settings.ENVIRONMENT == "production":
            logging.info("🔒 Production security mode active")
        
        logging.info("✅ Complete LLM Proxy Service started successfully")
        
    except Exception as e:
        logging.error(f"❌ Failed to start services: {e}")
        logging.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Complete LLM Proxy Service...")
    
    try:
        await cleanup_all_services()
        await memory_manager.stop_monitoring()
        await health_checker.stop_monitoring()
        logging.info("✅ All services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

async def setup_all_services():
    """Setup all services with enhanced imports and error handling"""
    global ollama_client, llm_router, auth_service, streaming_service, warmup_service, enhanced_capabilities
    
    # Setup enhanced imports
    enhanced_imports = setup_enhanced_imports()
    enhanced_capabilities = enhanced_imports['capabilities']
    
    # Initialize core services with memory allocation
    if not memory_manager.allocate('core_services', 512):  # 512MB for core services
        raise MemoryError("Insufficient memory for core services")
    
    try:
        # Initialize Ollama client
        ollama_client = enhanced_imports['EnhancedOllamaClient'](
            settings.OLLAMA_BASE_URL, 
            settings.OLLAMA_TIMEOUT
        )
        await ollama_client.initialize()
        logging.info("✅ Ollama client initialized")
        
        # Initialize router
        llm_router = enhanced_imports['EnhancedLLMRouter'](ollama_client)
        await llm_router.initialize()
        logging.info("✅ LLM router initialized")
        
        # Initialize authentication service
        if SECURITY_AVAILABLE:
            from services.auth import AuthService
            auth_service = AuthService(security_settings)
        else:
            from services.auth import AuthService
            auth_service = AuthService(settings)
        logging.info("✅ Authentication service initialized")
        
        # Initialize enhanced services if available
        await setup_enhanced_services(enhanced_imports)
        
    except Exception as e:
        memory_manager.deallocate('core_services')
        raise

async def setup_enhanced_services(enhanced_imports):
    """Setup enhanced services with proper memory management"""
    global streaming_service, warmup_service
    
    # Streaming service
    if enhanced_imports['StreamingService'] and enhanced_capabilities['streaming'] and settings.ENABLE_STREAMING:
        try:
            if memory_manager.allocate('streaming', 256):  # 256MB for streaming
                streaming_service = enhanced_imports['StreamingService'](ollama_client)
                logging.info("✅ Streaming service initialized")
            else:
                logging.warning("⚠️  Insufficient memory for streaming service")
        except Exception as e:
            logging.warning(f"⚠️  Failed to initialize streaming: {e}")
            memory_manager.deallocate('streaming')
    
    # Model warmup service
    if enhanced_imports['ModelWarmupService'] and enhanced_capabilities['model_warmup'] and settings.ENABLE_MODEL_WARMUP:
        try:
            if memory_manager.allocate('warmup', 128):  # 128MB for warmup
                warmup_service = enhanced_imports['ModelWarmupService'](ollama_client, llm_router)
                await warmup_service.start_warmup_service()
                logging.info("✅ Model warmup service initialized")
            else:
                logging.warning("⚠️  Insufficient memory for warmup service")
        except Exception as e:
            logging.warning(f"⚠️  Failed to initialize warmup: {e}")
            memory_manager.deallocate('warmup')

async def cleanup_all_services():
    """Cleanup all services with memory deallocation"""
    
    try:
        if warmup_service:
            await warmup_service.stop_warmup_service()
            memory_manager.deallocate('warmup')
    except Exception as e:
        logging.error(f"Error cleaning up warmup service: {e}")
    
    try:
        if llm_router:
            await llm_router.cleanup()
    except Exception as e:
        logging.error(f"Error cleaning up router: {e}")
    
    try:
        if ollama_client:
            await ollama_client.cleanup()
    except Exception as e:
        logging.error(f"Error cleaning up ollama client: {e}")
    
    # Deallocate core services memory
    memory_manager.deallocate('core_services')
    memory_manager.deallocate('streaming')

def log_memory_allocation():
    """Log memory allocation status"""
    status = memory_manager.get_status_summary()
    logging.info("💾 Memory Allocation Summary:")
    logging.info(f"   • Total Limit: {status['limits']['total_mb']}MB")
    logging.info(f"   • Available: {status['usage']['available_mb']}MB")
    logging.info(f"   • Utilization: {status['health']['utilization_percent']:.1f}%")
    for component, mb in status['usage']['tracked_components'].items():
        logging.info(f"   • {component.title()}: {mb}MB")

# Create FastAPI app
app = FastAPI(
    title="Complete LLM Proxy",
    description="Production-ready LLM routing proxy with enhanced features, security, and monitoring",
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup security middleware
if SECURITY_AVAILABLE:
    setup_security_middleware(app, security_settings)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Optional middleware (import only if available)
try:
    if settings.ENABLE_AUTH and auth_service:
        from middleware.auth import AuthMiddleware
        app.add_middleware(AuthMiddleware, auth_service=auth_service)
        logging.info("✅ Authentication middleware enabled")
except ImportError:
    logging.info("ℹ️  Authentication middleware not available")

try:
    if settings.ENABLE_RATE_LIMITING:
        from middleware.rate_limit import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware, default_limit=settings.DEFAULT_RATE_LIMIT)
        logging.info("✅ Rate limiting middleware enabled")
except ImportError:
    logging.info("ℹ️  Rate limiting middleware not available")

try:
    from middleware.logging import LoggingMiddleware
    app.add_middleware(LoggingMiddleware, enable_detailed_logging=settings.DEBUG)
    logging.info("✅ Logging middleware enabled")
except ImportError:
    logging.info("ℹ️  Logging middleware not available")

# Authentication dependency
async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get current user from request (if authentication is enabled)"""
    if hasattr(request.state, 'user'):
        return request.state.user
    return None

# Main API Routes with complete error handling
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@handle_errors(context_func=request_context)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Complete chat completions with all enhancements"""
    
    request_start_time = asyncio.get_event_loop().time()
    request_id = f"req_{int(request_start_time * 1000000) % 1000000}"
    
    async with ErrorContext(
        context={
            "component": "chat_completions",
            "model": request.model,
            "stream": request.stream,
            "user_id": current_user.get('user_id') if current_user else 'anonymous'
        },
        request_id=request_id
    ):
        # Track request
        user_id = current_user.get('user_id') if current_user else 'anonymous'
        metrics.track_request("chat_completions", user_id)
        
        logging.info(f"[{request_id}] Processing chat completion - Model: {request.model}, Stream: {request.stream}")
        
        # Check memory before processing
        if not memory_manager.check_allocation('request_processing', 64):  # 64MB per request
            raise MemoryError("Insufficient memory to process request")
        
        try:
            memory_manager.allocate('request_processing', 64)
            
            # Handle streaming if available and requested
            if request.stream and streaming_service and enhanced_capabilities.get('streaming'):
                return await handle_streaming_request(request, request_id, user_id)
            elif request.stream and not enhanced_capabilities.get('streaming'):
                logging.warning(f"[{request_id}] Streaming requested but not available - falling back to non-streaming")
                request.stream = False
            
            # Route to appropriate model
            selected_model = await llm_router.route_request(request)
            logging.info(f"[{request_id}] Routed to model: {selected_model}")
            
            # Record model usage for warmup if available
            if warmup_service and enhanced_capabilities.get('model_warmup'):
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
            
        finally:
            memory_manager.deallocate('request_processing', 64)

@handle_errors(context_func=request_context)
async def handle_streaming_request(
    request: ChatCompletionRequest, 
    request_id: str,
    user_id: str
) -> StreamingResponse:
    """Handle streaming requests with complete error handling"""
    
    try:
        selected_model = await llm_router.route_request(request)
        logging.info(f"[{request_id}] Streaming request routed to: {selected_model}")
        
        if warmup_service and enhanced_capabilities.get('model_warmup'):
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
                "X-Selected-Model": selected_model,
                "X-User-ID": user_id
            }
        )
        
    except Exception as e:
        logging.error(f"[{request_id}] Error in streaming request: {str(e)}")
        raise

@app.post("/v1/completions")
@handle_errors(context_func=request_context)
async def completions(
    request: CompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Complete completions endpoint"""
    
    try:
        user_id = current_user.get('user_id') if current_user else 'anonymous'
        metrics.track_request("completions", user_id)
        
        # Convert to chat format for unified processing
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream
        )
        
        response = await chat_completions(chat_request, http_request, current_user)
        
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
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with all systems"""
    
    try:
        health_status = await health_checker.get_health_status()
        
        # Add service-specific health checks
        services_status = []
        
        # Check Ollama connectivity
        if ollama_client:
            ollama_healthy = await ollama_client.health_check()
            services_status.append({
                "name": "ollama",
                "status": "healthy" if ollama_healthy else "unhealthy",
                "last_check": datetime.now(),
                "details": {"url": ollama_client.base_url}
            })
        
        # Check available models
        if llm_router:
            try:
                models = await llm_router.get_available_models()
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
        
        # Memory status
        memory_status = memory_manager.get_status_summary()
        services_status.append({
            "name": "memory",
            "status": memory_status['health']['status'],
            "last_check": datetime.now(),
            "details": {
                "utilization_percent": memory_status['health']['utilization_percent'],
                "available_mb": memory_status['health']['available_mb']
            }
        })
        
        # Enhanced health status
        enhanced_status = {
            **health_status,
            "services": health_status.get("services", []) + services_status,
            "features": {
                **enhanced_capabilities,
                "security": SECURITY_AVAILABLE,
                "memory_management": True,
                "error_handling": True
            },
            "performance": {
                "total_requests": sum(metrics.request_counts.values()),
                "avg_response_time": sum(metrics.response_times) / max(1, len(metrics.response_times)) if metrics.response_times else 0
            },
            "memory": memory_status['health']
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
    """Get comprehensive metrics including all systems"""
    
    try:
        basic_metrics = await metrics.get_all_metrics()
        memory_status = memory_manager.get_status_summary()
        
        comprehensive_metrics = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "features": {
                **enhanced_capabilities,
                "security": SECURITY_AVAILABLE,
                "memory_management": True,
                "error_handling": True
            },
            "metrics": basic_metrics,
            "memory": memory_status,
            "service_status": {
                "ollama_healthy": await ollama_client.health_check() if ollama_client else False,
                "import_status": import_manager.get_feature_status()
            }
        }
        
        # Add enhanced metrics if available
        if warmup_service and enhanced_capabilities.get('model_warmup'):
            comprehensive_metrics["warmup_stats"] = warmup_service.get_warmup_stats()
        
        if hasattr(llm_router, 'get_classification_stats') and enhanced_capabilities.get('semantic_classification'):
            comprehensive_metrics["classification_stats"] = llm_router.get_classification_stats()
        
        return comprehensive_metrics
        
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
@handle_errors()
async def list_available_models():
    """List all available models with complete metadata"""
    
    if not llm_router:
        raise ServiceUnavailableError("LLM Router")
    
    models = await llm_router.get_available_models()
    return {"object": "list", "data": models}

# Admin endpoints
@app.get("/admin/status")
async def get_admin_status():
    """Get complete admin status"""
    
    try:
        memory_status = memory_manager.get_status_summary()
        import_status = import_manager.get_feature_status()
        
        return {
            "service": "Complete LLM Proxy",
            "version": "2.2.0",
            "timestamp": datetime.now().isoformat(),
            "features": {
                **enhanced_capabilities,
                "security": SECURITY_AVAILABLE,
                "memory_management": True,
                "error_handling": True
            },
            "configuration": {
                "ollama_url": settings.OLLAMA_BASE_URL,
                "environment": getattr(settings, 'ENVIRONMENT', 'development'),
                "enable_auth": settings.ENABLE_AUTH,
                "memory_limit_mb": settings.MAX_MEMORY_MB
            },
            "memory": memory_status,
            "imports": import_status,
            "security": {
                "enabled": SECURITY_AVAILABLE,
                "environment": getattr(security_settings, 'ENVIRONMENT', 'unknown') if SECURITY_AVAILABLE else 'unknown'
            }
        }
    except Exception as e:
        logging.error(f"Error getting admin status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/warmup/{model}")
@handle_errors(context_func=model_context)
async def manual_warmup(model: str):
    """Manually trigger model warmup"""
    
    if not warmup_service or not enhanced_capabilities.get('model_warmup'):
        raise ServiceUnavailableError("Model warmup service")
    
    await warmup_service.warmup_model(model)
    return {
        "status": "success",
        "message": f"Model {model} warmed up successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/admin/memory")
async def get_memory_status():
    """Get detailed memory status"""
    
    try:
        status = memory_manager.get_status_summary()
        return {
            "status": "success",
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/memory/cleanup")
async def trigger_memory_cleanup():
    """Trigger manual memory cleanup"""
    
    try:
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory status after cleanup
        status = memory_manager.get_status_summary()
        
        return {
            "status": "success",
            "message": f"Cleanup completed. Collected {collected} objects.",
            "memory_status": status['health'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error during memory cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(LLMProxyError)
async def llm_proxy_error_handler(request: Request, exc: LLMProxyError):
    """Handle standardized LLM Proxy errors"""
    
    return JSONResponse(
        status_code=500 if exc.severity.value in ['high', 'critical'] else 400,
        content={
            **exc.to_dict(),
            "request_path": str(request.url.path),
            "version": "2.2.0"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "request_path": str(request.url.path),
            "version": "2.2.0",
            "features": enhanced_capabilities
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An internal error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat(),
            "request_path": str(request.url.path),
            "version": "2.2.0",
            "features": enhanced_capabilities
        }
    )

# Startup event logging
@app.on_event("startup")
async def log_startup_info():
    """Log comprehensive startup information"""
    
    logging.info("=" * 80)
    logging.info("🚀 Complete LLM Proxy - Startup Information")
    logging.info("=" * 80)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Version: 2.2.0 (Complete with All Enhancements)")
    logging.info(f"   • Environment: {getattr(security_settings, 'ENVIRONMENT', 'development') if SECURITY_AVAILABLE else 'development'}")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Memory Limit: {settings.MAX_MEMORY_MB}MB")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"🔒 Security:")
    logging.info(f"   • Security Module: {'✅ Available' if SECURITY_AVAILABLE else '⏸️  Basic'}")
    logging.info(f"   • Authentication: {'✅ Enabled' if settings.ENABLE_AUTH else '⏸️  Disabled'}")
    logging.info(f"   • Rate Limiting: {'✅ Enabled' if settings.ENABLE_RATE_LIMITING else '⏸️  Disabled'}")
    logging.info(f"🎯 Available Features:")
    for feature, available in enhanced_capabilities.items():
        status = "✅ Available" if available else "⏸️  Fallback"
        logging.info(f"   • {feature.replace('_', ' ').title()}: {status}")
    logging.info(f"🔗 Endpoints:")
    logging.info(f"   • Health Check: /health")
    logging.info(f"   • Chat Completions: /v1/chat/completions")
    logging.info(f"   • Completions: /v1/completions")
    logging.info(f"   • Models: /models")
    logging.info(f"   • Metrics: /metrics")
    logging.info(f"   • Admin Status: /admin/status")
    logging.info(f"   • Memory Status: /admin/memory")
    logging.info(f"   • API Docs: /docs")
    logging.info("=" * 80)

if __name__ == "__main__":
    # Run the complete application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
