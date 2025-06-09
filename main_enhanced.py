# main_enhanced.py - Complete Enhanced FastAPI Application
from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn
import asyncio
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime

# Import configuration
from config_enhanced import EnhancedSettings

# Import models
from models.requests import ChatCompletionRequest, CompletionRequest
from models.responses import ChatCompletionResponse, HealthResponse

# Import enhanced services
from services.enhanced_router import EnhancedLLMRouter
from services.enhanced_ollama_client import EnhancedOllamaClient
from services.streaming import StreamingService
from services.model_warmup import ModelWarmupService
from services.semantic_cache import SemanticCache
from services.auth import AuthService

# Import middleware
from middleware import (
    AuthMiddleware,
    RateLimitMiddleware, 
    LoggingMiddleware,
    EnhancedCORSMiddleware
)

# Import utilities
from utils.metrics import MetricsCollector
from utils.health import HealthChecker
from utils.performance_monitor import PerformanceMonitor
from utils.dashboard import EnhancedDashboard
from utils.websocket_dashboard import WebSocketDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Global instances
settings = EnhancedSettings()
ollama_client = EnhancedOllamaClient()
llm_router = EnhancedLLMRouter()
streaming_service = StreamingService(ollama_client)
warmup_service = ModelWarmupService(ollama_client, llm_router)
cache_service = SemanticCache(llm_router.semantic_classifier.model if hasattr(llm_router, 'semantic_classifier') else None)
auth_service = AuthService()
metrics = MetricsCollector()
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor()

# Dashboard services
enhanced_dashboard = EnhancedDashboard(
    metrics, performance_monitor, cache_service, warmup_service, 
    llm_router.semantic_classifier if hasattr(llm_router, 'semantic_classifier') else None
)
ws_dashboard = WebSocketDashboard(enhanced_dashboard)

@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Enhanced lifespan management with all services"""
    # Startup
    logging.info("🚀 Starting Enhanced LLM Proxy Service...")
    
    try:
        # Initialize core services
        await ollama_client.initialize()
        await llm_router.initialize()
        await cache_service.initialize()
        
        # Initialize monitoring services
        await health_checker.start_monitoring()
        await performance_monitor.start_monitoring()
        
        # Initialize warmup service
        await warmup_service.start_warmup_service()
        
        # Start dashboard broadcasting
        await ws_dashboard.start_broadcasting()
        
        logging.info("✅ Enhanced LLM Proxy Service started successfully")
        
    except Exception as e:
        logging.error(f"❌ Failed to start services: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Enhanced LLM Proxy Service...")
    
    try:
        await llm_router.cleanup()
        await cache_service.cleanup()
        await health_checker.stop_monitoring()
        await performance_monitor.stop_monitoring()
        await warmup_service.stop_warmup_service()
        await ws_dashboard.stop_broadcasting()
        await ollama_client.cleanup()
        
        logging.info("✅ All services shut down gracefully")
        
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Enhanced LLM Proxy",
    description="Intelligent routing proxy for multiple LLM models with semantic classification, streaming, and cost optimization",
    version="2.0.0",
    lifespan=enhanced_lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware in correct order (reverse of execution order)
app.add_middleware(
    EnhancedCORSMiddleware, 
    allowed_origins=settings.CORS_ORIGINS,
    enable_credentials=True
)

app.add_middleware(
    LoggingMiddleware, 
    enable_detailed_logging=settings.PERFORMANCE_LOGGING
)

app.add_middleware(
    RateLimitMiddleware, 
    default_limit=60  # 60 requests per minute default
)

app.add_middleware(
    AuthMiddleware, 
    auth_service=auth_service
)

# Main API Routes
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def enhanced_chat_completions(
    request: ChatCompletionRequest,
    http_request: Request
):
    """Enhanced OpenAI-compatible chat completions with streaming support"""
    
    request_start_time = asyncio.get_event_loop().time()
    
    try:
        # Get user info from middleware
        user_info = getattr(http_request.state, 'user', {"user_id": "anonymous"})
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        # Track request
        metrics.track_request("chat_completions", user_info.get("user_id"))
        
        logging.info(f"[{request_id}] Processing chat completion request - Model: {request.model}, Stream: {request.stream}")
        
        # Handle streaming requests
        if request.stream:
            return await handle_streaming_request(request, user_info, request_id)
        
        # Check cache first for non-streaming requests
        cache_key = cache_service.generate_cache_key(request.messages, request.model)
        cached_response = await cache_service.get(cache_key)
        
        if cached_response:
            metrics.track_cache_hit()
            logging.info(f"[{request_id}] Cache hit - returning cached response")
            
            # Track performance for cached response
            response_time = asyncio.get_event_loop().time() - request_start_time
            performance_monitor.track_request_performance(
                "cached", response_time, 0, success=True
            )
            
            return cached_response
        
        # Cache miss - track it
        metrics.track_cache_miss()
        
        # Route to appropriate model using enhanced router
        selected_model = await llm_router.route_request(request)
        logging.info(f"[{request_id}] Routed to model: {selected_model}")
        
        # Record model usage for warmup service
        warmup_service.record_model_usage(selected_model)
        
        # Process request
        response = await llm_router.process_chat_completion(request, selected_model)
        
        # Calculate response time
        response_time = asyncio.get_event_loop().time() - request_start_time
        
        # Cache the response for future use
        await cache_service.set(cache_key, response, ttl=settings.CACHE_TTL)
        
        # Track usage and performance
        metrics.track_model_usage(
            selected_model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        performance_monitor.track_request_performance(
            selected_model, 
            response_time, 
            response.usage.completion_tokens,
            success=True
        )
        
        # Add metadata to response
        response.cache_hit = False
        response.processing_time = response_time
        
        logging.info(f"[{request_id}] Request completed successfully - Duration: {response_time:.3f}s")
        
        return response
        
    except Exception as e:
        # Calculate error response time
        error_response_time = asyncio.get_event_loop().time() - request_start_time
        
        # Track error
        metrics.track_error(type(e).__name__)
        performance_monitor.track_request_performance(
            "error", error_response_time, 0, success=False
        )
        
        logging.error(f"[{getattr(http_request.state, 'request_id', 'unknown')}] Error in chat_completions: {str(e)}")
        
        raise HTTPException(status_code=500, detail=str(e))

async def handle_streaming_request(
    request: ChatCompletionRequest,
    user_info: dict,
    request_id: str
) -> StreamingResponse:
    """Handle streaming chat completion requests"""
    
    try:
        # Route to appropriate model
        selected_model = await llm_router.route_request(request)
        logging.info(f"[{request_id}] Streaming request routed to: {selected_model}")
        
        # Record model usage
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
        user_info = getattr(http_request.state, 'user', {"user_id": "anonymous"})
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        metrics.track_request("completions", user_info.get("user_id"))
        
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
        
        # Add enhanced service status
        enhanced_status = {
            **health_status,
            "enhanced_features": {
                "semantic_classification": {
                    "enabled": settings.ENABLE_SEMANTIC_CLASSIFICATION,
                    "model_loaded": hasattr(llm_router, 'semantic_classifier') and 
                                   llm_router.semantic_classifier.model is not None
                },
                "streaming": {
                    "enabled": settings.ENABLE_STREAMING
                },
                "model_warmup": {
                    "enabled": settings.ENABLE_MODEL_WARMUP,
                    "active_models": len(warmup_service.model_last_used)
                },
                "caching": {
                    "enabled": True,
                    "cache_size": len(cache_service.cache_store) if hasattr(cache_service, 'cache_store') else 0
                }
            },
            "performance": {
                "total_requests": sum(metrics.request_counts.values()),
                "cache_hit_rate": metrics.cache_hits / max(1, metrics.cache_hits + metrics.cache_misses),
                "avg_response_time": sum(metrics.response_times) / max(1, len(metrics.response_times))
            }
        }
        
        if not enhanced_status["healthy"]:
            raise HTTPException(status_code=503, detail="Service unhealthy")
        
        return HealthResponse(**enhanced_status)
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def get_comprehensive_metrics():
    """Get comprehensive system metrics"""
    
    try:
        # Require authentication for metrics
        # This will be handled by the auth middleware
        
        basic_metrics = await metrics.get_all_metrics()
        performance_data = await performance_monitor.get_current_performance_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "basic_metrics": basic_metrics,
            "performance": performance_data,
            "cache_stats": await cache_service.get_stats() if hasattr(cache_service, 'get_stats') else {},
            "warmup_stats": warmup_service.get_warmup_stats(),
            "classification_stats": llm_router.semantic_classifier.get_classification_stats() 
                                  if hasattr(llm_router, 'semantic_classifier') else {}
        }
        
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

# Admin endpoints (require admin authentication)
@app.get("/admin/dashboard")
async def get_admin_dashboard():
    """Get comprehensive admin dashboard data"""
    
    try:
        dashboard_data = await enhanced_dashboard.get_comprehensive_dashboard()
        return dashboard_data
        
    except Exception as e:
        logging.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/classification/stats")
async def get_classification_stats():
    """Get semantic classification statistics"""
    
    try:
        if hasattr(llm_router, 'semantic_classifier'):
            return llm_router.semantic_classifier.get_classification_stats()
        else:
            return {"error": "Semantic classification not enabled"}
            
    except Exception as e:
        logging.error(f"Error getting classification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/warmup/stats")
async def get_warmup_stats():
    """Get model warmup statistics"""
    
    try:
        return warmup_service.get_warmup_stats()
        
    except Exception as e:
        logging.error(f"Error getting warmup stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/warmup/{model}")
async def manual_warmup(model: str):
    """Manually trigger model warmup"""
    
    try:
        await warmup_service.warmup_model(model)
        return {
            "status": "success", 
            "message": f"Model {model} warmed up successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error warming up model {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/performance")
async def get_performance_summary():
    """Get detailed performance summary"""
    
    try:
        performance_data = await performance_monitor.get_current_performance_summary()
        recommendations = performance_monitor.get_optimization_recommendations()
        
        return {
            "performance": performance_data,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time dashboard
@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    
    try:
        await ws_dashboard.connect(websocket)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "get_dashboard":
                    # Send immediate dashboard update
                    dashboard_data = await enhanced_dashboard.get_comprehensive_dashboard()
                    await websocket.send_text(json.dumps({
                        "type": "dashboard_update",
                        "data": dashboard_data
                    }))
                
                elif message.get("type") == "get_metrics":
                    # Send metrics data
                    metrics_data = await get_comprehensive_metrics()
                    await websocket.send_text(json.dumps({
                        "type": "metrics_update",
                        "data": metrics_data
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": str(e)
                }))
                break
                
    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")
    finally:
        ws_dashboard.disconnect(websocket)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors"""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logging.error(f"[{request_id}] Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id,
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
    logging.info(f"   • Semantic Classification: {settings.ENABLE_SEMANTIC_CLASSIFICATION}")
    logging.info(f"   • Streaming Support: {settings.ENABLE_STREAMING}")
    logging.info(f"   • Model Warmup: {settings.ENABLE_MODEL_WARMUP}")
    logging.info(f"   • Authentication: {settings.ENABLE_AUTH}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"🔗 Endpoints:")
    logging.info(f"   • Health Check: /health")
    logging.info(f"   • Chat Completions: /v1/chat/completions")
    logging.info(f"   • Completions: /v1/completions")
    logging.info(f"   • Models: /models")
    logging.info(f"   • Metrics: /metrics")
    logging.info(f"   • Dashboard: /admin/dashboard")
    logging.info(f"   • WebSocket: /ws/dashboard")
    logging.info(f"   • API Docs: /docs")
    logging.info("=" * 60)

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.PERFORMANCE_LOGGING
    )