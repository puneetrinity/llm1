# main.py - Fixed LLM Proxy with Proper Configuration
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

# Import configuration with fallback handling
try:
    from config_enhanced import get_settings
    settings = get_settings()
except ImportError:
    # Fallback to basic configuration
    from pydantic_settings import BaseSettings
    
    class BasicSettings(BaseSettings):
        DEBUG: bool = False
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        MAX_MEMORY_MB: int = 4096
        ENABLE_STREAMING: bool = True
        ENABLE_AUTH: bool = False
        ENABLE_RATE_LIMITING: bool = False
        CORS_ORIGINS: list = ["*"]
        CORS_ALLOW_CREDENTIALS: bool = True
        DEFAULT_RATE_LIMIT: str = "100/hour"
        LOG_LEVEL: str = "INFO"
        
        class Config:
            env_file = '.env'
            extra = 'ignore'  # This is the key fix!
    
    settings = BasicSettings()

# Try to import security settings with fallback
SECURITY_AVAILABLE = False
security_settings = None

try:
    from security.config import SecuritySettings, setup_security_middleware, validate_production_config
    security_settings = SecuritySettings()
    SECURITY_AVAILABLE = True
    
    # Validate production configuration
    if hasattr(security_settings, 'ENVIRONMENT') and security_settings.ENVIRONMENT == "production":
        config_issues = validate_production_config(security_settings)
        if config_issues:
            for issue in config_issues:
                logging.error(f"🚨 Security Issue: {issue}")
            if any("CRITICAL" in issue for issue in config_issues):
                raise RuntimeError("Critical security issues detected. Cannot start in production mode.")
                
except ImportError as e:
    logging.warning(f"Security module not available - using basic configuration: {e}")
    # Create a basic security settings fallback
    class BasicSecuritySettings:
        ENVIRONMENT = "development"
        SECURITY_HEADERS_ENABLED = False
        USE_TLS = False
        CORS_ORIGINS = ["*"]
        RATE_LIMIT_ENABLED = False
    
    security_settings = BasicSecuritySettings()

# Configure logging with proper error handling
try:
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
except AttributeError:
    log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Try to import models with fallback
try:
    from models.requests import ChatCompletionRequest, CompletionRequest
    from models.responses import ChatCompletionResponse, HealthResponse
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
    
    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        timestamp: str
        services: List[Dict[str, Any]] = []

# Global service instances with None initialization
memory_manager = None
ollama_client = None
llm_router = None
metrics = None
health_checker = None
auth_service = None
streaming_service = None
warmup_service = None
enhanced_capabilities = {}

# Try to initialize services with fallbacks
async def initialize_core_services():
    """Initialize core services with fallback handling"""
    global ollama_client, llm_router, memory_manager, metrics, health_checker, auth_service
    global enhanced_capabilities
    
    try:
        # Initialize memory manager
        try:
            from utils.memory_manager import get_memory_manager
            memory_manager = get_memory_manager(settings.MAX_MEMORY_MB)
            await memory_manager.start_monitoring()
            logging.info("✅ Memory manager initialized")
        except ImportError:
            logging.warning("Memory manager not available - using basic memory tracking")
            memory_manager = None
        
        # Initialize metrics
        try:
            from utils.metrics import MetricsCollector
            metrics = MetricsCollector()
            logging.info("✅ Metrics collector initialized")
        except ImportError:
            logging.warning("Metrics collector not available")
            metrics = None
        
        # Initialize health checker
        try:
            from utils.health import HealthChecker
            health_checker = HealthChecker()
            await health_checker.start_monitoring()
            logging.info("✅ Health checker initialized")
        except ImportError:
            logging.warning("Health checker not available")
            health_checker = None
        
        # Initialize Ollama client
        try:
            from services.enhanced_imports import setup_enhanced_imports
            enhanced_imports = setup_enhanced_imports()
            enhanced_capabilities = enhanced_imports['capabilities']
            
            ollama_client = enhanced_imports['EnhancedOllamaClient'](
                settings.OLLAMA_BASE_URL, 
                getattr(settings, 'OLLAMA_TIMEOUT', 300)
            )
            await ollama_client.initialize()
            logging.info("✅ Enhanced Ollama client initialized")
        except Exception as e:
            logging.warning(f"Enhanced Ollama client failed, using basic client: {e}")
            # Fallback to basic client
            try:
                from services.ollama_client import BasicOllamaClient
                ollama_client = BasicOllamaClient(settings.OLLAMA_BASE_URL)
                enhanced_capabilities = {"streaming": False, "model_warmup": False}
                logging.info("✅ Basic Ollama client initialized")
            except Exception as e2:
                logging.error(f"Failed to initialize any Ollama client: {e2}")
                ollama_client = None
        
        # Initialize router
        if ollama_client:
            try:
                from services.llm_router import EnhancedLLMRouter
                llm_router = EnhancedLLMRouter(ollama_client)
                await llm_router.initialize()
                logging.info("✅ LLM router initialized")
            except Exception as e:
                logging.warning(f"Enhanced router failed, using basic router: {e}")
                llm_router = None
        
        # Initialize authentication service
        try:
            if SECURITY_AVAILABLE:
                from services.auth import AuthService
                auth_service = AuthService(security_settings)
            else:
                from services.auth import BasicAuthService
                auth_service = BasicAuthService()
            logging.info("✅ Authentication service initialized")
        except Exception as e:
            logging.warning(f"Authentication service failed: {e}")
            auth_service = None
            
    except Exception as e:
        logging.error(f"Failed to initialize core services: {e}")
        logging.error(traceback.format_exc())
        # Don't raise - allow app to start with limited functionality

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper error handling"""
    
    # Startup
    logging.info("🚀 Starting LLM Proxy Service...")
    
    try:
        await initialize_core_services()
        logging.info("✅ Core services initialized")
        
        # Log startup summary
        log_startup_summary()
        
    except Exception as e:
        logging.error(f"❌ Failed to start some services: {e}")
        # Don't fail completely - allow partial startup
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down LLM Proxy Service...")
    
    try:
        if memory_manager:
            await memory_manager.stop_monitoring()
        if health_checker:
            await health_checker.stop_monitoring()
        if ollama_client and hasattr(ollama_client, 'cleanup'):
            await ollama_client.cleanup()
        logging.info("✅ Services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

def log_startup_summary():
    """Log startup information"""
    logging.info("=" * 60)
    logging.info("🚀 LLM Proxy - Startup Summary")
    logging.info("=" * 60)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Host: {settings.HOST}:{settings.PORT}")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"   • Security: {'✅ Available' if SECURITY_AVAILABLE else '⏸️  Basic'}")
    logging.info(f"🎯 Services:")
    logging.info(f"   • Memory Manager: {'✅' if memory_manager else '❌'}")
    logging.info(f"   • Ollama Client: {'✅' if ollama_client else '❌'}")
    logging.info(f"   • LLM Router: {'✅' if llm_router else '❌'}")
    logging.info(f"   • Metrics: {'✅' if metrics else '❌'}")
    logging.info(f"   • Health Checker: {'✅' if health_checker else '❌'}")
    logging.info(f"   • Auth Service: {'✅' if auth_service else '❌'}")
    logging.info("=" * 60)

# Create FastAPI app
app = FastAPI(
    title="LLM Proxy",
    description="Production-ready LLM routing proxy",
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup security middleware if available
if SECURITY_AVAILABLE and hasattr(security_settings, 'SECURITY_HEADERS_ENABLED'):
    try:
        setup_security_middleware(app, security_settings)
    except Exception as e:
        logging.warning(f"Failed to setup security middleware: {e}")

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
    if hasattr(request.state, 'user'):
        return request.state.user
    return None

# Basic error handling
class LLMProxyError(Exception):
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

# Main API Routes
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Chat completions endpoint with error handling"""
    
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        # Track request if metrics available
        if metrics:
            user_id = current_user.get('user_id') if current_user else 'anonymous'
            metrics.track_request("chat_completions", user_id)
        
        # Route request
        selected_model = await llm_router.route_request(request)
        
        # Process request
        response = await llm_router.process_chat_completion(request, selected_model)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Completions endpoint"""
    
    try:
        # Convert to chat format
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
                    "text": response.choices[0].get("message", {}).get("content", ""),
                    "index": 0,
                    "finish_reason": response.choices[0].get("finish_reason", "stop")
                }],
                "usage": response.usage
            }
        
        return response
        
    except Exception as e:
        logging.error(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    
    try:
        services_status = []
        
        # Check Ollama
        if ollama_client:
            try:
                if hasattr(ollama_client, 'health_check'):
                    ollama_healthy = await ollama_client.health_check()
                else:
                    ollama_healthy = True  # Assume healthy if no health check method
                
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
        else:
            services_status.append({
                "name": "ollama",
                "status": "unavailable",
                "last_check": datetime.now().isoformat()
            })
        
        overall_healthy = all(s["status"] == "healthy" for s in services_status)
        
        health_response = HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            healthy=overall_healthy,
            timestamp=datetime.now().isoformat(),
            services=services_status
        )
        
        if not overall_healthy:
            raise HTTPException(status_code=503, detail="Service unhealthy")
        
        return health_response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/models")
async def list_available_models():
    """List available models"""
    
    try:
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        if hasattr(llm_router, 'get_available_models'):
            models = await llm_router.get_available_models()
        else:
            models = [{"id": "mistral:7b-instruct-q4_0", "object": "model"}]
        
        return {"object": "list", "data": models}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get basic metrics"""
    
    try:
        if metrics and hasattr(metrics, 'get_all_metrics'):
            return await metrics.get_all_metrics()
        else:
            return {
                "status": "basic_metrics",
                "timestamp": datetime.now().isoformat(),
                "message": "Enhanced metrics not available"
            }
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
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
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
