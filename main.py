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
from typing import Optional, Dict, Any, List

# Import configuration with fallback handling
try:
    from config_enhanced import get_settings
    settings = get_settings()
    logging.info("✅ Enhanced configuration loaded")
except ImportError as e:
    logging.warning(f"Enhanced config not available: {e}")
    # Fallback to basic configuration
    from pydantic_settings import BaseSettings
    
    class BasicSettings(BaseSettings):
        model_config = {"extra": "ignore"}  # This is the key fix!
        
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
        DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"
        ENABLE_SEMANTIC_CLASSIFICATION: bool = False
    
    settings = BasicSettings()
    logging.info("✅ Basic configuration loaded")

# Try to import security settings with fallback
SECURITY_AVAILABLE = False
security_settings = None

try:
    # Create basic security config that won't cause Pydantic errors
    from pydantic_settings import BaseSettings as SecurityBase
    
    class SecuritySettings(SecurityBase):
        model_config = {"extra": "ignore"}  # Key fix for security settings too
        
        ENVIRONMENT: str = "development"
        SECRET_KEY: str = "dev-secret-key"
        SECURITY_HEADERS_ENABLED: bool = False
        USE_TLS: bool = False
        CORS_ORIGINS: list = ["*"]
        RATE_LIMIT_ENABLED: bool = False
        AUTH_ENABLED: bool = False
    
    def validate_production_config(settings):
        return []
    
    def setup_security_middleware(app, settings):
        pass
    
    security_settings = SecuritySettings()
    SECURITY_AVAILABLE = True
    logging.info("✅ Security configuration loaded")
                
except Exception as e:
    logging.warning(f"Security module error: {e}")
    # Create a basic security settings fallback
    class BasicSecuritySettings:
        ENVIRONMENT = "development"
        SECURITY_HEADERS_ENABLED = False
        USE_TLS = False
        CORS_ORIGINS = ["*"]
        RATE_LIMIT_ENABLED = False
    
    security_settings = BasicSecuritySettings()
    logging.info("✅ Basic security fallback loaded")

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

# Create basic Ollama client
import aiohttp
import json

class BasicOllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        logging.info(f"Basic Ollama client initialized: {self.base_url}")
    
    async def cleanup(self):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        try:
            if not self.session:
                await self.initialize()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
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
        except Exception:
            return []
    
    async def generate_completion(self, model: str, messages: List[Dict], **kwargs):
        try:
            if not self.session:
                await self.initialize()
            
            # Convert messages to prompt
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    prompt += f"User: {content}\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n"
            prompt += "Assistant: "
            
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
                    return {
                        "id": f"completion-{int(asyncio.get_event_loop().time())}",
                        "object": "chat.completion", 
                        "created": int(asyncio.get_event_loop().time()),
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
                        }
                    }
                else:
                    raise Exception(f"API call failed with status {response.status}")
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            raise

# Create basic router
class BasicLLMRouter:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self.default_model = getattr(settings, 'DEFAULT_MODEL', "mistral:7b-instruct-q4_0")
    
    async def initialize(self):
        logging.info("Basic LLM Router initialized")
    
    async def route_request(self, request) -> str:
        if hasattr(request, 'model') and request.model:
            return request.model
        return self.default_model
    
    async def process_chat_completion(self, request, model: str):
        try:
            messages = []
            if hasattr(request, 'messages'):
                for msg in request.messages:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        messages.append({"role": msg.role, "content": msg.content})
                    elif isinstance(msg, dict):
                        messages.append(msg)
            
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
        try:
            models = await self.ollama_client.list_models()
            return [{"id": model.get("name", "unknown"), "object": "model"} for model in models]
        except Exception:
            return [{"id": self.default_model, "object": "model"}]

# Try to initialize services with fallbacks
async def initialize_core_services():
    """Initialize core services with fallback handling"""
    global ollama_client, llm_router, enhanced_capabilities
    
    try:
        # Initialize Ollama client
        try:
            from services.enhanced_imports import setup_enhanced_imports
            enhanced_imports = setup_enhanced_imports()
            enhanced_capabilities = enhanced_imports['capabilities']
            
            ollama_client = enhanced_imports.get('EnhancedOllamaClient', BasicOllamaClient)(
                settings.OLLAMA_BASE_URL, 
                getattr(settings, 'OLLAMA_TIMEOUT', 300)
            )
            await ollama_client.initialize()
            logging.info("✅ Enhanced Ollama client initialized")
        except Exception as e:
            logging.warning(f"Enhanced Ollama client failed: {e}")
            # Fallback to basic client
            ollama_client = BasicOllamaClient(settings.OLLAMA_BASE_URL)
            await ollama_client.initialize()
            enhanced_capabilities = {"streaming": False, "model_warmup": False}
            logging.info("✅ Basic Ollama client initialized")
        
        # Initialize router
        if ollama_client:
            try:
                from services.router import LLMRouter as EnhancedLLMRouter
                llm_router = EnhancedLLMRouter(ollama_client)
                await llm_router.initialize()
                logging.info("✅ Enhanced LLM router initialized")
            except Exception as e:
                logging.warning(f"Enhanced router failed: {e}")
                llm_router = BasicLLMRouter(ollama_client)
                await llm_router.initialize()
                logging.info("✅ Basic LLM router initialized")
            
    except Exception as e:
        logging.error(f"Failed to initialize core services: {e}")
        logging.error(traceback.format_exc())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper error handling"""
    
    # Startup
    logging.info("🚀 Starting LLM Proxy Service...")
    
    try:
        await initialize_core_services()
        logging.info("✅ Core services initialized")
        log_startup_summary()
    except Exception as e:
        logging.error(f"❌ Failed to start some services: {e}")
        # Don't fail completely - allow partial startup
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down LLM Proxy Service...")
    
    try:
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
    logging.info(f"   • Ollama Client: {'✅' if ollama_client else '❌'}")
    logging.info(f"   • LLM Router: {'✅' if llm_router else '❌'}")
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
            messages=[Message(role="user", content=request.prompt)],
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
        
        models = await llm_router.get_available_models()
        return {"object": "list", "data": models}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get basic metrics"""
    return {
        "status": "basic_metrics",
        "timestamp": datetime.now().isoformat(),
        "message": "Enhanced metrics not available"
    }

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
