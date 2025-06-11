# main.py - Enhanced LLM Proxy with Semantic Routing (GitHub Version)
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

# Enhanced configuration with fallback
try:
    from config_enhanced import get_settings
    settings = get_settings()
    logging.info("✅ Enhanced configuration loaded")
except ImportError as e:
    logging.warning(f"Enhanced config not available: {e}")
    from pydantic_settings import BaseSettings
    
    class BasicSettings(BaseSettings):
        model_config = {"extra": "ignore"}
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
        ENABLE_SEMANTIC_CLASSIFICATION: bool = True  # ENABLED by default
    
    settings = BasicSettings()
    logging.info("✅ Basic configuration with semantic routing enabled")

# Configure logging
try:
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
except AttributeError:
    log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import models with fallback
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
        selected_model: Optional[str] = None
        routing_reason: Optional[str] = None
        processing_time: Optional[float] = None
        cache_hit: Optional[bool] = False
    
    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        timestamp: str
        version: str = "2.2.0"
        services: List[Dict[str, Any]] = []

# Global service instances
ollama_client = None
llm_router = None
enhanced_capabilities = {}

# Enhanced service initialization with proper semantic router
async def initialize_enhanced_services():
    """Initialize services with enhanced semantic routing"""
    global ollama_client, llm_router, enhanced_capabilities
    
    logging.info("🚀 Initializing Enhanced Services with Semantic Routing...")
    
    try:
        # Initialize Ollama client (enhanced or basic)
        try:
            from services.enhanced_imports import setup_enhanced_imports
            enhanced_imports = setup_enhanced_imports()
            enhanced_capabilities = enhanced_imports['capabilities']
            
            ollama_client_class = enhanced_imports.get('EnhancedOllamaClient')
            if ollama_client_class:
                ollama_client = ollama_client_class(settings.OLLAMA_BASE_URL)
                await ollama_client.initialize()
                logging.info("✅ Enhanced Ollama client initialized")
            else:
                raise ImportError("Enhanced Ollama client not available")
                
        except Exception as e:
            logging.warning(f"Enhanced imports failed: {e}")
            from services.ollama_client import OllamaClient
            ollama_client = OllamaClient(settings.OLLAMA_BASE_URL)
            await ollama_client.initialize()
            enhanced_capabilities = {"semantic_classification": True}  # Will try semantic router anyway
            logging.info("✅ Basic Ollama client initialized")
        
        # Initialize SEMANTIC ROUTER specifically
        if ollama_client:
            try:
                # Try enhanced router with semantic classification
                if getattr(settings, 'ENABLE_SEMANTIC_CLASSIFICATION', True):
                    try:
                        from services.enhanced_router import EnhancedLLMRouter
                        llm_router = EnhancedLLMRouter(ollama_client)
                        await llm_router.initialize()
                        logging.info("✅ ENHANCED Semantic LLM Router initialized")
                        enhanced_capabilities['semantic_classification'] = True
                    except ImportError as e:
                        logging.warning(f"Enhanced router import failed: {e}")
                        # Create a basic enhanced router inline
                        llm_router = await create_basic_enhanced_router(ollama_client)
                        enhanced_capabilities['semantic_classification'] = True
                else:
                    raise ImportError("Semantic classification disabled")
                    
            except ImportError as e:
                logging.warning(f"Semantic router failed: {e}")
                from services.router import LLMRouter
                llm_router = LLMRouter(ollama_client)
                await llm_router.initialize()
                logging.warning("⚠️ Using Basic LLM Router (NO SEMANTIC ROUTING)")
                enhanced_capabilities['semantic_classification'] = False
            
    except Exception as e:
        logging.error(f"❌ Failed to initialize enhanced services: {e}")
        logging.error(traceback.format_exc())
        raise

async def create_basic_enhanced_router(ollama_client):
    """Create a basic enhanced router with semantic patterns if advanced router is not available"""
    from services.router import LLMRouter
    
    class BasicEnhancedRouter(LLMRouter):
        def __init__(self, ollama_client):
            super().__init__(ollama_client)
            
            # Enhanced intent patterns for semantic routing
            self.intent_patterns = {
                'math': r'\b(?:calculate|compute|solve|equation|math|arithmetic|area|volume|percentage|formula|\d+\s*[\+\-\*\/\%\^]\s*\d+)\b',
                'factual': r'\b(?:what is|who is|when did|where is|define|explain|fact|capital|population)\b',
                'creative': r'\b(?:write|create|compose|story|poem|creative|imagine|generate|blog|article)\b',
                'coding': r'\b(?:code|function|algorithm|debug|program|script|python|javascript|java|c\+\+|bug|error|programming)\b',
                'resume': r'\b(?:resume|cv|experience|skills|qualifications|work history|analyze.*resume|technical skills)\b',
                'interview': r'\b(?:interview|job|career|hiring|prepare.*interview|interview.*questions|google|microsoft|amazon)\b',
                'analysis': r'\b(?:analyze|review|evaluate|assess|compare|examine|advantages|disadvantages|pros.*cons|microservices|architecture)\b'
            }
            
            # Intent to model mapping
            self.intent_model_map = {
                'coding': 'deepseek-v2:7b-q4_0',
                'resume': 'deepseek-v2:7b-q4_0',
                'analysis': 'deepseek-v2:7b-q4_0',
                'interview': 'llama3:8b-instruct-q4_0',
                'creative': 'llama3:8b-instruct-q4_0',
                'math': 'mistral:7b-instruct-q4_0',
                'factual': 'mistral:7b-instruct-q4_0',
                'general': 'mistral:7b-instruct-q4_0'
            }
            
            logging.info("Basic Enhanced Router created with semantic patterns")
        
        def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
            """Enhanced intent classification"""
            if explicit_intent:
                return explicit_intent
            
            import re
            text_lower = text.lower()
            
            # Enhanced pattern matching with scoring
            intent_scores = {}
            for intent, pattern in self.intent_patterns.items():
                matches = re.findall(pattern, text_lower)
                if matches:
                    score = len(matches)
                    
                    # Boost scores for specific keywords
                    if intent == 'coding' and any(word in text_lower for word in ['debug', 'error', 'bug']):
                        score += 2
                    elif intent == 'resume' and any(word in text_lower for word in ['technical skills', 'analyze']):
                        score += 2
                    elif intent == 'interview' and 'prepare' in text_lower:
                        score += 2
                    elif intent == 'analysis' and any(word in text_lower for word in ['compare', 'advantages']):
                        score += 2
                    elif intent == 'math' and any(word in text_lower for word in ['calculate', 'area']):
                        score += 2
                    
                    intent_scores[intent] = score
            
            if intent_scores:
                best_intent = max(intent_scores, key=intent_scores.get)
                logging.info(f"Classified '{text[:50]}...' as '{best_intent}'")
                return best_intent
            
            # Fallback
            return 'general'
        
        async def route_request(self, request) -> str:
            """Enhanced routing with semantic classification"""
            if request.model in self.available_models:
                return request.model
            
            # Extract text for classification
            text_content = ' '.join(
                msg.get('content', '') for msg in 
                [{"role": msg.role, "content": msg.content} for msg in request.messages]
                if msg.get('role') == 'user'
            )
            
            # Classify intent
            intent = self.classify_intent(text_content, getattr(request, 'intent', None))
            
            # Select model by intent
            preferred_model = self.intent_model_map.get(intent, 'mistral:7b-instruct-q4_0')
            
            # Check availability
            if preferred_model in self.available_models:
                selected_model = preferred_model
            else:
                selected_model = list(self.available_models.keys())[0] if self.available_models else 'mistral:7b-instruct-q4_0'
            
            logging.info(f"SEMANTIC ROUTING: '{text_content[:50]}...' -> intent='{intent}' -> model='{selected_model}'")
            return selected_model
        
        def get_classification_stats(self):
            return {
                'enhanced_patterns': True,
                'available_intents': list(self.intent_patterns.keys()),
                'semantic_enabled': True
            }
    
    router = BasicEnhancedRouter(ollama_client)
    await router.initialize()
    logging.info("✅ Basic Enhanced Router with semantic patterns initialized")
    return router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with enhanced services"""
    
    # Startup
    logging.info("🚀 Starting Enhanced LLM Proxy Service...")
    
    try:
        await initialize_enhanced_services()
        logging.info("✅ Enhanced services initialized")
        log_startup_summary()
    except Exception as e:
        logging.error(f"❌ Failed to start services: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Enhanced LLM Proxy Service...")
    
    try:
        if ollama_client and hasattr(ollama_client, 'cleanup'):
            await ollama_client.cleanup()
        logging.info("✅ Services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

def log_startup_summary():
    """Log enhanced startup information"""
    logging.info("=" * 60)
    logging.info("🚀 Enhanced LLM Proxy - Startup Summary")
    logging.info("=" * 60)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Host: {settings.HOST}:{settings.PORT}")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"🎯 Services:")
    logging.info(f"   • Ollama Client: {'✅' if ollama_client else '❌'}")
    logging.info(f"   • Router Type: {type(llm_router).__name__ if llm_router else '❌'}")
    logging.info(f"   • Semantic Classification: {'✅' if enhanced_capabilities.get('semantic_classification') else '❌'}")
    logging.info(f"🔧 Enhanced Features:")
    for feature, enabled in enhanced_capabilities.items():
        status = '✅' if enabled else '❌'
        logging.info(f"   • {feature}: {status}")
    logging.info("=" * 60)

# Create FastAPI app
app = FastAPI(
    title="Enhanced LLM Proxy with Semantic Routing",
    description="Production-ready LLM routing proxy with semantic classification",
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

# Dependencies
async def get_llm_router():
    """Dependency to get the LLM router"""
    if not llm_router:
        raise HTTPException(status_code=503, detail="LLM router not available")
    return llm_router

async def get_ollama_client():
    """Dependency to get the Ollama client"""
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama client not available")
    return ollama_client

# Enhanced endpoints
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    router: Any = Depends(get_llm_router),
    client: Any = Depends(get_ollama_client)
):
    """Enhanced chat completions endpoint with semantic routing"""
    
    start_time = datetime.now()
    
    try:
        logging.info(f"📨 Processing request with router: {type(router).__name__}")
        
        # Use enhanced routing
        if hasattr(router, 'route_request'):
            selected_model = await router.route_request(request)
            routing_reason = "enhanced_semantic" if hasattr(router, 'classify_intent') else "rule_based"
        else:
            selected_model = getattr(settings, 'DEFAULT_MODEL', 'mistral:7b-instruct-q4_0')
            routing_reason = "fallback"
        
        logging.info(f"🎯 Routed to model: {selected_model} (reason: {routing_reason})")
        
        # Process request
        if hasattr(router, 'process_chat_completion'):
            response = await router.process_chat_completion(request, selected_model)
        else:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            response = await client.generate_completion(
                model=selected_model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 150,
                top_p=request.top_p
            )
        
        # Add enhanced metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if isinstance(response, dict):
            response.update({
                'selected_model': selected_model,
                'routing_reason': routing_reason,
                'processing_time': processing_time,
                'cache_hit': False
            })
        
        logging.info(f"✅ Request completed in {processing_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logging.error(f"❌ Error in chat completions: {str(e)} (after {processing_time:.3f}s)")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    router: Any = Depends(get_llm_router),
    client: Any = Depends(get_ollama_client)
):
    """Completions endpoint with semantic routing"""
    
    try:
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[Message(role="user", content=request.prompt)],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream
        )
        
        response = await chat_completions(chat_request, http_request, router, client)
        
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
                "usage": response.usage,
                "selected_model": getattr(response, 'selected_model', None),
                "routing_reason": getattr(response, 'routing_reason', None)
            }
        
        return response
        
    except Exception as e:
        logging.error(f"❌ Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with router information"""
    
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
        
        # Check Router
        if llm_router:
            router_type = type(llm_router).__name__
            has_semantic = hasattr(llm_router, 'classify_intent')
            services_status.append({
                "name": "router",
                "status": "healthy",
                "last_check": datetime.now().isoformat(),
                "type": router_type,
                "semantic_enabled": has_semantic
            })
        
        overall_healthy = all(s["status"] == "healthy" for s in services_status)
        
        health_response = HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            healthy=overall_healthy,
            timestamp=datetime.now().isoformat(),
            version="2.2.0",
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
async def list_available_models(router: Any = Depends(get_llm_router)):
    """List available models with routing information"""
    
    try:
        models = await router.get_available_models()
        return {"object": "list", "data": models}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/router/status")
async def get_router_status(router: Any = Depends(get_llm_router)):
    """Get detailed router status for debugging"""
    
    try:
        status = {
            "router_type": type(router).__name__,
            "router_module": type(router).__module__,
            "semantic_enabled": hasattr(router, 'classify_intent'),
            "enhanced_capabilities": enhanced_capabilities,
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(router, 'get_classification_stats'):
            status['classification_stats'] = router.get_classification_stats()
        
        if hasattr(router, 'intent_model_map'):
            status['intent_model_mapping'] = router.intent_model_map
        
        return status
        
    except Exception as e:
        logging.error(f"Error getting router status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Enhanced metrics with router information"""
    
    try:
        metrics = {
            "status": "enhanced_metrics_available",
            "timestamp": datetime.now().isoformat(),
            "router": {
                "type": type(llm_router).__name__ if llm_router else None,
                "semantic_enabled": enhanced_capabilities.get('semantic_classification', False)
            },
            "features": enhanced_capabilities
        }
        
        if llm_router and hasattr(llm_router, 'get_classification_stats'):
            metrics['classification'] = llm_router.get_classification_stats()
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return {
            "status": "basic_metrics",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
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
