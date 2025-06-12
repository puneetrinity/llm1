# main.py - Enhanced LLM Proxy with 4-Model Optimization
import asyncio
import logging
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError

# Import configuration with enhanced fallback
try:
    from config_enhanced import get_settings
    settings = get_settings()
    logging.info("✅ Enhanced configuration loaded")
except ImportError:
    # Fallback to basic configuration
    from pydantic_settings import BaseSettings
    
    class BasicSettings(BaseSettings):
        model_config = {"extra": "ignore"}
        
        DEBUG: bool = False
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        CORS_ORIGINS: list = ["*"]
        CORS_ALLOW_CREDENTIALS: bool = True
        LOG_LEVEL: str = "INFO"
        ENABLE_AUTH: bool = False
        MAX_MEMORY_MB: int = 8192
        
    settings = BasicSettings()
    logging.info("✅ Basic configuration loaded")

# Enhanced imports with graceful fallbacks
enhanced_features = {
    'connection_pooling': False,
    'circuit_breakers': False,
    'smart_caching': False,
    'semantic_classification': False,
    'streaming': False,
    'model_warmup': False,
    'performance_monitoring': False
}

# Try to import enhanced features
try:
    from services.enhanced_imports import setup_enhanced_imports
    enhanced_imports = setup_enhanced_imports()
    enhanced_features.update(enhanced_imports.get('capabilities', {}))
    logging.info("✅ Enhanced features loaded")
except Exception as e:
    logging.warning(f"⚠️ Enhanced features not available: {e}")

# Import models with fallback
try:
    from models.requests import ChatCompletionRequest, CompletionRequest
    from models.responses import ChatCompletionResponse, HealthResponse
    logging.info("✅ Custom models loaded")
except ImportError:
    from pydantic import BaseModel
    from typing import List as TypingList
    
    class Message(BaseModel):
        role: str
        content: str
    
    class ChatCompletionRequest(BaseModel):
        model: str
        messages: TypingList[Message]
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
        choices: TypingList[Dict[str, Any]]
        usage: Dict[str, Any]
    
    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        timestamp: str
        version: str = "2.2.0"
        services: TypingList[Dict[str, Any]] = []

# Configure logging
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Global service instances
ollama_client = None
llm_router = None
metrics_collector = None
health_checker = None
circuit_breaker_manager = None
smart_cache = None
websocket_dashboard = None

class EnhancedSemanticRouter:
    """Optimized 4-model semantic router with intelligent routing"""
    
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        
        # Optimized 4-model configuration
        self.model_config = {
            'qwen2.5-coder:7b-instruct': {
                'priority': 1,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'memory_mb': 4500,
                'good_for': ['coding', 'debugging', 'resume', 'analysis'],
                'specialization': 'code_analysis'
            },
            'qwen2.5-coder:7b-instruct-q6_k': {
                'priority': 1,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'memory_mb': 3500,
                'good_for': ['resume', 'analysis', 'coding'],
                'specialization': 'analysis'
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00015,
                'max_context': 8192,
                'memory_mb': 5000,
                'good_for': ['creative', 'interview', 'storytelling', 'conversation'],
                'specialization': 'creative'
            },
            'mistral:7b-instruct-q4_0': {
                'priority': 3,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'memory_mb': 4500,
                'good_for': ['math', 'factual', 'general'],
                'specialization': 'precision'
            }
        }
        
        # Enhanced intent patterns for 4-model optimization
        self.intent_patterns = {
            'coding': r'\b(?:code|function|algorithm|debug|program|script|python|javascript|sql|api|class|variable|loop|if|else|def|import|error|bug|syntax)\b',
            'resume': r'\b(?:resume|cv|experience|skills|qualifications|work history|employment|career|job|position|responsibilities|achievements)\b',
            'analysis': r'\b(?:analyze|review|evaluate|assess|compare|examine|study|research|investigate|report|findings|data|statistics)\b',
            'creative': r'\b(?:write|create|compose|story|poem|creative|imagine|generate|novel|character|plot|narrative|fiction)\b',
            'interview': r'\b(?:interview|job|career|hiring|prepare|questions|behavioral|technical|company|role|position)\b',
            'math': r'\b(?:calculate|compute|solve|equation|math|arithmetic|formula|percentage|probability|statistics|\d+\s*[\+\-\*\/\%]\s*\d+)\b',
            'factual': r'\b(?:what is|who is|when did|where is|define|explain|fact|information|knowledge|history|definition)\b'
        }
        
        self.loaded_models = set()
        self.model_performance = {}
        
    async def initialize(self):
        """Initialize the enhanced router"""
        available_models = await self.ollama_client.list_models() if self.ollama_client else []
        available_model_names = {model.get('name', '') for model in available_models}
        
        # Filter to only available models
        self.available_models = {
            name: config for name, config in self.model_config.items()
            if name in available_model_names
        }
        
        if not self.available_models:
            # Fallback to any available model
            if available_models:
                fallback_model = available_models[0].get('name', 'mistral:7b-instruct-q4_0')
                self.available_models[fallback_model] = {
                    'priority': 1,
                    'cost_per_token': 0.0001,
                    'max_context': 4096,
                    'memory_mb': 4000,
                    'good_for': ['general'],
                    'specialization': 'general'
                }
                logging.warning(f"Using fallback model: {fallback_model}")
        
        logging.info(f"Enhanced router initialized with models: {list(self.available_models.keys())}")
    
    async def route_request(self, request: ChatCompletionRequest) -> str:
        """Intelligent routing with 4-model optimization"""
        
        # Use explicit model if available
        if hasattr(request, 'model') and request.model in self.available_models:
            return request.model
        
        # Extract content for classification
        text_content = self._extract_text_content(request.messages)
        
        # Classify intent with enhanced patterns
        intent = self.classify_intent(text_content, getattr(request, 'intent', None))
        
        # Route based on optimization
        selected_model = self._select_optimal_model(intent, text_content, request)
        
        logging.info(f"Routed request (intent: {intent}) to: {selected_model}")
        return selected_model
    
    def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
        """Enhanced intent classification for 4-model routing"""
        
        if explicit_intent:
            return explicit_intent
        
        text_lower = text.lower()
        
        # Enhanced pattern matching with priority
        intent_scores = {}
        
        for intent, pattern in self.intent_patterns.items():
            import re
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                intent_scores[intent] = matches
        
        if intent_scores:
            # Return intent with highest score
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback heuristics
        word_count = len(text.split())
        
        if word_count < 10:
            return 'factual'
        elif word_count > 100:
            if any(word in text_lower for word in ['analyze', 'review', 'evaluate']):
                return 'analysis'
            else:
                return 'creative'
        elif any(word in text_lower for word in ['resume', 'cv', 'experience']):
            return 'resume'
        elif any(word in text_lower for word in ['interview', 'job', 'career']):
            return 'interview'
        elif any(word in text_lower for word in ['code', 'function', 'algorithm']):
            return 'coding'
        elif any(word in text_lower for word in ['calculate', 'solve', 'math']):
            return 'math'
        
        return 'factual'
    
    def _select_optimal_model(self, intent: str, text: str, request: ChatCompletionRequest) -> str:
        """Select optimal model based on intent and specialization"""
        
        # Model specialization mapping
        specialization_map = {
            'coding': ['qwen2.5-coder:7b-instruct', 'qwen2.5-coder:7b-instruct-q6_k'],
            'resume': ['qwen2.5-coder:7b-instruct-q6_k', 'qwen2.5-coder:7b-instruct'],
            'analysis': ['qwen2.5-coder:7b-instruct-q6_k', 'qwen2.5-coder:7b-instruct'],
            'debugging': ['qwen2.5-coder:7b-instruct'],
            'creative': ['llama3:8b-instruct-q4_0'],
            'interview': ['llama3:8b-instruct-q4_0'],
            'storytelling': ['llama3:8b-instruct-q4_0'],
            'conversation': ['llama3:8b-instruct-q4_0'],
            'math': ['mistral:7b-instruct-q4_0'],
            'factual': ['mistral:7b-instruct-q4_0'],
            'general': ['mistral:7b-instruct-q4_0']
        }
        
        # Get preferred models for this intent
        preferred_models = specialization_map.get(intent, ['mistral:7b-instruct-q4_0'])
        
        # Find first available preferred model
        for model in preferred_models:
            if model in self.available_models:
                return model
        
        # Fallback to any available model
        if self.available_models:
            return list(self.available_models.keys())[0]
        
        return 'mistral:7b-instruct-q4_0'  # Ultimate fallback
    
    def _extract_text_content(self, messages: List) -> str:
        """Extract text content from messages"""
        content_parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                content_parts.append(msg.content)
            elif isinstance(msg, dict) and 'content' in msg:
                content_parts.append(msg['content'])
        return ' '.join(content_parts)
    
    async def process_chat_completion(self, request: ChatCompletionRequest, model: str):
        """Process chat completion with enhanced error handling"""
        
        try:
            # Convert messages to proper format
            messages = []
            for msg in request.messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    messages.append(msg)
            
            # Prepare request for Ollama
            ollama_request = {
                "model": model,
                "messages": messages,
                "stream": request.stream,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
            }
            
            if request.max_tokens:
                ollama_request["options"]["num_predict"] = request.max_tokens
            
            # Send to Ollama
            start_time = time.time()
            
            if hasattr(self.ollama_client, 'chat_completion'):
                response = await self.ollama_client.chat_completion(ollama_request)
            else:
                # Fallback for basic client
                response = await self._basic_chat_completion(ollama_request)
            
            processing_time = time.time() - start_time
            
            # Convert to OpenAI format
            return self._convert_response(response, model, processing_time)
            
        except Exception as e:
            logging.error(f"Error processing chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _basic_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic chat completion fallback"""
        import aiohttp
        import json
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_client.base_url}/api/chat",
                json=request_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
    
    def _convert_response(self, ollama_response: Dict[str, Any], model: str, processing_time: float):
        """Convert Ollama response to OpenAI format"""
        
        message = ollama_response.get('message', {})
        content = message.get('content', '')
        
        # Estimate token usage
        prompt_tokens = len(str(ollama_response.get('prompt', '')).split())
        completion_tokens = len(content.split())
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "processing_time": processing_time,
            "selected_model": model,
            "enhanced_features": enhanced_features
        }
    
    async def get_available_models(self):
        """Get available models with metadata"""
        models = []
        for model_name, config in self.available_models.items():
            models.append({
                "id": model_name,
                "object": "model",
                "owned_by": "ollama",
                "specialization": config.get('specialization', 'general'),
                "good_for": config.get('good_for', []),
                "cost_per_token": config.get('cost_per_token', 0.0001),
                "memory_mb": config.get('memory_mb', 4000)
            })
        return models

async def initialize_services():
    """Initialize all services with enhanced features"""
    global ollama_client, llm_router, metrics_collector, health_checker
    global circuit_breaker_manager, smart_cache, websocket_dashboard
    
    try:
        # Initialize Ollama client
        if enhanced_features.get('connection_pooling'):
            from services.enhanced_ollama_client import EnhancedOllamaClient
            ollama_client = EnhancedOllamaClient(settings.OLLAMA_BASE_URL)
        else:
            # Basic client fallback
            import aiohttp
            
            class BasicOllamaClient:
                def __init__(self, base_url):
                    self.base_url = base_url.rstrip('/')
                    self.session = None
                
                async def initialize(self):
                    self.session = aiohttp.ClientSession()
                
                async def health_check(self):
                    try:
                        async with self.session.get(f"{self.base_url}/api/tags") as response:
                            return response.status == 200
                    except:
                        return False
                
                async def list_models(self):
                    try:
                        async with self.session.get(f"{self.base_url}/api/tags") as response:
                            if response.status == 200:
                                data = await response.json()
                                return data.get('models', [])
                    except:
                        return []
                
                async def cleanup(self):
                    if self.session:
                        await self.session.close()
            
            ollama_client = BasicOllamaClient(settings.OLLAMA_BASE_URL)
        
        await ollama_client.initialize()
        
        # Initialize enhanced router
        llm_router = EnhancedSemanticRouter(ollama_client)
        await llm_router.initialize()
        
        # Initialize enhanced features if available
        if enhanced_features.get('performance_monitoring'):
            from utils.performance_monitor import PerformanceMonitor
            performance_monitor = PerformanceMonitor()
            await performance_monitor.start_monitoring()
        
        if enhanced_features.get('circuit_breakers'):
            from services.circuit_breaker import get_circuit_breaker_manager
            circuit_breaker_manager = get_circuit_breaker_manager()
        
        if enhanced_features.get('smart_caching'):
            from services.smart_cache import initialize_smart_cache
            smart_cache = await initialize_smart_cache()
        
        # Basic health checker
        from utils.health import HealthChecker
        health_checker = HealthChecker()
        await health_checker.start_monitoring()
        
        # Basic metrics collector
        from utils.metrics import MetricsCollector
        metrics_collector = MetricsCollector()
        
        logging.info("✅ All services initialized successfully")
        
    except Exception as e:
        logging.error(f"❌ Service initialization failed: {e}")
        logging.error(traceback.format_exc())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logging.info("🚀 Starting Enhanced LLM Proxy Service...")
    
    try:
        await initialize_services()
        log_startup_summary()
    except Exception as e:
        logging.error(f"❌ Startup failed: {e}")
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down services...")
    
    try:
        if ollama_client and hasattr(ollama_client, 'cleanup'):
            await ollama_client.cleanup()
        if health_checker:
            await health_checker.stop_monitoring()
        logging.info("✅ Services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Shutdown error: {e}")

def log_startup_summary():
    """Log comprehensive startup summary"""
    logging.info("=" * 80)
    logging.info("🚀 Enhanced LLM Proxy - Startup Summary")
    logging.info("=" * 80)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Host: {settings.HOST}:{settings.PORT}")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Debug Mode: {settings.DEBUG}")
    logging.info(f"🎯 Enhanced Features:")
    for feature, enabled in enhanced_features.items():
        status = "✅" if enabled else "⏸️"
        logging.info(f"   • {feature}: {status}")
    if llm_router and hasattr(llm_router, 'available_models'):
        logging.info(f"🤖 Available Models: {list(llm_router.available_models.keys())}")
    logging.info("=" * 80)

# Create FastAPI app
app = FastAPI(
    title="Enhanced LLM Proxy",
    description="Production-ready LLM routing proxy with 4-model optimization",
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

# Main API Endpoints
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Enhanced chat completions with 4-model routing"""
    
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        if not llm_router:
            raise HTTPException(status_code=503, detail="Router not available")
        
        # Track metrics
        if metrics_collector:
            metrics_collector.track_request("/v1/chat/completions")
        
        # Route request with enhanced logic
        selected_model = await llm_router.route_request(request)
        
        # Handle streaming
        if request.stream and enhanced_features.get('streaming'):
            return await handle_streaming_request(request, selected_model)
        
        # Process request
        start_time = time.time()
        response = await llm_router.process_chat_completion(request, selected_model)
        processing_time = time.time() - start_time
        
        # Track metrics
        if metrics_collector:
            metrics_collector.track_response_time(processing_time)
            metrics_collector.track_model_usage(
                selected_model, 
                response.get('usage', {}).get('prompt_tokens', 0),
                response.get('usage', {}).get('completion_tokens', 0),
                processing_time
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in chat completions: {str(e)}")
        if metrics_collector:
            metrics_collector.track_error("chat_completion_error")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_streaming_request(request: ChatCompletionRequest, model: str):
    """Handle streaming requests"""
    
    async def generate_stream():
        try:
            # Mock streaming implementation
            # In a real implementation, this would use the streaming service
            yield f"data: {{'id': 'stream-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'choices': [{{'delta': {{'content': 'Hello'}}}}]}}\n\n"
            yield f"data: {{'id': 'stream-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'choices': [{{'delta': {{'content': ' world!'}}}}]}}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logging.error(f"Streaming error: {e}")
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Text completions endpoint"""
    
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
        
        # Convert back to completion format
        if not request.stream and isinstance(response, dict):
            return {
                "id": response.get("id"),
                "object": "text_completion",
                "created": response.get("created"),
                "model": response.get("model"),
                "choices": [{
                    "text": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": response.get("usage", {})
            }
        
        return response
        
    except Exception as e:
        logging.error(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    
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
        
        # Check enhanced features
        if circuit_breaker_manager:
            cb_health = circuit_breaker_manager.get_health_summary()
            services_status.append({
                "name": "circuit_breakers",
                "status": cb_health['overall_health'],
                "last_check": datetime.now().isoformat(),
                "details": cb_health
            })
        
        if smart_cache:
            cache_stats = smart_cache.get_stats()
            services_status.append({
                "name": "smart_cache",
                "status": "healthy",
                "last_check": datetime.now().isoformat(),
                "details": {"hit_rate": cache_stats.get("hit_rate", 0)}
            })
        
        overall_healthy = all(s["status"] == "healthy" for s in services_status if s["status"] != "unknown")
        
        health_response = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "services": services_status,
            "enhanced_features": enhanced_features
        }
        
        if not overall_healthy:
            raise HTTPException(status_code=503, detail=health_response)
        
        return health_response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/models")
async def list_available_models():
    """List available models with optimization info"""
    
    try:
        if not llm_router:
            return {"object": "list", "data": []}
        
        models = await llm_router.get_available_models()
        return {"object": "list", "data": models}
        
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive metrics"""
    
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "enhanced_features": enhanced_features,
            "version": "2.2.0"
        }
        
        # Add basic metrics
        if metrics_collector:
            basic_metrics = await metrics_collector.get_all_metrics()
            metrics.update(basic_metrics)
        
        # Add circuit breaker metrics
        if circuit_breaker_manager:
            metrics["circuit_breakers"] = circuit_breaker_manager.get_all_status()
        
        # Add cache metrics
        if smart_cache:
            metrics["cache"] = smart_cache.get_stats()
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "enhanced_features": enhanced_features
        }

# Admin endpoints
@app.get("/admin/status")
async def admin_status():
    """Comprehensive admin status"""
    
    return {
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0",
        "enhanced_features": enhanced_features,
        "services": {
            "ollama_client": ollama_client is not None,
            "llm_router": llm_router is not None,
            "circuit_breakers": circuit_breaker_manager is not None,
            "smart_cache": smart_cache is not None,
            "metrics_collector": metrics_collector is not None
        },
        "available_models": list(llm_router.available_models.keys()) if llm_router else []
    }

@app.post("/admin/warmup/{model}")
async def warmup_model(model: str):
    """Warm up a specific model"""
    
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        if hasattr(ollama_client, 'warm_up_model'):
            success = await ollama_client.warm_up_model(model)
            return {"model": model, "warmed_up": success}
        else:
            return {"model": model, "warmed_up": False, "message": "Warmup not supported"}
        
    except Exception as e:
        logging.error(f"Error warming up model {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time dashboard
@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """Real-time dashboard WebSocket"""
    
    await websocket.accept()
    
    try:
        while True:
            # Send dashboard data every 5 seconds
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": await get_metrics() if metrics_collector else {},
                "health": await health_check(),
                "enhanced_features": enhanced_features
            }
            
            await websocket.send_text(json.dumps(dashboard_data))
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logging.info("Dashboard WebSocket disconnected")
    except Exception as e:
        logging.error(f"Dashboard WebSocket error: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "enhanced_features": enhanced_features
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": exc.errors(),
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
            "path": str(request.url.path),
            "enhanced_features": enhanced_features
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
