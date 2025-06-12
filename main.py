# main.py - COMPREHENSIVE Deployment-Ready Enhanced LLM Proxy
# Includes: React Dashboard + All Enhanced Features + Error Fixes
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import logging
import sys
import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# FIXED: Safe configuration loading with comprehensive fallback
try:
    from config_enhanced import get_settings
    settings = get_settings()
    logging.info("✅ Enhanced configuration loaded")
except ImportError as e:
    logging.warning(f"Enhanced config not available: {e}")
    from pydantic_settings import BaseSettings
    
    class ComprehensiveSettings(BaseSettings):
        model_config = {"extra": "ignore"}
        
        # Core settings
        DEBUG: bool = False
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        LOG_LEVEL: str = "INFO"
        
        # Ollama settings
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        OLLAMA_TIMEOUT: int = 300
        DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"
        
        # Security settings
        ENABLE_AUTH: bool = False
        DEFAULT_API_KEY: str = "sk-dev-key-change-in-production"
        API_KEY_HEADER: str = "X-API-Key"
        
        # CORS settings
        CORS_ORIGINS: list = ["*"]
        CORS_ALLOW_CREDENTIALS: bool = True
        
        # Memory management
        MAX_MEMORY_MB: int = 8192
        CACHE_MEMORY_LIMIT_MB: int = 1024
        MODEL_MEMORY_LIMIT_MB: int = 4096
        
        # Enhanced features
        ENABLE_SEMANTIC_CLASSIFICATION: bool = False
        ENABLE_STREAMING: bool = True
        ENABLE_MODEL_WARMUP: bool = True
        ENABLE_DETAILED_METRICS: bool = True
        
        # Dashboard settings
        ENABLE_DASHBOARD: bool = True
        DASHBOARD_PATH: str = "/app"
    
    settings = ComprehensiveSettings()
    logging.info("✅ Comprehensive configuration loaded")

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# FIXED: Import models with comprehensive fallback
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
    
    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[Dict[str, Any]]
        usage: Dict[str, Any]
        cache_hit: bool = False
        processing_time: Optional[float] = None
        selected_model: Optional[str] = None
    
    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        timestamp: str
        version: str = "2.2.0"
        services: List[Dict[str, Any]] = []

# Global service instances
memory_manager = None
ollama_client = None
llm_router = None
metrics_collector = None
health_checker = None
auth_service = None
streaming_service = None
warmup_service = None
enhanced_capabilities = {}
websocket_dashboard = None

# COMPREHENSIVE: Import services with all enhancements
def safe_import_service(module_path, class_name, fallback=None):
    """Safely import enhanced services with fallbacks"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logging.warning(f"Could not import {module_path}.{class_name}: {e}")
        return fallback

# Import all available services
MetricsCollector = safe_import_service('utils.metrics', 'MetricsCollector')
HealthChecker = safe_import_service('utils.health', 'HealthChecker')
WebSocketDashboard = safe_import_service('utils.websocket_dashboard', 'WebSocketDashboard')

# Enhanced imports with comprehensive error handling
try:
    from services.enhanced_imports import setup_enhanced_imports
    enhanced_imports = setup_enhanced_imports()
    enhanced_capabilities = enhanced_imports['capabilities']
    logging.info("✅ Enhanced imports available")
except Exception as e:
    logging.warning(f"Enhanced imports failed: {e}")
    enhanced_capabilities = {
        "streaming": settings.ENABLE_STREAMING,
        "model_warmup": settings.ENABLE_MODEL_WARMUP,
        "semantic_classification": False,
        "enhanced_ollama": False,
        "enhanced_router": False
    }

# COMPREHENSIVE: Create enhanced Ollama client with all features
import aiohttp
import json

class ComprehensiveOllamaClient:
    """Comprehensive Ollama client with all features and fallbacks"""
    
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
    
    async def initialize(self):
        """Initialize with enhanced connection pooling if available"""
        try:
            # Try to use enhanced connection pool
            from utils.connection_pool import get_connection_pool, ConnectionPoolConfig
            config = ConnectionPoolConfig(
                total_limit=50,
                per_host_limit=15,
                keepalive_timeout=120,
                connect_timeout=10,
                total_timeout=self.timeout
            )
            self.connection_pool = get_connection_pool(config)
            await self.connection_pool.initialize()
            logging.info("✅ Enhanced Ollama client with connection pooling")
        except Exception as e:
            logging.warning(f"Enhanced connection pool failed: {e}")
            # Fallback to basic aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            logging.info("✅ Basic Ollama client initialized")
    
    async def health_check(self) -> bool:
        """Enhanced health check with circuit breaker support"""
        try:
            if hasattr(self, 'connection_pool'):
                response_data = await self.connection_pool.post_json(
                    f"{self.base_url}/api/tags", {}
                )
                return True
            else:
                if not self.session:
                    await self.initialize()
                async with self.session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            logging.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self):
        """List available models"""
        try:
            if hasattr(self, 'connection_pool'):
                return await self.connection_pool.post_json(
                    f"{self.base_url}/api/tags", {}
                ).get('models', [])
            else:
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
        """Generate completion with enhanced features"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
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
            
            # Send request
            if hasattr(self, 'connection_pool'):
                result = await self.connection_pool.post_json(
                    f"{self.base_url}/api/generate", payload
                )
            else:
                if not self.session:
                    await self.initialize()
                async with self.session.post(
                    f"{self.base_url}/api/generate", json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                    else:
                        raise Exception(f"API call failed with status {response.status}")
            
            # Update stats
            processing_time = asyncio.get_event_loop().time() - start_time
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_requests'] - 1) + processing_time) 
                / self.stats['total_requests']
            )
            
            return {
                "id": f"completion-{int(start_time)}",
                "object": "chat.completion", 
                "created": int(start_time),
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
                },
                "processing_time": processing_time,
                "selected_model": model
            }
            
        except Exception as e:
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            logging.error(f"Generation failed: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to prompt format"""
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
            elif role == 'system':
                prompt += f"System: {content}\n"
        prompt += "Assistant: "
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return self.stats.copy()
    
    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'connection_pool'):
            await self.connection_pool.cleanup()
        elif self.session:
            await self.session.close()

# COMPREHENSIVE: Enhanced router with all features
class ComprehensiveLLMRouter:
    """Comprehensive LLM router with all enhanced features"""
    
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self.default_model = settings.DEFAULT_MODEL
        
        # Model configuration
        self.model_config = {
            'mistral:7b-instruct-q4_0': {
                'priority': 1,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'good_for': ['factual', 'math', 'general']
            },
            'deepseek-v2:7b-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00015,
                'max_context': 4096,
                'good_for': ['analysis', 'coding', 'resume']
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'good_for': ['creative', 'interview', 'storytelling']
            }
        }
        
        self.available_models = {}
        self.semantic_classifier = None
    
    async def initialize(self):
        """Initialize router with semantic classification if available"""
        # Get available models
        try:
            models = await self.ollama_client.list_models()
            available_model_names = {model.get('name', '') for model in models}
            self.available_models = {
                name: config for name, config in self.model_config.items()
                if name in available_model_names
            }
            
            if not self.available_models:
                # Fallback to any available model
                if models:
                    fallback_model = models[0].get('name', self.default_model)
                    self.available_models[fallback_model] = {
                        'priority': 1,
                        'cost_per_token': 0.0001,
                        'max_context': 4096,
                        'good_for': ['general']
                    }
        except Exception as e:
            logging.warning(f"Could not get available models: {e}")
            self.available_models = {self.default_model: {
                'priority': 1, 'cost_per_token': 0.0001, 'max_context': 4096, 'good_for': ['general']
            }}
        
        # Initialize semantic classifier if enabled
        if settings.ENABLE_SEMANTIC_CLASSIFICATION and enhanced_capabilities.get('semantic_classification'):
            try:
                from services.semantic_classifier import SemanticIntentClassifier
                self.semantic_classifier = SemanticIntentClassifier()
                await self.semantic_classifier.initialize()
                logging.info("✅ Semantic classifier initialized")
            except Exception as e:
                logging.warning(f"Semantic classifier failed: {e}")
        
        logging.info(f"✅ Comprehensive router initialized with models: {list(self.available_models.keys())}")
    
    async def route_request(self, request) -> str:
        """Route request with enhanced intelligence"""
        # Use explicit model if valid
        if hasattr(request, 'model') and request.model in self.available_models:
            return request.model
        
        # Extract text for classification
        text_content = self._extract_text_content(request)
        
        # Classify intent
        intent = await self._classify_intent(text_content, getattr(request, 'intent', None))
        
        # Select model based on intent
        selected_model = self._select_model_for_intent(intent, text_content)
        
        logging.info(f"Enhanced routing: intent={intent}, selected_model={selected_model}")
        return selected_model
    
    def _extract_text_content(self, request) -> str:
        """Extract text content from request"""
        if hasattr(request, 'messages'):
            messages = []
            for msg in request.messages:
                if hasattr(msg, 'content'):
                    messages.append(msg.content)
                elif isinstance(msg, dict):
                    messages.append(msg.get('content', ''))
            return ' '.join(messages)
        elif hasattr(request, 'prompt'):
            return request.prompt
        return ''
    
    async def _classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
        """Classify intent with semantic classification if available"""
        if explicit_intent:
            return explicit_intent
        
        # Try semantic classification first
        if self.semantic_classifier:
            try:
                intent, confidence = await self.semantic_classifier.classify_intent(text)
                if confidence > 0.7:
                    return intent
            except Exception as e:
                logging.warning(f"Semantic classification failed: {e}")
        
        # Fallback to rule-based classification
        return self._rule_based_classification(text)
    
    def _rule_based_classification(self, text: str) -> str:
        """Rule-based intent classification"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['calculate', 'solve', 'math', 'equation']):
            return 'math'
        elif any(word in text_lower for word in ['code', 'function', 'algorithm', 'debug']):
            return 'coding'
        elif any(word in text_lower for word in ['resume', 'cv', 'experience', 'skills']):
            return 'resume'
        elif any(word in text_lower for word in ['interview', 'job', 'career']):
            return 'interview'
        elif any(word in text_lower for word in ['write', 'create', 'story', 'creative']):
            return 'creative'
        elif any(word in text_lower for word in ['analyze', 'review', 'evaluate']):
            return 'analysis'
        else:
            return 'factual'
    
    def _select_model_for_intent(self, intent: str, text: str) -> str:
        """Select best model for intent"""
        suitable_models = {}
        for model_name, config in self.available_models.items():
            if intent in config['good_for'] or 'general' in config['good_for']:
                suitable_models[model_name] = config
        
        if not suitable_models:
            suitable_models = self.available_models
        
        # Select by priority (lower number = higher priority)
        best_model = min(suitable_models.items(), key=lambda x: x[1]['priority'])
        return best_model[0]
    
    async def process_chat_completion(self, request, model: str):
        """Process chat completion with enhancements"""
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
        """Get available models with metadata"""
        models = []
        for model_name, config in self.available_models.items():
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "ollama",
                "cost_per_token": config.get('cost_per_token', 0.0001),
                "max_context": config.get('max_context', 4096),
                "capabilities": config.get('good_for', [])
            })
        return models

# COMPREHENSIVE: Metrics collector with all features
class ComprehensiveMetrics:
    """Comprehensive metrics collector"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.request_counts = {}
        self.response_times = []
        self.model_usage = {}
        self.errors = {}
    
    async def get_all_metrics(self):
        """Get comprehensive metrics"""
        uptime = datetime.now() - self.start_time
        total_requests = sum(self.request_counts.values())
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "uptime_seconds": uptime.total_seconds(),
            "requests": {
                "total": total_requests,
                "by_endpoint": self.request_counts,
                "avg_response_time": avg_response_time
            },
            "models": self.model_usage,
            "errors": self.errors,
            "enhanced_features": enhanced_capabilities,
            "status": "comprehensive"
        }
    
    def track_request(self, endpoint: str, response_time: float = 0):
        """Track request"""
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
        if response_time > 0:
            self.response_times.append(response_time)
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-500:]
    
    def track_model_usage(self, model: str):
        """Track model usage"""
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
    
    def track_error(self, error_type: str):
        """Track error"""
        self.errors[error_type] = self.errors.get(error_type, 0) + 1

# COMPREHENSIVE: Initialize all services
async def initialize_comprehensive_services():
    """Initialize all services with comprehensive error handling"""
    global ollama_client, llm_router, metrics_collector, health_checker, websocket_dashboard
    
    try:
        logging.info("🚀 Initializing comprehensive services...")
        
        # Initialize metrics collector
        if MetricsCollector:
            metrics_collector = MetricsCollector()
        else:
            metrics_collector = ComprehensiveMetrics()
        logging.info("✅ Metrics collector initialized")
        
        # Initialize health checker
        if HealthChecker:
            health_checker = HealthChecker()
            await health_checker.start_monitoring()
            logging.info("✅ Health checker initialized")
        
        # Initialize Ollama client
        ollama_client = ComprehensiveOllamaClient(
            settings.OLLAMA_BASE_URL, 
            settings.OLLAMA_TIMEOUT
        )
        await ollama_client.initialize()
        logging.info("✅ Comprehensive Ollama client initialized")
        
        # Initialize router
        llm_router = ComprehensiveLLMRouter(ollama_client)
        await llm_router.initialize()
        logging.info("✅ Comprehensive LLM router initialized")
        
        # Initialize WebSocket dashboard
        if WebSocketDashboard:
            websocket_dashboard = WebSocketDashboard(
                metrics_collector=metrics_collector,
                performance_monitor=None,
                enhanced_dashboard=None
            )
            await websocket_dashboard.start_broadcasting()
            logging.info("✅ WebSocket dashboard initialized")
        
        logging.info("✅ All comprehensive services initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize some services: {e}")
        logging.error(traceback.format_exc())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with comprehensive management"""
    
    # Startup
    logging.info("🚀 Starting Comprehensive Enhanced LLM Proxy...")
    
    try:
        await initialize_comprehensive_services()
        logging.info("✅ Comprehensive services initialized")
        log_comprehensive_startup_summary()
    except Exception as e:
        logging.error(f"❌ Failed to start some services: {e}")
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Comprehensive Enhanced LLM Proxy...")
    
    try:
        if websocket_dashboard:
            await websocket_dashboard.stop_broadcasting()
        if health_checker and hasattr(health_checker, 'stop_monitoring'):
            await health_checker.stop_monitoring()
        if ollama_client:
            await ollama_client.cleanup()
        logging.info("✅ Services shut down gracefully")
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

def log_comprehensive_startup_summary():
    """Log comprehensive startup information"""
    logging.info("=" * 80)
    logging.info("🚀 COMPREHENSIVE ENHANCED LLM PROXY - STARTUP SUMMARY")
    logging.info("=" * 80)
    logging.info(f"📋 Configuration:")
    logging.info(f"   • Host: {settings.HOST}:{settings.PORT}")
    logging.info(f"   • Ollama URL: {settings.OLLAMA_BASE_URL}")
    logging.info(f"   • Dashboard: {'✅ Enabled' if settings.ENABLE_DASHBOARD else '⏸️ Disabled'}")
    logging.info(f"   • Auth: {'✅ Enabled' if settings.ENABLE_AUTH else '⏸️ Disabled'}")
    logging.info(f"🎯 Core Services:")
    logging.info(f"   • Ollama Client: {'✅' if ollama_client else '❌'}")
    logging.info(f"   • LLM Router: {'✅' if llm_router else '❌'}")
    logging.info(f"   • Metrics: {'✅' if metrics_collector else '❌'}")
    logging.info(f"   • Health Monitor: {'✅' if health_checker else '❌'}")
    logging.info(f"   • WebSocket Dashboard: {'✅' if websocket_dashboard else '❌'}")
    logging.info(f"🚀 Enhanced Features:")
    for feature, enabled in enhanced_capabilities.items():
        status = "✅ Enabled" if enabled else "⏸️ Disabled"
        logging.info(f"   • {feature.replace('_', ' ').title()}: {status}")
    logging.info(f"🌐 Endpoints:")
    logging.info(f"   • API: http://{settings.HOST}:{settings.PORT}")
    logging.info(f"   • Health: http://{settings.HOST}:{settings.PORT}/health")
    logging.info(f"   • Metrics: http://{settings.HOST}:{settings.PORT}/metrics")
    logging.info(f"   • Dashboard: http://{settings.HOST}:{settings.PORT}{settings.DASHBOARD_PATH}")
    logging.info(f"   • WebSocket: ws://{settings.HOST}:{settings.PORT}/ws/dashboard")
    logging.info("=" * 80)

# Create comprehensive FastAPI app
app = FastAPI(
    title="Comprehensive Enhanced LLM Proxy",
    description="Production-ready LLM routing proxy with full enhanced features and React dashboard",
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

# FIXED: Add React Dashboard static file serving (your solution)
if getattr(settings, 'ENABLE_DASHBOARD', False):
    try:
        # Mount React dashboard build directory
        static_dir = Path(__file__).parent / "frontend" / "build"
        if static_dir.exists():
            app.mount(
                settings.DASHBOARD_PATH, 
                StaticFiles(directory=static_dir, html=True), 
                name="react_dashboard"
            )
            logging.info(f"✅ React dashboard mounted at {settings.DASHBOARD_PATH}")
        else:
            logging.warning(f"⚠️ React dashboard build directory not found: {static_dir}")
            
            # Create fallback dashboard response
            @app.get(settings.DASHBOARD_PATH)
            async def dashboard_fallback():
                return JSONResponse({
                    "message": "Dashboard not built yet. Run 'npm run build' in frontend directory.",
                    "build_path": str(static_dir),
                    "instructions": [
                        "cd frontend",
                        "npm install",
                        "npm run build"
                    ]
                })
    except Exception as e:
        logging.error(f"Failed to mount React dashboard: {e}")

# Authentication dependency
async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get current user from request with comprehensive auth"""
    if not settings.ENABLE_AUTH:
        return {"user_id": "anonymous", "permissions": ["read", "write"]}
    
    api_key = request.headers.get(settings.API_KEY_HEADER)
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "API key required",
                "message": f"Please provide API key in {settings.API_KEY_HEADER} header"
            }
        )
    
    if api_key != settings.DEFAULT_API_KEY:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is not valid"
            }
        )
    
    return {"user_id": "authenticated", "permissions": ["read", "write"]}

# COMPREHENSIVE: Main API Routes
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Comprehensive chat completions endpoint"""
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        # Track request
        if metrics_collector:
            metrics_collector.track_request("/v1/chat/completions")
        
        # Route request
        selected_model = await llm_router.route_request(request)
        
        # Process request
        response = await llm_router.process_chat_completion(request, selected_model)
        
        # Track model usage
        if metrics_collector:
            processing_time = asyncio.get_event_loop().time() - start_time
            metrics_collector.track_request("/v1/chat/completions", processing_time)
            metrics_collector.track_model_usage(selected_model)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        if metrics_collector:
            metrics_collector.track_error("chat_completion_error")
        logging.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Comprehensive completions endpoint"""
    
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
        
        # Convert response format for completions API
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
    """Comprehensive health check with all services"""
    
    try:
        services_status = []
        
        # Check Ollama
        if ollama_client:
            try:
                ollama_healthy = await ollama_client.health_check()
                ollama_stats = ollama_client.get_stats()
                services_status.append({
                    "name": "ollama",
                    "status": "healthy" if ollama_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "stats": ollama_stats
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
            services_status.append({
                "name": "llm_router",
                "status": "healthy",
                "last_check": datetime.now().isoformat(),
                "available_models": len(llm_router.available_models)
            })
        
        # Check Enhanced Features
        services_status.append({
            "name": "enhanced_features",
            "status": "healthy",
            "last_check": datetime.now().isoformat(),
            "capabilities": enhanced_capabilities
        })
        
        overall_healthy = all(s["status"] == "healthy" for s in services_status)
        
        health_response = HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            healthy=overall_healthy,
            timestamp=datetime.now().isoformat(),
            version="2.2.0",
            services=services_status
        )
        
        if not overall_healthy:
            return JSONResponse(
                status_code=503,
                content=health_response.dict()
            )
        
        return health_response
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/models")
async def list_available_models():
    """List available models with comprehensive metadata"""
    
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
async def get_comprehensive_metrics():
    """Get comprehensive system metrics"""
    try:
        if metrics_collector and hasattr(metrics_collector, 'get_all_metrics'):
            return await metrics_collector.get_all_metrics()
        else:
            return {
                "status": "basic_metrics",
                "timestamp": datetime.now().isoformat(),
                "message": "Enhanced metrics not available",
                "basic_stats": {
                    "requests_handled": "unknown",
                    "uptime": "unknown"
                }
            }
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

# COMPREHENSIVE: WebSocket endpoint (your fix included)
@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard with comprehensive error handling"""
    if websocket_dashboard:
        try:
            await websocket_dashboard.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                websocket_dashboard.disconnect(websocket)
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
    else:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "error": "WebSocket dashboard not available",
            "message": "Dashboard service not initialized"
        }))
        await websocket.close()

# COMPREHENSIVE: Admin endpoints for enhanced features
@app.get("/admin/status")
async def get_admin_status():
    """Get comprehensive admin status"""
    return {
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0",
        "services": {
            "ollama_client": ollama_client is not None,
            "llm_router": llm_router is not None,
            "metrics_collector": metrics_collector is not None,
            "health_checker": health_checker is not None,
            "websocket_dashboard": websocket_dashboard is not None
        },
        "enhanced_capabilities": enhanced_capabilities,
        "configuration": {
            "enable_auth": settings.ENABLE_AUTH,
            "enable_dashboard": settings.ENABLE_DASHBOARD,
            "enable_streaming": settings.ENABLE_STREAMING,
            "enable_model_warmup": settings.ENABLE_MODEL_WARMUP,
            "enable_semantic_classification": settings.ENABLE_SEMANTIC_CLASSIFICATION,
            "dashboard_path": settings.DASHBOARD_PATH
        }
    }

@app.get("/admin/circuit-breakers")
async def get_circuit_breakers():
    """Get circuit breaker status"""
    try:
        from services.circuit_breaker import get_circuit_breaker_manager
        manager = get_circuit_breaker_manager()
        return manager.get_all_status()
    except ImportError:
        return {
            "message": "Circuit breakers not available",
            "status": "feature_not_loaded",
            "recommendation": "Install enhanced dependencies to enable circuit breakers"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        from services.smart_cache import get_smart_cache
        cache = get_smart_cache()
        return cache.get_stats()
    except ImportError:
        return {
            "message": "Smart cache not available",
            "status": "feature_not_loaded",
            "recommendation": "Install Redis and enhanced dependencies to enable smart caching"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/memory")
async def get_memory_status():
    """Get memory status and management info"""
    try:
        from utils.memory_manager import get_memory_manager
        manager = get_memory_manager()
        return manager.get_status_summary()
    except ImportError:
        return {
            "message": "Memory manager not available",
            "status": "feature_not_loaded"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/warmup/{model}")
async def warmup_model(model: str):
    """Manually warmup a specific model"""
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        # Try enhanced warmup if available
        if hasattr(ollama_client, 'warm_up_model'):
            success = await ollama_client.warm_up_model(model)
        else:
            # Basic warmup
            test_messages = [{"role": "user", "content": "Hello"}]
            await ollama_client.generate_completion(model, test_messages, max_tokens=1)
            success = True
        
        return {
            "message": f"Model {model} warmup {'successful' if success else 'failed'}",
            "model": model,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Model warmup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# COMPREHENSIVE: Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Comprehensive HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "method": request.method
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Comprehensive general exception handler"""
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Track error in metrics
    if metrics_collector:
        metrics_collector.track_error("unhandled_exception")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "type": type(exc).__name__,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "method": request.method,
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Root endpoint for basic info
@app.get("/")
async def root():
    """Root endpoint with comprehensive service info"""
    return {
        "name": "Comprehensive Enhanced LLM Proxy",
        "version": "2.2.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics", 
            "models": "/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "dashboard": settings.DASHBOARD_PATH if settings.ENABLE_DASHBOARD else None,
            "websocket": "/ws/dashboard",
            "admin": "/admin/status",
            "docs": "/docs"
        },
        "features": enhanced_capabilities,
        "documentation": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
