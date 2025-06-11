# main.py - Clean Version with Fixed Syntax
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging
import sys
import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Simple settings from environment
class Settings:
    def __init__(self):
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', '8000'))
        self.OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.CORS_ORIGINS = os.getenv('CORS_ORIGINS', '["*"]')
        self.CORS_ALLOW_CREDENTIALS = os.getenv('CORS_ALLOW_CREDENTIALS', 'true').lower() == 'true'
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'mistral:7b-instruct-q4_0')
        self.ENABLE_SEMANTIC_CLASSIFICATION = os.getenv('ENABLE_SEMANTIC_CLASSIFICATION', 'true').lower() == 'true'
        self.ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'

settings = Settings()

# Basic models
from pydantic import BaseModel

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

class HealthResponse(BaseModel):
    status: str
    healthy: bool
    timestamp: str
    version: str = "2.2.0"
    services: List[Dict[str, Any]] = []

# Global services
ollama_client = None
llm_router = None
semantic_enabled = False

# Create enhanced router with semantic classification
class EnhancedSemanticRouter:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self.available_models = {}
        
        # Enhanced intent patterns
        self.intent_patterns = {
            'coding': r'\b(?:code|function|algorithm|debug|program|script|python|javascript|java|c\+\+|bug|error|programming|def\s+\w+|class\s+\w+)\b',
            'resume': r'\b(?:resume|cv|experience|skills|qualifications|work history|analyze.*resume|technical skills|candidate|hiring)\b',
            'analysis': r'\b(?:analyze|review|evaluate|assess|compare|examine|advantages|disadvantages|pros.*cons|microservices|architecture|vs\.|versus)\b',
            'interview': r'\b(?:interview|job|career|hiring|prepare.*interview|interview.*questions|google|microsoft|amazon|software engineer)\b',
            'creative': r'\b(?:write|create|compose|story|poem|creative|imagine|generate|blog|article|narrative|fiction)\b',
            'math': r'\b(?:calculate|compute|solve|equation|math|arithmetic|area|volume|percentage|formula|\d+\s*[\+\-\*\/\%\^]\s*\d+|radius|circle)\b',
            'factual': r'\b(?:what is|who is|when did|where is|define|explain|fact|capital|population|history)\b'
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
        
        logging.info("Enhanced Semantic Router initialized")
    
    async def initialize(self):
        """Initialize router and get available models"""
        try:
            models = await self.ollama_client.list_models()
            self.available_models = {model['name']: model for model in models}
            logging.info(f"Router initialized with {len(self.available_models)} models")
        except Exception as e:
            logging.error(f"Failed to initialize router: {e}")
            # Set some defaults
            self.available_models = {
                'mistral:7b-instruct-q4_0': {},
                'deepseek-v2:7b-q4_0': {},
                'llama3:8b-instruct-q4_0': {}
            }
    
    def classify_intent(self, text: str) -> str:
        """Classify intent using enhanced patterns"""
        import re
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, pattern in self.intent_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                score = len(matches)
                
                # Boost scores for specific keywords
                if intent == 'coding' and any(word in text_lower for word in ['debug', 'error', 'bug', 'def ', 'function']):
                    score += 3
                elif intent == 'resume' and any(word in text_lower for word in ['technical skills', 'analyze', 'experience']):
                    score += 3
                elif intent == 'analysis' and any(word in text_lower for word in ['compare', 'advantages', 'vs', 'versus']):
                    score += 3
                elif intent == 'math' and any(word in text_lower for word in ['calculate', 'area', 'radius', 'circle']):
                    score += 3
                elif intent == 'interview' and any(word in text_lower for word in ['prepare', 'questions', 'software engineer']):
                    score += 3
                
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            logging.info(f"🎯 Classified '{text[:50]}...' as '{best_intent}' (score: {intent_scores[best_intent]})")
            return best_intent
        
        return 'general'
    
    async def route_request(self, request: ChatCompletionRequest) -> str:
        """Route request based on semantic classification"""
        
        # Extract text content
        text_content = ' '.join(
            msg.content for msg in request.messages 
            if msg.role == 'user'
        )
        
        # Classify intent
        intent = self.classify_intent(text_content)
        
        # Get preferred model
        preferred_model = self.intent_model_map.get(intent, 'mistral:7b-instruct-q4_0')
        
        # Check if preferred model is available
        if preferred_model in self.available_models:
            selected_model = preferred_model
        else:
            # Fallback to any available model
            if self.available_models:
                selected_model = list(self.available_models.keys())[0]
            else:
                selected_model = 'mistral:7b-instruct-q4_0'
        
        logging.info(f"🚀 SEMANTIC ROUTING: '{text_content[:50]}...' -> intent='{intent}' -> model='{selected_model}'")
        return selected_model
    
    async def process_chat_completion(self, request: ChatCompletionRequest, model: str):
        """Process chat completion with selected model"""
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Use ollama client to generate response
        response = await self.ollama_client.generate_completion(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 150,
            top_p=request.top_p
        )
        
        return response
    
    async def get_available_models(self):
        """Get available models"""
        return [{"id": name, "object": "model"} for name in self.available_models.keys()]
    
    def get_classification_stats(self):
        """Get classification stats"""
        return {
            'semantic_enabled': True,
            'available_intents': list(self.intent_patterns.keys()),
            'intent_model_mapping': self.intent_model_map,
            'available_models': list(self.available_models.keys())
        }

# Basic Ollama client
class BasicOllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def initialize(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        logging.info(f"Ollama client initialized: {self.base_url}")
    
    async def cleanup(self):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def list_models(self):
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('models', [])
                return []
        except Exception:
            return []
    
    async def generate_completion(self, model: str, messages: List[Dict], **kwargs):
        try:
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
            
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
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

async def initialize_services():
    """Initialize all services"""
    global ollama_client, llm_router, semantic_enabled
    
    logging.info("🚀 Initializing Enhanced LLM Proxy Services...")
    
    try:
        # Initialize Ollama client
        ollama_client = BasicOllamaClient(settings.OLLAMA_BASE_URL)
        await ollama_client.initialize()
        logging.info("✅ Ollama client initialized")
        
        # Initialize semantic router if enabled
        if settings.ENABLE_SEMANTIC_CLASSIFICATION:
            llm_router = EnhancedSemanticRouter(ollama_client)
            await llm_router.initialize()
            semantic_enabled = True
            logging.info("✅ Enhanced Semantic Router initialized")
        else:
            # Basic router fallback (would need to implement)
            llm_router = None
            semantic_enabled = False
            logging.info("⚠️ Semantic classification disabled")
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize services: {e}")
        logging.error(traceback.format_exc())
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("🚀 Starting Enhanced LLM Proxy...")
    await initialize_services()
    
    logging.info("=" * 60)
    logging.info("🎯 Service Status:")
    logging.info(f"   • Ollama Client: {'✅' if ollama_client else '❌'}")
    logging.info(f"   • Router: {'✅' if llm_router else '❌'}")
    logging.info(f"   • Semantic Routing: {'✅' if semantic_enabled else '❌'}")
    logging.info("=" * 60)
    
    yield
    
    # Shutdown
    if ollama_client:
        await ollama_client.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Enhanced LLM Proxy with Semantic Routing",
    description="Production-ready LLM routing proxy",
    version="2.2.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Dependencies
async def get_router():
    if not llm_router:
        raise HTTPException(status_code=503, detail="Router not available")
    return llm_router

async def get_client():
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama client not available")
    return ollama_client

# Routes
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    router = Depends(get_router)
):
    """Enhanced chat completions with semantic routing"""
    start_time = datetime.now()
    
    try:
        # Route request
        selected_model = await router.route_request(request)
        
        # Process request
        response = await router.process_chat_completion(request, selected_model)
        
        # Add metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if isinstance(response, dict):
            response.update({
                'selected_model': selected_model,
                'routing_reason': 'semantic_classification',
                'processing_time': processing_time
            })
        
        logging.info(f"✅ Request completed: {selected_model} in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = []
    
    if ollama_client:
        try:
            healthy = await ollama_client.health_check()
            services.append({
                "name": "ollama",
                "status": "healthy" if healthy else "unhealthy",
                "last_check": datetime.now().isoformat()
            })
        except Exception as e:
            services.append({
                "name": "ollama", 
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            })
    
    if llm_router:
        services.append({
            "name": "router",
            "status": "healthy",
            "type": "EnhancedSemanticRouter",
            "semantic_enabled": semantic_enabled,
            "last_check": datetime.now().isoformat()
        })
    
    overall_healthy = all(s["status"] == "healthy" for s in services)
    
    return HealthResponse(
        status="healthy" if overall_healthy else "unhealthy",
        healthy=overall_healthy,
        timestamp=datetime.now().isoformat(),
        services=services
    )

@app.get("/admin/router/status")
async def get_router_status():
    """Get router status"""
    if not llm_router:
        raise HTTPException(status_code=503, detail="Router not available")
    
    return {
        "router_type": "EnhancedSemanticRouter",
        "semantic_enabled": semantic_enabled,
        "classification_stats": llm_router.get_classification_stats(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models(router = Depends(get_router)):
    """List available models"""
    models = await router.get_available_models()
    return {"object": "list", "data": models}

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return {
        "status": "enhanced_metrics",
        "timestamp": datetime.now().isoformat(),
        "semantic_enabled": semantic_enabled,
        "router_available": llm_router is not None,
        "ollama_available": ollama_client is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
