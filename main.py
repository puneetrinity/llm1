# main.py - FIXED for RunPod Deployment
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List
import aiohttp
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Basic configuration
class BasicConfig:
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

config = BasicConfig()

# Basic models
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

class HealthResponse(BaseModel):
    healthy: bool
    status: str
    timestamp: str
    version: str = "2.0.0"
    services: List[Dict[str, Any]] = []

# Simple Ollama client
class SimpleOllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=300)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logging.info(f"Ollama client initialized: {self.base_url}")
    
    async def cleanup(self):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        try:
            if not self.session:
                await self.initialize()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logging.error(f"Health check failed: {e}")
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
        except Exception as e:
            logging.error(f"List models failed: {e}")
            return []
    
    async def chat_completion(self, request: ChatCompletionRequest):
        try:
            if not self.session:
                await self.initialize()
            
            # Convert messages
            messages = []
            for msg in request.messages:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Prepare Ollama request
            ollama_request = {
                "model": request.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens or 150
                }
            }
            
            # Make request
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=ollama_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to OpenAI format
                    return {
                        "id": f"chatcmpl-{int(asyncio.get_event_loop().time())}",
                        "object": "chat.completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": data.get("message", {}).get("content", "")
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 50,  # Estimate
                            "completion_tokens": 50,  # Estimate
                            "total_tokens": 100
                        }
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama error {response.status}: {error_text}")
                    
        except Exception as e:
            logging.error(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global client
ollama_client = SimpleOllamaClient(config.OLLAMA_BASE_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    
    # Startup
    logging.info("🚀 Starting FIXED LLM Proxy Service...")
    
    try:
        await ollama_client.initialize()
        
        # Test Ollama connection
        health = await ollama_client.health_check()
        if health:
            logging.info("✅ Ollama connection successful")
        else:
            logging.warning("⚠️ Ollama connection failed - service may be unhealthy")
        
    except Exception as e:
        logging.error(f"❌ Startup error: {e}")
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down...")
    try:
        await ollama_client.cleanup()
    except Exception as e:
        logging.error(f"❌ Shutdown error: {e}")

# Create FastAPI app
app = FastAPI(
    title="FIXED LLM Proxy",
    description="Fixed LLM routing proxy for RunPod",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if config.CORS_ORIGINS != ['*'] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "FIXED LLM Proxy API",
        "version": "2.1.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Fixed health check"""
    
    try:
        services_status = []
        
        # Check Ollama
        ollama_healthy = await ollama_client.health_check()
        services_status.append({
            "name": "ollama",
            "status": "healthy" if ollama_healthy else "unhealthy",
            "url": config.OLLAMA_BASE_URL,
            "last_check": datetime.now().isoformat()
        })
        
        # Check models
        try:
            models = await ollama_client.list_models()
            services_status.append({
                "name": "models",
                "status": "healthy" if models else "no_models",
                "count": len(models),
                "last_check": datetime.now().isoformat()
            })
        except Exception as e:
            services_status.append({
                "name": "models",
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            })
        
        overall_healthy = ollama_healthy
        
        health_response = HealthResponse(
            healthy=overall_healthy,
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=datetime.now().isoformat(),
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
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint"""
    
    try:
        # Map common model names to available models
        model_mapping = {
            "gpt-3.5-turbo": "mistral:7b-instruct-q4_0",
            "gpt-4": "mistral:7b-instruct-q4_0",
            "gpt-4-turbo": "mistral:7b-instruct-q4_0"
        }
        
        actual_model = model_mapping.get(request.model, request.model)
        
        # Create new request with mapped model
        mapped_request = ChatCompletionRequest(
            model=actual_model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        response = await ollama_client.chat_completion(mapped_request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    
    try:
        ollama_models = await ollama_client.list_models()
        
        # Convert to OpenAI format
        models = []
        for model in ollama_models:
            models.append({
                "id": model.get("name", "unknown"),
                "object": "model",
                "created": 1677610602,
                "owned_by": "ollama"
            })
        
        # Add common model aliases
        common_models = [
            {
                "id": "gpt-3.5-turbo",
                "object": "model", 
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai"
            }
        ]
        
        return {
            "object": "list",
            "data": models + common_models
        }
        
    except Exception as e:
        logging.error(f"List models error: {e}")
        return {
            "object": "list",
            "data": [
                {
                    "id": "gpt-3.5-turbo",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "openai"
                }
            ]
        }

@app.get("/admin/status")
async def admin_status():
    """Admin status endpoint"""
    
    try:
        ollama_health = await ollama_client.health_check()
        models = await ollama_client.list_models()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "services": {
                "ollama": {
                    "healthy": ollama_health,
                    "url": config.OLLAMA_BASE_URL,
                    "models_available": len(models)
                }
            },
            "config": {
                "host": config.HOST,
                "port": config.PORT,
                "debug": config.DEBUG
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
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
    logging.info(f"Starting server on {config.HOST}:{config.PORT}")
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )
