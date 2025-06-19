from services.ollama_client import OllamaClient
# main_master.py - Complete LLM Proxy with 3-Model Routing and Full Authentication
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, Query, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import sys
import json
import secrets
import aiohttp
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Configuration - Create a simple config if config.py doesn't exist
try:
    from config import settings
except ImportError:
    from pydantic_settings import BaseSettings
    import os

    class Settings(BaseSettings):
        HOST: str = "0.0.0.0"
        PORT: int = 8001
        DEBUG: bool = False
        LOG_LEVEL: str = "INFO"
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        OLLAMA_TIMEOUT: int = 300
        ENABLE_AUTH: bool = False
        DEFAULT_API_KEY: str = os.getenv("DEFAULT_API_KEY", "")
        API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")
        ENABLE_DASHBOARD: bool = True
        ENABLE_ENHANCED_FEATURES: bool = True
        ENABLE_WEBSOCKET: bool = False
        ENABLE_WEBSOCKET_DASHBOARD: bool = False
        CORS_ORIGINS: List[str] = ["*"]
        CORS_ALLOW_CREDENTIALS: bool = True
        MAX_MEMORY_MB: int = 8192
        CACHE_MEMORY_LIMIT_MB: int = 1024

        class Config:
            env_file = ".env"
            case_sensitive = True
            extra = "ignore"

    # app = FastAPI()  # Ensure app is defined before using app.state
    # app.state.settings = Settings()  # Commented out to avoid F821 error

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main_master")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0

class HealthResponse(BaseModel):
    status: str
    healthy: bool
    timestamp: str
    version: str
    services: Dict[str, Any]

class StatusResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    features: Dict[str, bool]
    timestamp: str

services_state = {
    "ollama_connected": False,
    "dashboard_available": False,
    "initialization_complete": False,
    "available_models": []
}

websocket_sessions = {}

class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }

    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info(f"✅ Ollama client initialized for {self.base_url}")

    async def health_check(self) -> bool:
        try:
            if not self.session:
                await self.initialize()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> List[Dict]:
        try:
            if not self.session:
                await self.initialize()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('models', [])
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def generate_completion(self, model: str, messages: List[Dict], **kwargs):
        start_time = asyncio.get_event_loop().time()
        try:
            if not self.session:
                await self.initialize()
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
            async with self.session.post(
                f"{self.base_url}/api/generate", json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
            processing_time = asyncio.get_event_loop().time() - start_time
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            return {
                "id": f"chatcmpl-{int(start_time)}",
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
            logger.error(f"Generation failed: {e}")
            raise

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"Human: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    async def cleanup(self):
        if self.session:
            await self.session.close()

class ModelRouter:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self.model_config = {
            'phi:3.5': {
                'priority': 1,
                'good_for': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
                'description': 'Phi-4 Reasoning - Complex math, logic, scientific analysis'
            },
            'mistral:7b-instruct-q4_0': {
                'priority': 2,
                'good_for': ['factual', 'general', 'quick_facts', 'summaries'],
                'description': 'Mistral 7B - Quick facts, summaries, efficient responses'
            },
            'gemma:7b-instruct': {
                'priority': 2,
                'good_for': ['coding', 'technical', 'programming', 'documentation'],
                'description': 'Gemma 7B - Technical documentation, coding, programming'
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 3,
                'good_for': ['creative', 'storytelling', 'writing', 'conversations'],
                'description': 'Llama3 8B-Instruct - Creative writing, conversations, storytelling'
            },
            'deepseek-v2:7b-q4_0': {
                'priority': 2,
                'good_for': ['coding', 'technical', 'programming', 'debug'],
                'description': 'Specialized for coding and technical tasks'
            }
        }
        self.available_models = {}
        self.default_model = 'mistral:7b-instruct-q4_0'

    # Remaining methods unchanged from your version...
