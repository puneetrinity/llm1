# config.py - Base Configuration
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional
import os

class Settings(BaseSettings):
    # Basic API Settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_TITLE: str = "LLM Proxy"
    API_VERSION: str = "1.0.0"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 300
    OLLAMA_MAX_RETRIES: int = 3
    
    # Authentication
    ENABLE_AUTH: bool = False
    API_KEY_HEADER: str = "X-API-Key"
    DEFAULT_API_KEY: str = "sk-default"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000
    
    # Model Settings
    DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"
    MAX_TOKENS: int = 2048
    DEFAULT_TEMPERATURE: float = 0.7
    
    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = True
    DEFAULT_RATE_LIMIT: int = 60  # requests per minute
    
    # Memory Management
    MAX_MEMORY_MB: int = 8192  # 8GB default
    CACHE_MEMORY_LIMIT_MB: int = 1024  # 1GB for cache
    MODEL_MEMORY_LIMIT_MB: int = 4096  # 4GB for models
    
    # Resource Limits
    MAX_CONCURRENT_REQUESTS: int = 10
    MAX_QUEUE_SIZE: int = 100
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    
# requirements.txt - Fixed Dependencies with Memory-Safe Versions
# Core FastAPI dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# HTTP client and middleware
aiohttp==3.9.1
python-multipart==0.0.6

# System monitoring
psutil==5.9.6

# Authentication and security
PyJWT==2.8.0

# Optional dependencies for enhanced features
# Only install these if you enable semantic classification
sentence-transformers==2.2.2  # ~500MB memory usage
faiss-cpu==1.7.4              # Lightweight CPU version
numpy==1.24.3                 # Required by transformers

# Streaming support
sse-starlette==1.6.5

# Optional GPU monitoring (install separately if needed)
# GPUtil==1.4.0

# Optional Redis for distributed caching (install separately if needed)
# redis==5.0.1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
