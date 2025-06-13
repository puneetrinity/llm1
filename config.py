# config.py - Master Configuration (Single Source of Truth)
from pydantic_settings import BaseSettings
from typing import List

class MasterSettings(BaseSettings):
    # CRITICAL: This line fixes ALL Pydantic v2 issues
    model_config = {"extra": "ignore", "env_file": ".env"}
    
    # Core server settings
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    
    # Ollama configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 300
    DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"
    
    # CORS settings - FIXED for React development
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001", "*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Authentication
    ENABLE_AUTH: bool = False
    DEFAULT_API_KEY: str = "sk-dev-key-change-in-production"
    API_KEY_HEADER: str = "X-API-Key"
    
    # Feature toggles - Start simple, add complexity later
    ENABLE_ENHANCED_FEATURES: bool = False
    ENABLE_DASHBOARD: bool = True
    ENABLE_WEBSOCKET: bool = False
    
    # Memory and performance
    MAX_MEMORY_MB: int = 4096
    CACHE_MEMORY_LIMIT_MB: int = 512

# Global settings instance - Import this everywhere
settings = MasterSettings()
