# config_enhanced.py - Fixed Enhanced Configuration
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import List, Optional
import os

class EnhancedSettings(BaseSettings):
    """Enhanced application settings with proper field definitions"""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'  # Allow extra environment variables without errors
    )
    
    # Basic application settings
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    HOST: str = Field(
        default="0.0.0.0",
        description="Host address to bind to"
    )
    
    PORT: int = Field(
        default=8000,
        description="Port to listen on"
    )
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Ollama configuration
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    
    OLLAMA_TIMEOUT: int = Field(
        default=300,
        description="Ollama request timeout in seconds"
    )
    
    # Model settings
    DEFAULT_MODEL: str = Field(
        default="mistral:7b-instruct-q4_0",
        description="Default model to use"
    )
    
    ENABLE_SEMANTIC_CLASSIFICATION: bool = Field(
        default=False,
        description="Enable semantic classification for model routing"
    )
    
    # Feature toggles
    ENABLE_STREAMING: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    
    ENABLE_MODEL_WARMUP: bool = Field(
        default=True,
        description="Enable model warmup service"
    )
    
    ENABLE_CACHING: bool = Field(
        default=False,
        description="Enable response caching"
    )
    
    ENABLE_AUTH: bool = Field(
        default=False,
        description="Enable authentication"
    )
    
    ENABLE_RATE_LIMITING: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    
    # Memory and performance
    MAX_MEMORY_MB: int = Field(
        default=4096,
        description="Maximum memory usage in MB"
    )
    
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=10,
        description="Maximum concurrent requests"
    )
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    
    # Rate limiting
    DEFAULT_RATE_LIMIT: str = Field(
        default="100/hour",
        description="Default rate limit"
    )
    
    # Cache settings
    CACHE_TTL_SECONDS: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    
    CACHE_MAX_SIZE: int = Field(
        default=100,
        description="Maximum cache size"
    )
    
    # Model warmup settings
    WARMUP_MODELS: List[str] = Field(
        default=["mistral:7b-instruct-q4_0"],
        description="Models to warm up on startup"
    )
    
    WARMUP_CONCURRENT: int = Field(
        default=3,
        description="Number of models to warm up concurrently"
    )
    
    # Health check settings
    HEALTH_CHECK_INTERVAL: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    @validator('OLLAMA_BASE_URL')
    def validate_ollama_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('OLLAMA_BASE_URL must start with http:// or https://')
        return v
    
    @validator('DEFAULT_RATE_LIMIT')
    def validate_rate_limit(cls, v):
        # Basic validation for rate limit format
        if '/' not in v:
            raise ValueError('DEFAULT_RATE_LIMIT must be in format "number/period"')
        return v

class DevelopmentSettings(EnhancedSettings):
    """Development-specific settings"""
    
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENABLE_AUTH: bool = False
    ENABLE_RATE_LIMITING: bool = False

class ProductionSettings(EnhancedSettings):
    """Production-specific settings"""
    
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    ENABLE_AUTH: bool = True
    ENABLE_RATE_LIMITING: bool = True
    CORS_ORIGINS: List[str] = Field(
        default=[],  # Should be configured with actual domains
        description="Allowed CORS origins for production"
    )

def get_settings() -> EnhancedSettings:
    """Get settings based on environment"""
    
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "development":
        return DevelopmentSettings()
    else:
        return EnhancedSettings()
