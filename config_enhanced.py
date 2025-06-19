# config_enhanced.py - Enhanced configuration with all features
import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """Enhanced settings for production LLM Proxy"""
    
    # Environment
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8001, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_TIMEOUT: int = Field(default=300, env="OLLAMA_TIMEOUT")
    OLLAMA_MAX_RETRIES: int = Field(default=3, env="OLLAMA_MAX_RETRIES")
    OLLAMA_KEEP_ALIVE: str = Field(default="5m", env="OLLAMA_KEEP_ALIVE")
    OLLAMA_MAX_LOADED_MODELS: int = Field(default=4, env="OLLAMA_MAX_LOADED_MODELS")
    
    # Model Configuration (4-Model Setup)
    DEFAULT_MODEL: str = Field(default="mistral:7b-instruct-q4_0", env="DEFAULT_MODEL")
    PHI_MODEL: str = Field(default="phi3.5", env="PHI_MODEL")
    MISTRAL_MODEL: str = Field(default="mistral:7b-instruct-q4_0", env="MISTRAL_MODEL")
    GEMMA_MODEL: str = Field(default="gemma:7b-instruct", env="GEMMA_MODEL")
    LLAMA_MODEL: str = Field(default="llama3:8b-instruct-q4_0", env="LLAMA_MODEL")
    
    # Feature Flags
    ENABLE_AUTH: bool = Field(default=False, env="ENABLE_AUTH")
    ENABLE_CACHE: bool = Field(default=True, env="ENABLE_CACHE")
    ENABLE_STREAMING: bool = Field(default=True, env="ENABLE_STREAMING")
    ENABLE_WEBSOCKET: bool = Field(default=False, env="ENABLE_WEBSOCKET")
    ENABLE_DASHBOARD: bool = Field(default=True, env="ENABLE_DASHBOARD")
    ENABLE_MODEL_ROUTING: bool = Field(default=True, env="ENABLE_MODEL_ROUTING")
    ENABLE_4_MODEL_ROUTING: bool = Field(default=True, env="ENABLE_4_MODEL_ROUTING")
    ENABLE_SEMANTIC_CLASSIFICATION: bool = Field(default=True, env="ENABLE_SEMANTIC_CLASSIFICATION")
    ENABLE_MODEL_WARMUP: bool = Field(default=True, env="ENABLE_MODEL_WARMUP")
    ENABLE_CIRCUIT_BREAKER: bool = Field(default=True, env="ENABLE_CIRCUIT_BREAKER")
    ENABLE_RATE_LIMITING: bool = Field(default=False, env="ENABLE_RATE_LIMITING")
    ENABLE_ENHANCED_FEATURES: bool = Field(default=True, env="ENABLE_ENHANCED_FEATURES")
    
    # Authentication
    API_KEYS: List[str] = Field(default_factory=lambda: ["sk-default-key"], env="API_KEYS")
    DEFAULT_API_KEY: Optional[str] = Field(default=None, env="DEFAULT_API_KEY")
    
    # Cache Configuration
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    CACHE_MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")
    CACHE_EVICTION_POLICY: str = Field(default="lru", env="CACHE_EVICTION_POLICY")
    CACHE_MEMORY_LIMIT_MB: int = Field(default=1024, env="CACHE_MEMORY_LIMIT_MB")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    RATE_LIMIT_BURST_SIZE: int = Field(default=10, env="RATE_LIMIT_BURST_SIZE")
    
    # Memory Management
    MAX_MEMORY_MB: int = Field(default=8192, env="MAX_MEMORY_MB")
    MODEL_MEMORY_LIMIT_MB: int = Field(default=4096, env="MODEL_MEMORY_LIMIT_MB")
    SEMANTIC_MODEL_MAX_MEMORY_MB: int = Field(default=512, env="SEMANTIC_MODEL_MAX_MEMORY_MB")
    
    # Model Routing Configuration
    ROUTING_STRATEGY: str = Field(default="intelligent", env="ROUTING_STRATEGY")  # intelligent, round-robin, random
    MODEL_SELECTION_THRESHOLD: float = Field(default=0.7, env="MODEL_SELECTION_THRESHOLD")
    ENABLE_FALLBACK_ROUTING: bool = Field(default=True, env="ENABLE_FALLBACK_ROUTING")
    
    # Semantic Classification
    SEMANTIC_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2", env="SEMANTIC_MODEL_NAME")
    SEMANTIC_CACHE_SIZE: int = Field(default=500, env="SEMANTIC_CACHE_SIZE")
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(default=0.85, env="SEMANTIC_SIMILARITY_THRESHOLD")
    
    # Model Warmup Configuration
    WARMUP_INTERVAL_SECONDS: int = Field(default=300, env="WARMUP_INTERVAL_SECONDS")  # 5 minutes
    WARMUP_TIMEOUT_SECONDS: int = Field(default=30, env="WARMUP_TIMEOUT_SECONDS")
    WARMUP_MODELS: List[str] = Field(
        default_factory=lambda: ["phi3.5", "mistral:7b-instruct-q4_0"],
        env="WARMUP_MODELS"
    )
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=30, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = Field(default="Exception", env="CIRCUIT_BREAKER_EXPECTED_EXCEPTION")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ORIGINS")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    CORS_ALLOW_METHODS: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="CORS_ALLOW_METHODS"
    )
    CORS_ALLOW_HEADERS: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ALLOW_HEADERS")
    
    # Performance Tuning
    CONNECTION_POOL_SIZE: int = Field(default=100, env="CONNECTION_POOL_SIZE")
    CONNECTION_TIMEOUT: int = Field(default=30, env="CONNECTION_TIMEOUT")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")
    MAX_CONCURRENT_REQUESTS: int = Field(default=50, env="MAX_CONCURRENT_REQUESTS")
    
    # Monitoring and Metrics
    ENABLE_PROMETHEUS_METRICS: bool = Field(default=False, env="ENABLE_PROMETHEUS_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    COLLECT_DETAILED_METRICS: bool = Field(default=True, env="COLLECT_DETAILED_METRICS")
    
    # Security
    ENABLE_HTTPS: bool = Field(default=False, env="ENABLE_HTTPS")
    SSL_CERT_FILE: Optional[str] = Field(default=None, env="SSL_CERT_FILE")
    SSL_KEY_FILE: Optional[str] = Field(default=None, env="SSL_KEY_FILE")
    TRUSTED_HOSTS: List[str] = Field(default_factory=lambda: ["*"], env="TRUSTED_HOSTS")
    
    # Model Capabilities (for routing)
    MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "phi3.5": {
                "good_for": ["math", "reasoning", "logic", "analysis"],
                "context_length": 4096,
                "speed": "fast",
                "quality": "high",
                "size": "3.8B"
            },
            "mistral:7b-instruct-q4_0": {
                "good_for": ["general", "conversation", "factual", "balanced"],
                "context_length": 8192,
                "speed": "medium",
                "quality": "good",
                "size": "7B"
            },
            "gemma:7b-instruct": {
                "good_for": ["coding", "technical", "documentation", "structured"],
                "context_length": 8192,
                "speed": "medium",
                "quality": "high",
                "size": "7B"
            },
            "llama3:8b-instruct-q4_0": {
                "good_for": ["creative", "writing", "storytelling", "detailed"],
                "context_length": 8192,
                "speed": "slower",
                "quality": "excellent",
                "size": "8B"
            }
        }
    )
    
    # Logging Configuration
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    LOG_FILE: Optional[str] = Field(default="data/logs/app.log", env="LOG_FILE")
    LOG_MAX_SIZE_MB: int = Field(default=100, env="LOG_MAX_SIZE_MB")
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    @validator("API_KEYS", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v
    
    @validator("WARMUP_MODELS", pre=True)
    def parse_warmup_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.MODEL_CAPABILITIES.get(model_name, {})
    
    def get_all_models(self) -> List[str]:
        """Get list of all configured models"""
        return [self.PHI_MODEL, self.MISTRAL_MODEL, self.GEMMA_MODEL, self.LLAMA_MODEL]
    
    def get_warmup_config(self) -> Dict[str, Any]:
        """Get warmup configuration"""
        return {
            "enabled": self.ENABLE_MODEL_WARMUP,
            "interval": self.WARMUP_INTERVAL_SECONDS,
            "timeout": self.WARMUP_TIMEOUT_SECONDS,
            "models": self.WARMUP_MODELS
        }
    
    def validate_settings(self) -> bool:
        """Validate settings consistency"""
        # Check if routing is enabled but no models configured
        if self.ENABLE_MODEL_ROUTING and not self.get_all_models():
            raise ValueError("Model routing enabled but no models configured")
        
        # Check memory limits
        if self.MODEL_MEMORY_LIMIT_MB > self.MAX_MEMORY_MB:
            raise ValueError("Model memory limit exceeds max memory limit")
        
        # Check API keys if auth is enabled
        if self.ENABLE_AUTH and not self.API_KEYS:
            raise ValueError("Authentication enabled but no API keys configured")
        
        return True

# Create singleton instance
settings = Settings()

# Validate on import
settings.validate_settings()
