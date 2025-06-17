# config_enhanced.py - Fixed Enhanced Configuration
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List, Any

class EnhancedSettings(BaseSettings):
    """Enhanced application settings with proper Pydantic configuration"""
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": True,
        "extra": "ignore"  # This fixes the Pydantic validation error
    }
    
    # Basic application settings
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    HOST: str = Field(default="0.0.0.0", description="Host address")
    PORT: int = Field(default=8000, description="Port number")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Ollama configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama base URL")
    OLLAMA_TIMEOUT: int = Field(default=300, description="Ollama timeout in seconds")
    
    # Basic model settings
    DEFAULT_MODEL: str = Field(default="mistral:7b-instruct-q4_0", description="Default model")
    MAX_MEMORY_MB: int = Field(default=4096, description="Max memory in MB")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"], description="CORS origins")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, description="Allow CORS credentials")
    
    # Feature flags
    ENABLE_AUTH: bool = Field(default=False, description="Enable authentication")
    ENABLE_RATE_LIMITING: bool = Field(default=False, description="Enable rate limiting")
    DEFAULT_RATE_LIMIT: str = Field(default="100/hour", description="Default rate limit")
    
    # Semantic Classification Settings
    ENABLE_SEMANTIC_CLASSIFICATION: bool = Field(default=True, description="Enable semantic classification")
    SEMANTIC_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Semantic model name")
    SEMANTIC_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Semantic confidence threshold")
    CLASSIFICATION_CACHE_SIZE: int = Field(default=1000, description="Classification cache size")
    
    # Streaming Settings
    ENABLE_STREAMING: bool = Field(default=True, description="Enable streaming")
    STREAM_CHUNK_SIZE: int = Field(default=1024, description="Stream chunk size")
    STREAM_TIMEOUT: int = Field(default=300, description="Stream timeout in seconds")
    
    # Model Warmup Settings
    ENABLE_MODEL_WARMUP: bool = Field(default=True, description="Enable model warmup")
    WARMUP_INTERVAL_MINUTES: int = Field(default=5, description="Warmup interval in minutes")
    WARMUP_TIMEOUT_SECONDS: int = Field(default=30, description="Warmup timeout in seconds")
    
    # Enhanced Model Configuration
    MODEL_PRIORITIES: Dict[str, int] = Field(
        default={
            "phi:3.5": 1,                           # Highest priority (reasoning)
            "mistral:7b-instruct-q4_0": 2,          # High priority (general)
            "gemma:7b-instruct": 2,                 # High priority (technical)  
            "llama3:8b-instruct-q4_0": 3            # Medium priority (creative)
        },
        description="Model priorities for 4-model system"
    )
    
    # Memory Management
    MAX_CONCURRENT_MODELS: int = Field(default=3, description="Max concurrent models")
    MODEL_IDLE_TIMEOUT_MINUTES: int = Field(default=30, description="Model idle timeout in minutes")
    ENABLE_DYNAMIC_MODEL_LOADING: bool = Field(default=True, description="Enable dynamic model loading")
    
    # Enhanced Caching
    ENABLE_SEMANTIC_CACHE: bool = Field(default=True, description="Enable semantic cache")
    SEMANTIC_CACHE_TTL: int = Field(default=7200, description="Semantic cache TTL in seconds")
    CACHE_COMPRESSION: bool = Field(default=True, description="Enable cache compression")
    
    # Performance Monitoring
    ENABLE_DETAILED_METRICS: bool = Field(default=True, description="Enable detailed metrics")
    METRICS_EXPORT_INTERVAL: int = Field(default=60, description="Metrics export interval in seconds")
    PERFORMANCE_LOGGING: bool = Field(default=True, description="Enable performance logging")

# Dashboard Settings (Required for enhanced features)
    ENABLE_DASHBOARD: bool = Field(default=True, description="Enable dashboard")
    ENABLE_WEBSOCKET_DASHBOARD: bool = Field(default=True, description="Enable WebSocket dashboard")  
    DASHBOARD_UPDATE_INTERVAL: int = Field(default=10, description="Dashboard update interval")
    DASHBOARD_PATH: str = Field(default="/dashboard", description="Dashboard path")


    # Dashboard Settings (Added for compatibility)
    ENABLE_DASHBOARD: bool = Field(default=True, description="Enable dashboard")
    ENABLE_WEBSOCKET_DASHBOARD: bool = Field(default=True, description="Enable WebSocket dashboard")
    DASHBOARD_UPDATE_INTERVAL: int = Field(default=10, description="Dashboard update interval")


    # Dashboard Settings (Added for compatibility)
    ENABLE_DASHBOARD: bool = Field(default=True, description="Enable dashboard")
    ENABLE_WEBSOCKET_DASHBOARD: bool = Field(default=True, description="Enable WebSocket dashboard")
    DASHBOARD_UPDATE_INTERVAL: int = Field(default=10, description="Dashboard update interval")


    # Dashboard Settings (Added for compatibility)
    ENABLE_DASHBOARD: bool = Field(default=True, description="Enable dashboard")
    ENABLE_WEBSOCKET_DASHBOARD: bool = Field(default=True, description="Enable WebSocket dashboard")
    DASHBOARD_UPDATE_INTERVAL: int = Field(default=10, description="Dashboard update interval")

def get_settings() -> EnhancedSettings:
    """Get application settings"""
    return EnhancedSettings()
