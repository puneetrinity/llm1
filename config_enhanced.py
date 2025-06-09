# Configuration for enhanced features
# config_enhanced.py - Enhanced Configuration
from config import Settings as BaseSettings
from typing import Dict, List, Any

class EnhancedSettings(BaseSettings):
    # Semantic Classification Settings
    ENABLE_SEMANTIC_CLASSIFICATION: bool = True
    SEMANTIC_MODEL: str = "all-MiniLM-L6-v2"
    SEMANTIC_CONFIDENCE_THRESHOLD: float = 0.7
    CLASSIFICATION_CACHE_SIZE: int = 1000
    
    # Streaming Settings
    ENABLE_STREAMING: bool = True
    STREAM_CHUNK_SIZE: int = 1024
    STREAM_TIMEOUT: int = 300  # 5 minutes
    
    # Model Warmup Settings
    ENABLE_MODEL_WARMUP: bool = True
    WARMUP_INTERVAL_MINUTES: int = 5
    WARMUP_TIMEOUT_SECONDS: int = 30
    
    # Enhanced Model Configuration
    MODEL_PRIORITIES: Dict[str, int] = {
        "mistral:7b-instruct-q4_0": 1,      # Highest priority
        "deepseek-v2:7b-q4_0": 2,           # Medium priority  
        "llama3:8b-instruct-q4_0": 2        # Medium priority
    }
    
    # Memory Management
    MAX_CONCURRENT_MODELS: int = 3
    MODEL_IDLE_TIMEOUT_MINUTES: int = 30
    ENABLE_DYNAMIC_MODEL_LOADING: bool = True
    
    # Enhanced Caching
    ENABLE_SEMANTIC_CACHE: bool = True
    SEMANTIC_CACHE_TTL: int = 7200  # 2 hours
    CACHE_COMPRESSION: bool = True
    
    # Performance Monitoring
    ENABLE_DETAILED_METRICS: bool = True
    METRICS_EXPORT_INTERVAL: int = 60  # seconds
    PERFORMANCE_LOGGING: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"