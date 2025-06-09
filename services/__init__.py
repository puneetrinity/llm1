# services/__init__.py
from .ollama_client import OllamaClient
from .router import LLMRouter
from .auth import AuthService

# Enhanced services (optional imports)
try:
    from .enhanced_ollama_client import EnhancedOllamaClient
    from .enhanced_router import EnhancedLLMRouter
    from .streaming import StreamingService
    from .model_warmup import ModelWarmupService
    from .semantic_cache import SemanticCache
    from .semantic_classifier import SemanticIntentClassifier
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

__all__ = [
    "OllamaClient",
    "LLMRouter", 
    "AuthService",
    "ENHANCED_AVAILABLE"
]

if ENHANCED_AVAILABLE:
    __all__.extend([
        "EnhancedOllamaClient",
        "EnhancedLLMRouter",
        "StreamingService", 
        "ModelWarmupService",
        "SemanticCache",
        "SemanticIntentClassifier"
    ])
