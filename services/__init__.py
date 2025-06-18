# services/__init__.py - Safe imports without circular dependencies
"""
Core services package with safe import handling
"""

# Basic imports that are always safe
try:
    from .ollama_client import OllamaClient
except ImportError as e:
    print(f"Warning: Could not import OllamaClient: {e}")
    OllamaClient = None

try:
    from .router import LLMRouter
except ImportError as e:
    print(f"Warning: Could not import LLMRouter: {e}")
    LLMRouter = None

try:
    from .auth import AuthService
except ImportError as e:
    print(f"Warning: Could not import AuthService: {e}")
    AuthService = None

# Enhanced imports with fallbacks
try:
    from .circuit_breaker import CircuitBreakerManager, get_circuit_breaker_manager
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError as e:
    print(f"Info: Circuit breaker not available: {e}")
    CircuitBreakerManager = None
    def get_circuit_breaker_manager(): return None
    CIRCUIT_BREAKER_AVAILABLE = False

__all__ = [
    "OllamaClient",
    "LLMRouter",
    "AuthService",
    "CIRCUIT_BREAKER_AVAILABLE"
]

if CIRCUIT_BREAKER_AVAILABLE:
    __all__.extend(["CircuitBreakerManager", "get_circuit_breaker_manager"])
