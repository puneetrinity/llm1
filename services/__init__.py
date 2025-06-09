# services/__init__.py - Updated with Enhanced Import System
from services.ollama_client import OllamaClient
from services.router import LLMRouter
from services.auth import AuthService

# Use the enhanced import manager instead of manual imports
from services.enhanced_imports import import_manager, ENHANCED_IMPORTS_AVAILABLE

# Re-export for backward compatibility
__all__ = [
    "OllamaClient",
    "LLMRouter", 
    "AuthService",
    "ENHANCED_IMPORTS_AVAILABLE"
]

# Enhanced services will be available through the enhanced_imports module
# This avoids circular imports and provides better error handling
