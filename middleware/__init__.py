# middleware/__init__.py - Middleware Package
"""
Custom middleware package for LLM Proxy

Provides:
- Security middleware with headers and rate limiting
- CORS configuration
- Request logging and monitoring
- Authentication middleware integration
"""

from .security import (
    SecurityMiddleware,
    SecurityHeadersMiddleware,
    RateLimiter,
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
    CORSConfig
)

__all__ = [
    "SecurityMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimiter", 
    "RateLimitingMiddleware",
    "RequestLoggingMiddleware",
    "CORSConfig"
]

__version__ = "1.0.0"
