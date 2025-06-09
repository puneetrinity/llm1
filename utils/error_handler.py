# utils/error_handler.py - Standardized Error Handling
import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, Type, Union
from datetime import datetime
from functools import wraps
from enum import Enum
import json

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    CONFIGURATION = "configuration"

class LLMProxyError(Exception):
    """Base exception for LLM Proxy with standardized structure"""
    
    def __init__(
        self, 
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Dict[str, Any] = None,
        user_message: str = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or "An error occurred while processing your request"
        self.retry_after = retry_after
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error_code": self.error_code,
            "message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "retry_after": self.retry_after
        }

class OllamaConnectionError(LLMProxyError):
    """Ollama service connection error"""
    def __init__(self, message: str, ollama_url: str = None):
        super().__init__(
            message=message,
            error_code="OLLAMA_CONNECTION_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details={"ollama_url": ollama_url},
            user_message="Unable to connect to the AI service. Please try again later."
        )

class ModelNotFoundError(LLMProxyError):
    """Model not found error"""
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found or not available",
            error_code="MODEL_NOT_FOUND",
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            details={"model_name": model_name},
            user_message=f"The requested model '{model_name}' is not available. Please check the model name."
        )

class RateLimitExceededError(LLMProxyError):
    """Rate limit exceeded error"""
    def __init__(self, limit: int, retry_after: int = 60):
        super().__init__(
            message=f"Rate limit of {limit} requests exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.LOW,
            details={"limit": limit},
            user_message=f"Rate limit exceeded. Please wait {retry_after} seconds before retrying.",
            retry_after=retry_after
        )

class AuthenticationError(LLMProxyError):
    """Authentication error"""
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            user_message="Authentication failed. Please check your API key."
        )

class ValidationError(LLMProxyError):
    """Request validation error"""
    def __init__(self, message: str, field: str = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details={"field": field} if field else {},
            user_message=f"Invalid request: {message}"
        )

class MemoryError(LLMProxyError):
    """Memory-related error"""
    def __init__(self, message: str, requested_mb: int = None, available_mb: int = None):
        super().__init__(
            message=message,
            error_code="MEMORY_ERROR",
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            details={"requeste
