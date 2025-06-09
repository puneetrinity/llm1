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
            details={"requested_mb": requested_mb, "available_mb": available_mb},
            user_message="Insufficient memory to process request. Please try again later."
        )

class ServiceUnavailableError(LLMProxyError):
    """Service temporarily unavailable"""
    def __init__(self, service_name: str, retry_after: int = 30):
        super().__init__(
            message=f"Service '{service_name}' is temporarily unavailable",
            error_code="SERVICE_UNAVAILABLE",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details={"service_name": service_name},
            user_message=f"Service temporarily unavailable. Please try again in {retry_after} seconds.",
            retry_after=retry_after
        )

class ErrorHandler:
    """Centralized error handling with consistent patterns"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_callbacks: Dict[str, Callable] = {}
        
    def register_callback(self, error_code: str, callback: Callable):
        """Register callback for specific error types"""
        self.error_callbacks[error_code] = callback
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None,
        request_id: str = None
    ) -> LLMProxyError:
        """Handle and standardize any error"""
        
        context = context or {}
        
        # Convert to standardized error if needed
        if isinstance(error, LLMProxyError):
            std_error = error
        else:
            std_error = self._convert_to_standard_error(error, context)
        
        # Track error
        self._track_error(std_error)
        
        # Log error with context
        self._log_error(std_error, context, request_id)
        
        # Execute callback if registered
        if std_error.error_code in self.error_callbacks:
            try:
                await self.error_callbacks[std_error.error_code](std_error, context)
            except Exception as callback_error:
                logging.error(f"Error in error callback: {callback_error}")
        
        return std_error
    
    def _convert_to_standard_error(self, error: Exception, context: Dict[str, Any]) -> LLMProxyError:
        """Convert any exception to standardized LLMProxyError"""
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # Map common exceptions to standardized errors
        if "connection" in error_message.lower() or "timeout" in error_message.lower():
            return OllamaConnectionError(
                message=error_message,
                ollama_url=context.get("ollama_url")
            )
        
        elif "memory" in error_message.lower() or error_type == "MemoryError":
            return MemoryError(
                message=error_message,
                requested_mb=context.get("requested_mb"),
                available_mb=context.get("available_mb")
            )
        
        elif "validation" in error_message.lower() or error_type == "ValidationError":
            return ValidationError(
                message=error_message,
                field=context.get("field")
            )
        
        elif "not found" in error_message.lower():
            return ModelNotFoundError(context.get("model_name", "unknown"))
        
        elif "rate limit" in error_message.lower():
            return RateLimitExceededError(
                limit=context.get("rate_limit", 60),
                retry_after=context.get("retry_after", 60)
            )
        
        elif "auth" in error_message.lower() or "forbidden" in error_message.lower():
            return AuthenticationError(error_message)
        
        else:
            # Generic internal error
            return LLMProxyError(
                message=error_message,
                error_code="INTERNAL_ERROR",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.MEDIUM,
                details={"original_type": error_type, "traceback": traceback.format_exc()},
                user_message="An internal error occurred. Please try again later."
            )
    
    def _track_error(self, error: LLMProxyError):
        """Track error occurrence for metrics"""
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
    
    def _log_error(self, error: LLMProxyError, context: Dict[str, Any], request_id: str = None):
        """Log error with appropriate level and context"""
        
        log_data = {
            "error_code": error.error_code,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "details": error.details,
            "context": context,
            "request_id": request_id,
            "timestamp": error.timestamp.isoformat()
        }
        
        log_message = f"[{request_id or 'NO_ID'}] {error.error_code}: {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logging.critical(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logging.error(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logging.warning(log_message, extra=log_data)
        else:
            logging.info(log_message, extra=log_data)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_counts": dict(self.error_counts),
            "top_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

# Decorator for automatic error handling
def handle_errors(
    context_func: Callable = None,
    reraise: bool = True,
    default_return: Any = None
):
    """Decorator for automatic error handling
    
    Args:
        context_func: Function to generate context dict from function args
        reraise: Whether to reraise the standardized error
        default_return: Default return value if not reraising
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {}
                if context_func:
                    try:
                        context = context_func(*args, **kwargs)
                    except Exception:
                        pass
                
                std_error = await error_handler.handle_error(e, context)
                
                if reraise:
                    raise std_error
                else:
                    return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {}
                if context_func:
                    try:
                        context = context_func(*args, **kwargs)
                    except Exception:
                        pass
                
                # For sync functions, we can't await, so handle synchronously
                std_error = error_handler._convert_to_standard_error(e, context)
                error_handler._track_error(std_error)
                error_handler._log_error(std_error, context)
                
                if reraise:
                    raise std_error
                else:
                    return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Async context manager for error handling
class ErrorContext:
    """Async context manager for error handling"""
    
    def __init__(self, context: Dict[str, Any] = None, request_id: str = None):
        self.context = context or {}
        self.request_id = request_id
        self.error: Optional[LLMProxyError] = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.error = await error_handler.handle_error(exc_val, self.context, self.request_id)
            # Don't suppress the exception - let it propagate as standardized error
            return False

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions
async def handle_error(error: Exception, context: Dict[str, Any] = None, request_id: str = None) -> LLMProxyError:
    """Convenience function for handling errors"""
    return await error_handler.handle_error(error, context, request_id)

def register_error_callback(error_code: str, callback: Callable):
    """Register error callback"""
    error_handler.register_callback(error_code, callback)

# Example usage context functions for common scenarios
def ollama_context(*args, **kwargs) -> Dict[str, Any]:
    """Generate context for Ollama-related functions"""
    return {
        "component": "ollama",
        "ollama_url": getattr(args[0], 'base_url', None) if args else None
    }

def model_context(*args, **kwargs) -> Dict[str, Any]:
    """Generate context for model-related functions"""
    return {
        "component": "model",
        "model_name": kwargs.get("model") or (args[1] if len(args) > 1 else None)
    }

def request_context(*args, **kwargs) -> Dict[str, Any]:
    """Generate context for request processing"""
    request = args[0] if args else None
    return {
        "component": "request_processing",
        "model": getattr(request, 'model', None),
        "stream": getattr(request, 'stream', None),
        "max_tokens": getattr(request, 'max_tokens', None)
    }
