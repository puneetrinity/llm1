# FILE 1: utils/exceptions.py - NEW FILE
# Add this file to handle all your errors properly

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException

class ErrorCategory(Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    EXTERNAL_SERVICE = "external_service" 
    RESOURCE = "resource"
    INTERNAL = "internal"

class LLMProxyError(Exception):
    """Standardized error with proper HTTP response"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        status_code: int = 500,
        user_message: str = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.status_code = status_code
        self.user_message = user_message or message
        self.timestamp = datetime.utcnow()
    
    def to_http_exception(self) -> HTTPException:
        return HTTPException(
            status_code=self.status_code,
            detail={
                "error": {
                    "code": self.error_code,
                    "message": self.user_message,
                    "category": self.category.value,
                    "timestamp": self.timestamp.isoformat() + "Z"
                }
            }
        )

# Specific errors your app needs
class ModelNotAvailableError(LLMProxyError):
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' is not available",
            error_code="MODEL_NOT_AVAILABLE",
            category=ErrorCategory.RESOURCE,
            status_code=400,
            user_message=f"The model '{model_name}' is not available. Please check the model name."
        )

class OllamaConnectionError(LLMProxyError):
    def __init__(self, ollama_url: str, reason: str):
        super().__init__(
            message=f"Cannot connect to Ollama at {ollama_url}: {reason}",
            error_code="OLLAMA_CONNECTION_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            status_code=503,
            user_message="AI service is temporarily unavailable. Please try again in a few moments."
        )

class InvalidRequestError(LLMProxyError):
    def __init__(self, field: str, issue: str):
        super().__init__(
            message=f"Invalid request: {field} - {issue}",
            error_code="INVALID_REQUEST",
            category=ErrorCategory.VALIDATION,
            status_code=400,
            user_message=f"Invalid request: {issue}"
        )

# FILE 2: utils/error_decorator.py - NEW FILE  
# Add this file to automatically handle errors in your endpoints

import logging
import traceback
from functools import wraps
from .exceptions import LLMProxyError, OllamaConnectionError, ModelNotAvailableError, ErrorCategory

def handle_errors(func):
    """Decorator to automatically handle errors in your endpoints"""
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
            
        except LLMProxyError as e:
            # Already standardized - just re-raise as HTTP
            logging.warning(f"LLM Proxy Error: {e.error_code} - {e.message}")
            raise e.to_http_exception()
            
        except Exception as e:
            error_msg = str(e)
            
            # Convert common errors to standardized ones
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                std_error = OllamaConnectionError("localhost:11434", error_msg)
            elif "not found" in error_msg.lower():
                std_error = ModelNotAvailableError("unknown")
            else:
                # Generic internal error
                std_error = LLMProxyError(
                    message=error_msg,
                    error_code="INTERNAL_ERROR", 
                    category=ErrorCategory.INTERNAL,
                    status_code=500,
                    user_message="An internal error occurred. Please try again later."
                )
            
            # Log with full traceback
            logging.error(f"Unhandled error: {error_msg}")
            logging.error(traceback.format_exc())
            
            raise std_error.to_http_exception()
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Same logic for sync functions
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Sync error: {str(e)}")
            raise
    
    # Return appropriate wrapper
    return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper

# NOW UPDATE YOUR EXISTING main.py - ADD THESE IMPORTS AND HANDLERS

# At the top of main.py, add:
from utils.exceptions import LLMProxyError
from utils.error_decorator import handle_errors
from fastapi.exceptions import RequestValidationError

# Add these error handlers to your existing FastAPI app:

@app.exception_handler(LLMProxyError) 
async def llm_proxy_error_handler(request, exc: LLMProxyError):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_http_exception().detail
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {}
    field = " -> ".join(str(loc) for loc in first_error.get('loc', []))
    message = first_error.get('msg', 'Validation error')
    
    from utils.exceptions import InvalidRequestError
    error = InvalidRequestError(field, message)
    
    return JSONResponse(
        status_code=error.status_code,
        content=error.to_http_exception().detail
    )

# Add the decorator to your main endpoints:

@app.post("/v1/chat/completions")
@handle_errors  # <-- ADD THIS LINE
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    # Your existing code - errors will be handled automatically
    pass

@app.post("/v1/completions") 
@handle_errors  # <-- ADD THIS LINE
async def completions(request: CompletionRequest, http_request: Request):
    # Your existing code - errors will be handled automatically
    pass
