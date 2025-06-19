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
        user_message: str = None,
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
                    "timestamp": self.timestamp.isoformat() + "Z",
                }
            },
        )


# Specific errors your app needs


class ModelNotAvailableError(LLMProxyError):
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' is not available",
            error_code="MODEL_NOT_AVAILABLE",
            category=ErrorCategory.RESOURCE,
            status_code=400,
            user_message=f"The model '{model_name}' is not available. Please check the model name.",
        )


class OllamaConnectionError(LLMProxyError):
    def __init__(self, ollama_url: str, reason: str):
        super().__init__(
            message=f"Cannot connect to Ollama at {ollama_url}: {reason}",
            error_code="OLLAMA_CONNECTION_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            status_code=503,
            user_message="AI service is temporarily unavailable. Please try again in a few moments.",
        )


class InvalidRequestError(LLMProxyError):
    def __init__(self, field: str, issue: str):
        super().__init__(
            message=f"Invalid request: {field} - {issue}",
            error_code="INVALID_REQUEST",
            category=ErrorCategory.VALIDATION,
            status_code=400,
            user_message=f"Invalid request: {issue}",
        )
