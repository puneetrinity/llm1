# models/__init__.py
from .requests import (
    Message,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    TranscriptionRequest,
    validate_chat_request,
    validate_completion_request,
)

from .responses import (
    Choice,
    Usage,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionResponse,
    ModelResponse,
    ModelListResponse,
    ErrorResponse,
    EmbeddingResponse,
    ImageGenerationResponse,
    format_chat_completion_response,
    format_streaming_chunk,
    format_completion_response,
    format_error_response,
)

__all__ = [
    # Request models
    "Message",
    "ChatCompletionRequest",
    "CompletionRequest",
    "EmbeddingRequest",
    "ImageGenerationRequest",
    "TranscriptionRequest",
    # Response models
    "Choice",
    "Usage",
    "ChatCompletionResponse",
    "ChatCompletionStreamResponse",
    "CompletionResponse",
    "ModelResponse",
    "ModelListResponse",
    "ErrorResponse",
    "EmbeddingResponse",
    "ImageGenerationResponse",
    # Helper functions
    "validate_chat_request",
    "validate_completion_request",
    "format_chat_completion_response",
    "format_streaming_chunk",
    "format_completion_response",
    "format_error_response",
]
