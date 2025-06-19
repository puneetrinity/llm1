# models/responses.py
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import time


class Choice(BaseModel):
    """Single choice in a completion response"""

    index: int
    message: Optional[Dict[str, str]] = None
    text: Optional[str] = None
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None
    delta: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    """Token usage information"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    system_fingerprint: Optional[str] = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible text completion response (legacy)"""

    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time() * 1000)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class ModelResponse(BaseModel):
    """Single model information"""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ollama"
    permission: Optional[List[Dict[str, Any]]] = None
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of available models"""

    object: str = "list"
    data: List[ModelResponse]


class ErrorResponse(BaseModel):
    """Error response format"""

    error: Dict[str, Any]


class EmbeddingResponse(BaseModel):
    """Embedding response (for future implementation)"""

    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Usage


class ImageGenerationResponse(BaseModel):
    """Image generation response (for future implementation)"""

    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[Dict[str, str]]


# Compatibility: Some routers expect ChatCompletionChoice
class ChatCompletionChoice(Choice):
    pass


# Helper functions for response formatting
def format_chat_completion_response(
    content: str,
    model: str,
    finish_reason: str = "stop",
    usage: Optional[Dict[str, int]] = None,
) -> ChatCompletionResponse:
    """Format a chat completion response"""
    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                index=0,
                message={"role": "assistant", "content": content},
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(**usage) if usage else None,
    )


def format_streaming_chunk(
    content: str,
    model: str,
    finish_reason: Optional[str] = None,
    is_first: bool = False,
) -> str:
    """Format a streaming chunk for SSE"""
    chunk = ChatCompletionStreamResponse(
        model=model,
        choices=[
            Choice(
                index=0,
                delta={"role": "assistant" if is_first else None, "content": content},
                finish_reason=finish_reason,
            )
        ],
    )

    if finish_reason:
        return f"data: {chunk.model_dump_json()}\n\ndata: [DONE]\n\n"
    else:
        return f"data: {chunk.model_dump_json()}\n\n"


def format_completion_response(
    text: str,
    model: str,
    finish_reason: str = "stop",
    usage: Optional[Dict[str, int]] = None,
) -> CompletionResponse:
    """Format a legacy completion response"""
    return CompletionResponse(
        model=model,
        choices=[Choice(index=0, text=text, finish_reason=finish_reason)],
        usage=Usage(**usage) if usage else None,
    )


def format_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> ErrorResponse:
    """Format an error response"""
    error_dict = {
        "message": message,
        "type": error_type,
    }

    if param:
        error_dict["param"] = param
    if code:
        error_dict["code"] = code

    return ErrorResponse(error=error_dict)
