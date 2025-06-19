# models/requests.py
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    """Chat message"""

    role: str
    content: str
    name: Optional[str] = None

    @validator("role")
    def validate_role(cls, v):
        allowed_roles = ["system", "user", "assistant", "function"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""

    model: str = Field(default="gpt-3.5-turbo")
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None

    class Config:
        extra = "allow"  # Allow additional fields for forward compatibility


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request (legacy)"""

    model: str = Field(default="text-davinci-003")
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    stream: Optional[bool] = Field(default=False)
    logprobs: Optional[int] = None
    echo: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(default=1, ge=1)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


class EmbeddingRequest(BaseModel):
    """Embedding request (for future implementation)"""

    model: str = Field(default="text-embedding-ada-002")
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[str] = Field(default="float")
    user: Optional[str] = None

    @validator("encoding_format")
    def validate_encoding_format(cls, v):
        allowed_formats = ["float", "base64"]
        if v not in allowed_formats:
            raise ValueError(f"Encoding format must be one of {allowed_formats}")
        return v


class ImageGenerationRequest(BaseModel):
    """Image generation request (for future implementation)"""

    prompt: str
    model: Optional[str] = Field(default="dall-e-2")
    n: Optional[int] = Field(default=1, ge=1, le=10)
    quality: Optional[str] = Field(default="standard")
    response_format: Optional[str] = Field(default="url")
    size: Optional[str] = Field(default="1024x1024")
    style: Optional[str] = Field(default="vivid")
    user: Optional[str] = None

    @validator("quality")
    def validate_quality(cls, v):
        allowed_qualities = ["standard", "hd"]
        if v not in allowed_qualities:
            raise ValueError(f"Quality must be one of {allowed_qualities}")
        return v

    @validator("response_format")
    def validate_response_format(cls, v):
        allowed_formats = ["url", "b64_json"]
        if v not in allowed_formats:
            raise ValueError(f"Response format must be one of {allowed_formats}")
        return v

    @validator("size")
    def validate_size(cls, v):
        allowed_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
        if v not in allowed_sizes:
            raise ValueError(f"Size must be one of {allowed_sizes}")
        return v


class TranscriptionRequest(BaseModel):
    """Audio transcription request (for future implementation)"""

    file: bytes
    model: str = Field(default="whisper-1")
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = Field(default="json")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

    @validator("response_format")
    def validate_response_format(cls, v):
        allowed_formats = ["json", "text", "srt", "verbose_json", "vtt"]
        if v not in allowed_formats:
            raise ValueError(f"Response format must be one of {allowed_formats}")
        return v


# Helper functions for request validation
def validate_chat_request(request: ChatCompletionRequest) -> None:
    """Validate chat completion request"""
    if not request.messages:
        raise ValueError("Messages cannot be empty")

    if request.n and request.n > 1 and request.stream:
        raise ValueError("Cannot use n > 1 with streaming")

    # Check for at least one user message
    has_user_message = any(msg.role == "user" for msg in request.messages)
    if not has_user_message:
        raise ValueError("At least one user message is required")


def validate_completion_request(request: CompletionRequest) -> None:
    """Validate completion request"""
    if isinstance(request.prompt, list) and len(request.prompt) == 0:
        raise ValueError("Prompt cannot be empty")

    if isinstance(request.prompt, str) and not request.prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if request.best_of and request.best_of > request.n:
        raise ValueError("best_of must be less than or equal to n")
