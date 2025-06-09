# models/responses.py - Response Data Models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Usage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")

class ChatCompletionChoice(BaseModel):
    index: int = Field(..., description="Index of the choice")
    message: Dict[str, Any] = Field(..., description="The generated message")
    finish_reason: Optional[str] = Field(None, description="Reason the generation finished")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for the completion")
    choices: List[ChatCompletionChoice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")
    
    # Enhanced metadata
    cache_hit: Optional[bool] = Field(False, description="Whether response came from cache")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    selected_model: Optional[str] = Field(None, description="Actually selected model (for routing)")
    routing_reason: Optional[str] = Field(None, description="Reason for model selection")

class CompletionChoice(BaseModel):
    text: str = Field(..., description="The generated text")
    index: int = Field(..., description="Index of the choice")
    finish_reason: Optional[str] = Field(None, description="Reason the generation finished")

class CompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for the completion")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")

class ModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    created: Optional[int] = Field(None, description="Creation timestamp")
    owned_by: str = Field("ollama", description="Model owner")
    
    # Enhanced model metadata
    size_gb: Optional[float] = Field(None, description="Model size in GB")
    parameter_count: Optional[str] = Field(None, description="Number of parameters")
    quantization: Optional[str] = Field(None, description="Quantization level")
    context_length: Optional[int] = Field(None, description="Maximum context length")
    capabilities: Optional[List[str]] = Field(None, description="Model capabilities")
    cost_per_token: Optional[float] = Field(None, description="Estimated cost per token")

class ModelsResponse(BaseModel):
    object: str = Field("list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of available models")

class HealthStatus(BaseModel):
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")

class ServiceStatus(BaseModel):
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status (healthy, unhealthy, unknown)")
    last_check: datetime = Field(..., description="Last health check time")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional status details")

class HealthResponse(BaseModel):
    healthy: bool = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    
    # Service statuses
    services: List[ServiceStatus] = Field(default_factory=list, description="Individual service statuses")
    
    # System metrics
    system: Optional[Dict[str, Any]] = Field(None, description="System resource metrics")
    
    # Model availability
    models: Optional[Dict[str, Any]] = Field(None, description="Available models status")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
