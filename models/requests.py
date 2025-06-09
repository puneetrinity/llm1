# models/requests.py - Request Data Models
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import json

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Role must be system, user, or assistant')
        return v

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    user: Optional[str] = Field(None, description="User identifier")
    
    # Enhanced routing hints
    intent: Optional[str] = Field(None, description="Explicit intent for routing")
    priority: Optional[str] = Field("normal", description="Request priority (low, normal, high)")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError('At least one message is required')
        return v
    
    @validator('model')
    def validate_model(cls, v):
        if not v or not v.strip():
            raise ValueError('Model name cannot be empty')
        return v.strip()

class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    prompt: str = Field(..., description="The prompt to complete")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    echo: Optional[bool] = Field(False, description="Echo the prompt in the response")
    user: Optional[str] = Field(None, description="User identifier")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v

class ModelRequest(BaseModel):
    model: str = Field(..., description="Model name to operate on")
    action: Optional[str] = Field("load", description="Action to perform (load, unload, info)")

class WarmupRequest(BaseModel):
    model: str = Field(..., description="Model to warmup")
    force: Optional[bool] = Field(False, description="Force warmup even if recently used")
