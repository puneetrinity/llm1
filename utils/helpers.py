# utils/helpers.py
import json
import time
import asyncio
from typing import Dict, List, Any, AsyncGenerator, Optional
from datetime import datetime
import logging

from models.responses import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionResponse,
    Choice,
    Usage,
    format_streaming_chunk,
)

logger = logging.getLogger(__name__)


def format_openai_response(
    ollama_response: Dict[str, Any], model: str, is_chat: bool = True
) -> Dict[str, Any]:
    """Convert Ollama response to OpenAI format"""

    # Extract content
    content = ollama_response.get("response", "") or ollama_response.get(
        "message", {}
    ).get("content", "")

    # Create usage info
    usage = Usage(
        prompt_tokens=ollama_response.get("prompt_eval_count", 0),
        completion_tokens=ollama_response.get("eval_count", 0),
        total_tokens=(
            ollama_response.get("prompt_eval_count", 0)
            + ollama_response.get("eval_count", 0)
        ),
    )

    if is_chat:
        # Chat completion format
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message={"role": "assistant", "content": content},
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
    else:
        # Text completion format
        response = CompletionResponse(
            id=f"cmpl-{int(time.time() * 1000)}",
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=[Choice(index=0, text=content, finish_reason="stop")],
            usage=usage,
        )

    return response.model_dump()


async def handle_streaming_response(
    ollama_client,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Handle streaming responses from Ollama"""

    try:
        # Start streaming
        is_first_chunk = True

        async for chunk in ollama_client.generate_completion_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            # Extract content from chunk
            content = chunk.get("response", "")

            if content:
                # Format as SSE chunk
                sse_chunk = format_streaming_chunk(
                    content=content, model=model, is_first=is_first_chunk
                )
                is_first_chunk = False
                yield sse_chunk

            # Check if done
            if chunk.get("done", False):
                # Send final chunk
                final_chunk = format_streaming_chunk(
                    content="", model=model, finish_reason="stop"
                )
                yield final_chunk
                break

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        # Send error in stream format
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "streaming_error",
                "code": "stream_failed",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


def convert_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages to a single prompt string"""
    prompt_parts = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    # Add final assistant prompt
    prompt = "\n\n".join(prompt_parts)
    if messages and messages[-1].get("role") != "assistant":
        prompt += "\n\nAssistant:"

    return prompt


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approximation)"""
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_messages(
    messages: List[Dict[str, str]], max_tokens: int = 4096
) -> List[Dict[str, str]]:
    """Truncate messages to fit within token limit"""
    total_tokens = 0
    truncated_messages = []

    # Process messages in reverse order (keep most recent)
    for message in reversed(messages):
        content = message.get("content", "")
        message_tokens = estimate_tokens(content)

        if total_tokens + message_tokens > max_tokens:
            # Truncate this message if needed
            if total_tokens < max_tokens:
                remaining_tokens = max_tokens - total_tokens
                remaining_chars = remaining_tokens * 4  # Rough conversion
                truncated_content = content[:remaining_chars] + "..."
                truncated_messages.insert(0, {**message, "content": truncated_content})
            break

        total_tokens += message_tokens
        truncated_messages.insert(0, message)

    return truncated_messages


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for compatibility"""
    # Map OpenAI model names to Ollama models
    model_mapping = {
        "gpt-3.5-turbo": "mistral:7b-instruct-q4_0",
        "gpt-3.5-turbo-16k": "mistral:7b-instruct-q4_0",
        "gpt-4": "llama3:8b-instruct-q4_0",
        "gpt-4-32k": "llama3:8b-instruct-q4_0",
        "text-davinci-003": "mistral:7b-instruct-q4_0",
        "text-curie-001": "gemma:7b-instruct",
        "text-babbage-001": "phi3.5",
        "text-ada-001": "phi3.5",
        "claude-2": "llama3:8b-instruct-q4_0",
        "claude-instant": "mistral:7b-instruct-q4_0",
    }

    return model_mapping.get(model, model)


def calculate_cost(usage: Dict[str, int], model: str) -> float:
    """Calculate approximate cost based on usage"""
    # Rough cost estimates per 1M tokens (in USD)
    cost_per_million = {
        "phi3.5": 0.0,  # Local model
        "mistral:7b-instruct-q4_0": 0.0,
        "gemma:7b-instruct": 0.0,
        "llama3:8b-instruct-q4_0": 0.0,
    }

    # Local models have no cost
    return 0.0


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime as ISO timestamp"""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON from text"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(text[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                pass

    return None


def get_client_ip(request) -> str:
    """Extract client IP from request"""
    # Check for forwarded IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Check for real IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to client host
    if request.client:
        return request.client.host

    return "unknown"


# Export all helpers
__all__ = [
    "format_openai_response",
    "handle_streaming_response",
    "convert_messages_to_prompt",
    "estimate_tokens",
    "truncate_messages",
    "sanitize_model_name",
    "calculate_cost",
    "format_timestamp",
    "parse_json_safely",
    "get_client_ip",
]
