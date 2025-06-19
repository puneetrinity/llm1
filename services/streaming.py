# services/streaming.py - Streaming Support Implementation
import asyncio
import json
import uuid
from typing import AsyncGenerator, Dict, Any
from fastapi import Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import logging


class StreamingService:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client

    async def stream_chat_completion(
        self, request_data: Dict[str, Any], model: str
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion responses"""

        try:
            # Prepare streaming request to Ollama
            ollama_request = {
                "model": model,
                "messages": request_data["messages"],
                "stream": True,
                "options": {
                    "temperature": request_data.get("temperature", 0.7),
                    "top_p": request_data.get("top_p", 1.0),
                    "num_predict": request_data.get("max_tokens", -1),
                },
            }

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_timestamp = int(asyncio.get_event_loop().time())

            # Send initial response
            initial_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }

            yield f"data: {json.dumps(initial_chunk)}\n\n"

            # Stream from Ollama
            async for chunk in self.ollama_client.stream_chat(ollama_request):
                if chunk:
                    # Parse Ollama response
                    content = chunk.get("message", {}).get("content", "")
                    done = chunk.get("done", False)

                    # Format as OpenAI-compatible chunk
                    response_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content} if content else {},
                                "finish_reason": "stop" if done else None,
                            }
                        ],
                    }

                    yield f"data: {json.dumps(response_chunk)}\n\n"

                    if done:
                        break

            # Send final chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logging.error(f"Error in streaming: {str(e)}")

            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
