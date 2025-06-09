# Enhanced Ollama client with streaming support
# services/enhanced_ollama_client.py - Enhanced Ollama Client with Streaming
from services.ollama_client import OllamaClient as BaseOllamaClient
import aiohttp
import asyncio
import json

class EnhancedOllamaClient(BaseOllamaClient):
    
    async def stream_chat(self, request_data: Dict[str, Any]):
        """Stream chat completion from Ollama"""
        
        if not self.session:
            await self.initialize()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=request_data
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"Ollama streaming error: {response.status}")
                
                # Read streaming response line by line
                async for line in response.content:
                    if line:
                        try:
                            chunk_data = json.loads(line.decode().strip())
                            yield chunk_data
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
                            
        except Exception as e:
            logging.error(f"Error in Ollama streaming: {str(e)}")
            raise
