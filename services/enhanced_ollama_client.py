# services/enhanced_ollama_client.py - Enhanced Ollama Client with Streaming (Fixed)
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, AsyncGenerator
from services.ollama_client import OllamaClient as BaseOllamaClient

class EnhancedOllamaClient(BaseOllamaClient):
    
    async def stream_chat(self, request_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion from Ollama with proper error handling"""
        
        if not self.session:
            await self.initialize()
        
        try:
            # Ensure streaming is enabled in request
            request_data = request_data.copy()
            request_data["stream"] = True
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama streaming error {response.status}: {error_text}")
                
                # Read streaming response line by line
                async for line in response.content:
                    if line:
                        try:
                            line_text = line.decode().strip()
                            if line_text:  # Skip empty lines
                                chunk_data = json.loads(line_text)
                                yield chunk_data
                                
                                # Break if done
                                if chunk_data.get("done", False):
                                    break
                                    
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse streaming line: {line_text}, error: {e}")
                            continue  # Skip malformed lines
                            
        except asyncio.TimeoutError:
            logging.error("Ollama streaming request timed out")
            raise Exception("Streaming request timed out")
        except Exception as e:
            logging.error(f"Error in Ollama streaming: {str(e)}")
            raise
    
    async def chat_completion_with_retry(
        self, 
        request_data: Dict[str, Any], 
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Chat completion with retry logic and better error handling"""
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await self.chat_completion(request_data)
            except Exception as e:
                last_exception = e
                logging.warning(f"Chat completion attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logging.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"All {max_retries} attempts failed")
        
        # Re-raise the last exception
        raise last_exception
    
    async def health_check_detailed(self) -> Dict[str, Any]:
        """Detailed health check with model information"""
        
        try:
            if not self.session:
                await self.initialize()
            
            # Check basic connectivity
            basic_health = await self.health_check()
            if not basic_health:
                return {
                    "status": "unhealthy",
                    "error": "Basic connectivity check failed"
                }
            
            # Get model list to verify Ollama is responding properly
            models = await self.list_models()
            
            # Get system info if available
            system_info = await self._get_system_info()
            
            return {
                "status": "healthy",
                "models_available": len(models),
                "models": [model.get("name", "unknown") for model in models[:5]],  # First 5 models
                "system_info": system_info
            }
            
        except Exception as e:
            logging.error(f"Detailed health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get Ollama system information if available"""
        
        try:
            # This endpoint may not exist in all Ollama versions
            async with self.session.get(f"{self.base_url}/api/ps") as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            # Ignore errors for optional system info
            pass
        
        return {"info": "System info not available"}
    
    async def warm_up_model(self, model: str) -> bool:
        """Warm up a model with a minimal request"""
        
        try:
            warmup_request = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "options": {
                    "num_predict": 1,  # Minimal response
                    "temperature": 0.1
                }
            }
            
            response = await self.chat_completion(warmup_request)
            return response is not None
            
        except Exception as e:
            logging.error(f"Failed to warm up model {model}: {str(e)}")
            return False
