# services/ollama_client.py - Base Ollama Client
import aiohttp
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the HTTP session"""
        if self._initialized:
            return

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self._initialized = True

        # Test connection
        try:
            await self.health_check()
            logging.info(f"Successfully connected to Ollama at {self.base_url}")
        except Exception as e:
            logging.warning(f"Could not connect to Ollama: {str(e)}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self._initialized = False

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logging.error(f"Ollama health check failed: {str(e)}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    logging.error(f"Failed to list models: {response.status}")
                    return []
        except Exception as e:
            logging.error(f"Error listing models: {str(e)}")
            return []

    async def pull_model(self, model: str) -> bool:
        """Pull a model to make it available"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.post(
                f"{self.base_url}/api/pull", json={"name": model}
            ) as response:
                if response.status == 200:
                    # Consume the streaming response
                    async for line in response.content:
                        if line:
                            try:
                                status = json.loads(line.decode())
                                if status.get("status") == "success":
                                    logging.info(f"Successfully pulled model: {model}")
                                    return True
                            except json.JSONDecodeError:
                                continue
                return False
        except Exception as e:
            logging.error(f"Error pulling model {model}: {str(e)}")
            return False

    async def chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send chat completion request to Ollama"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.post(
                f"{self.base_url}/api/chat", json=request_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except Exception as e:
            logging.error(f"Error in chat completion: {str(e)}")
            raise

    async def generate_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send completion request to Ollama"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.post(
                f"{self.base_url}/api/generate", json=request_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except Exception as e:
            logging.error(f"Error in completion: {str(e)}")
            raise

    async def model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.post(
                f"{self.base_url}/api/show", json={"name": model}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.warning(
                        f"Could not get info for model {model}: {response.status}"
                    )
                    return None
        except Exception as e:
            logging.error(f"Error getting model info for {model}: {str(e)}")
            return None

    async def delete_model(self, model: str) -> bool:
        """Delete a model from Ollama"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.delete(
                f"{self.base_url}/api/delete", json={"name": model}
            ) as response:
                return response.status == 200
        except Exception as e:
            logging.error(f"Error deleting model {model}: {str(e)}")
            return False
