# services/enhanced_ollama_client.py - Enhanced with Optimized Connection Pooling

import aiohttp
import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, Any, AsyncGenerator, Optional
from datetime import datetime

from services.ollama_client import OllamaClient as BaseOllamaClient
from utils.connection_pool import get_connection_pool, ConnectionPoolConfig
from utils.memory_manager import allocate_memory, deallocate_memory


class EnhancedOllamaClient(BaseOllamaClient):
    """Enhanced Ollama client with optimized connection pooling and performance features"""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        super().__init__(base_url, timeout)
        self.connection_pool = None
        self.performance_stats = {
            "total_requests": 0,
            "total_response_time": 0,
            "streaming_requests": 0,
            "cache_hits": 0,
            "errors": 0,
        }
        self._request_cache = {}

    async def initialize(self):
        if self._initialized:
            return

        if not allocate_memory("ollama_client", 50):
            logging.warning("Could not allocate memory for Ollama client optimization")

        pool_config = ConnectionPoolConfig(
            total_limit=50,
            per_host_limit=15,
            keepalive_timeout=120,
            connect_timeout=10,
            total_timeout=self.timeout,
            dns_cache_ttl=600,
        )

        self.connection_pool = get_connection_pool(pool_config)
        await self.connection_pool.initialize()

        self._initialized = True
        logging.info(
            "Enhanced Ollama client initialized with optimized connection pool"
        )

        try:
            await self.health_check()
            logging.info("✅ Initial Ollama connection successful")
        except Exception as e:
            logging.warning(f"⚠️ Initial Ollama connection failed: {e}")

    async def chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        request_key = self._generate_request_key(request_data)

        if request_key in self._request_cache:
            cache_entry = self._request_cache[request_key]
            if time.time() - cache_entry["timestamp"] < 5:
                self.performance_stats["cache_hits"] += 1
                logging.debug("Request deduplicated (identical request within 5s)")
                return cache_entry["response"]

        try:
            self.performance_stats["total_requests"] += 1

            response_data = await self.connection_pool.post_json(
                f"{self.base_url}/api/chat", request_data
            )

            self._request_cache[request_key] = {
                "response": response_data,
                "timestamp": time.time(),
            }

            if len(self._request_cache) > 100:
                oldest_key = min(
                    self._request_cache.keys(),
                    key=lambda k: self._request_cache[k]["timestamp"],
                )
                del self._request_cache[oldest_key]

            response_time = time.time() - start_time
            self.performance_stats["total_response_time"] += response_time

            logging.debug(f"Chat completion successful in {response_time:.3f}s")
            return response_data

        except Exception as e:
            self.performance_stats["errors"] += 1
            logging.error(f"Chat completion failed: {str(e)}")
            raise

    async def stream_chat(
        self, request_data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self.performance_stats["streaming_requests"] += 1

        try:
            request_data = request_data.copy()
            request_data["stream"] = True

            async with await self.connection_pool.request(
                "POST", f"{self.base_url}/api/chat", json=request_data
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Ollama streaming error {response.status}: {error_text}"
                    )

                async for line in response.content:
                    if line:
                        try:
                            line_text = line.decode().strip()
                            if line_text:
                                chunk_data = json.loads(line_text)
                                yield chunk_data

                                if chunk_data.get("done", False):
                                    break

                        except json.JSONDecodeError as e:
                            logging.warning(
                                f"Failed to parse streaming line: {line_text}, error: {e}"
                            )
                            continue
                        except Exception as e:
                            logging.error(f"Streaming processing error: {e}")
                            break

            response_time = time.time() - start_time
            self.performance_stats["total_response_time"] += response_time
            logging.debug(f"Streaming completed in {response_time:.3f}s")

        except Exception as e:
            self.performance_stats["errors"] += 1
            logging.error(f"Error in streaming: {str(e)}")
            raise

    async def health_check(self) -> bool:
        if not self._initialized:
            await self.initialize()

        try:
            await self.connection_pool.post_json(f"{self.base_url}/api/tags", {})
            return True
        except Exception as e:
            logging.error(f"Health check failed: {str(e)}")
            return False

    async def health_check_detailed(self) -> Dict[str, Any]:
        basic_health = await self.health_check()

        health_data = {
            "status": "healthy" if basic_health else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "connection_pool": {},
            "performance": self.get_performance_stats(),
            "optimization_active": True,
        }

        if self.connection_pool:
            try:
                pool_health = await self.connection_pool.health_check()
                health_data["connection_pool"] = pool_health
            except Exception as e:
                health_data["connection_pool"] = {"error": str(e)}

        return health_data

    def get_performance_stats(self) -> Dict[str, Any]:
        total_requests = self.performance_stats["total_requests"]

        stats = {
            "total_requests": total_requests,
            "streaming_requests": self.performance_stats["streaming_requests"],
            "cache_hits": self.performance_stats["cache_hits"],
            "errors": self.performance_stats["errors"],
            "error_rate": (self.performance_stats["errors"] / max(1, total_requests))
            * 100,
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / max(1, total_requests)
            )
            * 100,
            "avg_response_time": (
                self.performance_stats["total_response_time"] / total_requests
                if total_requests > 0
                else 0
            ),
        }

        return stats

    def _generate_request_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a stable cache key for deduplication using SHA256"""
        key_data = {
            "model": request_data.get("model"),
            # Only last message
            "messages": request_data.get("messages", [])[-1:],
            "temperature": request_data.get("options", {}).get("temperature", 0.7),
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def warm_up_model(self, model: str) -> bool:
        try:
            warmup_request = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "options": {"num_predict": 1, "temperature": 0.1},
            }

            response = await self.chat_completion(warmup_request)
            logging.info(f"Model {model} warmed up successfully")
            return response is not None

        except Exception as e:
            logging.error(f"Failed to warm up model {model}: {str(e)}")
            return False

    async def cleanup(self):
        if self.connection_pool:
            pass
        deallocate_memory("ollama_client")
        self._initialized = False
        logging.info("Enhanced Ollama client cleaned up")


def create_enhanced_ollama_client(
    base_url: str = "http://localhost:11434", timeout: int = 300
) -> EnhancedOllamaClient:
    return EnhancedOllamaClient(base_url, timeout)
