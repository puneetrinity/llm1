# middleware/caching.py - Smart Caching Middleware with Circuit Breaker Protection
import json
import time
import logging
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from services.smart_cache import get_smart_cache, CacheConfig
from utils.error_handler import handle_error_with_circuit_breaker, cache_context
from utils.memory_manager import check_memory_allocation


class SmartCachingMiddleware(BaseHTTPMiddleware):
    """Intelligent caching middleware with semantic similarity and circuit breaker protection"""

    def __init__(
        self,
        app,
        cache_config: CacheConfig = None,
        enable_cache: bool = True,
        cacheable_paths: list = None,
        cache_ttl: int = 3600
    ):
        super().__init__(app)
        self.cache_config = cache_config or CacheConfig()
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.smart_cache = None

        # Default cacheable paths (can be customized)
        self.cacheable_paths = cacheable_paths or [
            "/v1/chat/completions",
            "/v1/completions"
        ]

        # Paths that should never be cached
        self.non_cacheable_paths = {
            "/health",
            "/metrics",
            "/admin",
            "/docs",
            "/openapi.json"
        }

        # Request methods that can be cached
        self.cacheable_methods = {"POST"}  # Mainly for LLM requests

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_sets': 0,
            'cache_errors': 0,
            'total_requests': 0,
            'bytes_saved': 0
        }

    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch with smart caching logic"""

        # Initialize cache on first request
        if self.smart_cache is None and self.enable_cache:
            await self._initialize_cache()

        self.stats['total_requests'] += 1

        # Check if request should be cached
        if not self._should_cache_request(request):
            return await call_next(request)

        # Try to get cached response
        cached_response = await self._get_cached_response(request)
        if cached_response:
            self.stats['cache_hits'] += 1
            return cached_response

        # Process request and cache response
        response = await call_next(request)

        # Cache successful responses
        if self._should_cache_response(response):
            await self._cache_response(request, response)

        return response

    async def invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern"""
        if self.smart_cache:
            try:
                logging.info(f"Cache invalidation requested for pattern: {pattern}")
                # Implementation depends on the backend (e.g., Redis pattern delete)
            except Exception as e:
                logging.error(f"Error invalidating cache pattern {pattern}: {e}")

    async def _initialize_cache(self):
        """Initialize smart cache with circuit breaker protection"""
        try:
            if not check_memory_allocation('cache', 100):  # 100MB for caching
                logging.warning("Insufficient memory for caching - disabling cache")
                self.enable_cache = False
                return

            self.smart_cache = await handle_error_with_circuit_breaker(
                'cache',
                self._init_cache_backend,
                context={'action': 'initialize_cache'}
            )

            logging.info("Smart caching middleware initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize cache: {e}")
            self.enable_cache = False

    async def _init_cache_backend(self):
        """Initialize cache backend"""
        from services.smart_cache import initialize_smart_cache
        return await initialize_smart_cache(self.cache_config)

    def _should_cache_request(self, request: Request) -> bool:
        """Determine if request should be cached"""
        if not self.enable_cache or not self.smart_cache:
            return False
        if request.method not in self.cacheable_methods:
            return False

        path = request.url.path

        if any(path.startswith(excluded) for excluded in self.non_cacheable_paths):
            return False

        if not any(path.startswith(cacheable) for cacheable in self.cacheable_paths):
            return False

        if self._is_streaming_request(request):
            return False

        return True

    def _is_streaming_request(self, request: Request) -> bool:
        """Check if request is for streaming response"""
        return request.headers.get("accept") == "text/event-stream"

    async def _get_cached_response(self, request: Request) -> Optional[Response]:
        """Get cached response if available"""
        try:
            cache_key, query_text = await self._generate_cache_key(request)

            cached_data = await handle_error_with_circuit_breaker(
                'cache',
                self.smart_cache.get,
                cache_key,
                query_text,
                context=cache_context(cache_key=cache_key)
            )

            if cached_data:
                response = self._reconstruct_response(cached_data)
                response.headers["X-Cache"] = "HIT"
                response.headers["X-Cache-Key"] = cache_key[:16] + "..."
                self.stats['bytes_saved'] += len(json.dumps(cached_data).encode())
                logging.debug(f"Cache hit for key: {cache_key[:16]}...")
                return response

        except Exception as e:
            logging.error(f"Cache get error: {e}")
            self.stats['cache_errors'] += 1

        self.stats['cache_misses'] += 1
        return None

    async def _cache_response(self, request: Request, response: Response):
        """Cache response with circuit breaker protection"""
        try:
            response_data = await self._extract_response_data(response)
            if not response_data:
                return

            cache_key, query_text = await self._generate_cache_key(request)
            ttl = self._calculate_ttl(request, response_data)

            await handle_error_with_circuit_breaker(
                'cache',
                self.smart_cache.set,
                cache_key,
                response_data,
                ttl,
                query_text,
                context=cache_context(cache_key=cache_key)
            )

            self.stats['cache_sets'] += 1
            logging.debug(f"Cached response for key: {cache_key[:16]}... (TTL: {ttl}s)")

        except Exception as e:
            logging.error(f"Cache set error: {e}")
            self.stats['cache_errors'] += 1

    async def _generate_cache_key(self, request: Request) -> tuple:
        """Generate cache key and extract query text from request"""
        try:
            body = await request.body()
            if not body:
                return None, None

            request_data = json.loads(body.decode())
            cache_key = self.smart_cache.generate_cache_key(request_data)
            query_text = self.smart_cache.extract_query_text(request_data)
            return cache_key, query_text

        except Exception as e:
            logging.error(f"Error generating cache key: {e}")
            return None, None

    def _should_cache_response(self, response: Response) -> bool:
        """Determine if response should be cached"""
        if response.status_code != 200:
            return False
        if isinstance(response, StreamingResponse):
            return False
        if response.headers.get("Cache-Control") == "no-cache":
            return False
        return True

    async def _extract_response_data(self, response: Response) -> Optional[Dict[str, Any]]:
        """Extract data from response for caching"""
        try:
            if isinstance(response, JSONResponse):
                return {
                    'type': 'json',
                    'data': response.body.decode() if hasattr(response, 'body') else None,
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'timestamp': time.time()
                }
            return None
        except Exception as e:
            logging.error(f"Error extracting response data: {e}")
            return None

    def _reconstruct_response(self, cached_data: Dict[str, Any]) -> Response:
        """Reconstruct response from cached data"""
        try:
            if cached_data['type'] == 'json':
                response = JSONResponse(
                    content=json.loads(cached_data['data']) if cached_data['data'] else {},
                    status_code=cached_data.get('status_code', 200)
                )
                excluded_headers = {'content-length', 'transfer-encoding'}
                for key, value in cached_data.get('headers', {}).items():
                    if key.lower() not in excluded_headers:
                        response.headers[key] = value
                return response
        except Exception as e:
            logging.error(f"Error reconstructing response: {e}")

        return JSONResponse(content={"error": "Cache reconstruction failed"}, status_code=500)

    def _calculate_ttl(self, request: Request, response_data: Dict[str, Any]) -> int:
        """Calculate appropriate TTL based on request and response characteristics"""
        base_ttl = self.cache_ttl
        try:
            if response_data.get('type') == 'json':
                data = json.loads(response_data.get('data', '{}'))
                if isinstance(data, dict):
                    if 'choices' in data and data['choices']:
                        return int(base_ttl * 0.5)
                    return base_ttl
        except Exception:
            pass
        return base_ttl

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics"""
        total_cacheable = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / max(1, total_cacheable)) * 100
        stats = {
            'enabled': self.enable_cache,
            'hit_rate': hit_rate,
            'total_requests': self.stats['total_requests'],
            'cacheable_requests': total_cacheable,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_sets': self.stats['cache_sets'],
            'cache_errors': self.stats['cache_errors'],
            'bytes_saved': self.stats['bytes_saved'],
            'config': {
                'cache_ttl': self.cache_ttl,
                'cacheable_paths': self.cacheable_paths,
                'non_cacheable_paths': list(self.non_cacheable_paths)
            }
        }
        if self.smart_cache:
            try:
                smart_cache_stats = self.smart_cache.get_stats()
                stats['smart_cache'] = smart_cache_stats
            except Exception as e:
                stats['smart_cache_error'] = str(e)
        return stats

    async def clear_cache(self):
        """Clear all cached data"""
        if self.smart_cache:
            try:
                await handle_error_with_circuit_breaker(
                    'cache',
                    self.smart_cache.clear,
                    context={'action': 'clear_cache'}
                )
                logging.info("Cache cleared successfully")
            except Exception as e:
                logging.error(f"Error clearing cache: {e}")


# Factory function for easy integration
def create_caching_middleware(
    cache_config: CacheConfig = None,
    enable_cache: bool = True,
    cacheable_paths: list = None,
    cache_ttl: int = 3600
):
    """Factory function to create caching middleware with configuration"""
    def middleware_factory(app):
        return SmartCachingMiddleware(
            app=app,
            cache_config=cache_config,
            enable_cache=enable_cache,
            cacheable_paths=cacheable_paths,
            cache_ttl=cache_ttl
        )
    return middleware_factory
