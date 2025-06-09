# middleware/rate_limit.py
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, default_limit: int = 60):
        super().__init__(app)
        self.default_limit = default_limit
        self.requests = defaultdict(deque)
        self.user_limits = {}  # Store user-specific limits
        self.cleanup_task = None
        
        # Start cleanup task
        asyncio.create_task(self.start_cleanup_task())
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get identifier for rate limiting
        identifier = await self.get_rate_limit_identifier(request)
        
        # Get rate limit for this user/IP
        rate_limit = await self.get_rate_limit(request, identifier)
        
        # Check if rate limited
        if await self.is_rate_limited(identifier, rate_limit):
            logging.warning(f"Rate limit exceeded for {identifier}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {rate_limit} requests per minute exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record this request
        await self.record_request(identifier)
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Calculate remaining requests
        current_count = len(self.requests[identifier])
        remaining = max(0, rate_limit - current_count)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        return response
    
    async def get_rate_limit_identifier(self, request: Request) -> str:
        """Get identifier for rate limiting (API key or IP)"""
        
        # Try to get API key first (more specific)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:8]}..."  # Truncate for privacy
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        
        if forwarded_for:
            # Use first IP in X-Forwarded-For chain
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def get_rate_limit(self, request: Request, identifier: str) -> int:
        """Get rate limit for this request"""
        
        # Check if user has custom rate limit
        if hasattr(request.state, 'user') and request.state.user:
            user_limit = request.state.user.get('rate_limit')
            if user_limit:
                return user_limit
        
        # Check for API key specific limits
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key in self.user_limits:
            return self.user_limits[api_key]
        
        return self.default_limit
    
    async def is_rate_limited(self, identifier: str, limit: int) -> bool:
        """Check if identifier is rate limited"""
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests for this identifier
        while self.requests[identifier] and self.requests[identifier][0] <= minute_ago:
            self.requests[identifier].popleft()
        
        # Check if over limit
        return len(self.requests[identifier]) >= limit
    
    async def record_request(self, identifier: str):
        """Record a request for rate limiting"""
        now = datetime.now()
        self.requests[identifier].append(now)
        
        # Limit deque size to prevent memory issues
        if len(self.requests[identifier]) > 1000:
            # Keep only recent requests
            self.requests[identifier] = deque(
                list(self.requests[identifier])[-500:], 
                maxlen=1000
            )
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                await self.cleanup_old_records()
            except Exception as e:
                logging.error(f"Error in rate limit cleanup: {e}")
    
    async def cleanup_old_records(self):
        """Clean up old rate limit records"""
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Remove identifiers with no recent requests
        identifiers_to_remove = []
        
        for identifier, request_times in self.requests.items():
            # Remove old requests
            while request_times and request_times[0] <= hour_ago:
                request_times.popleft()
            
            # Mark for removal if no recent requests
            if not request_times:
                identifiers_to_remove.append(identifier)
        
        # Remove empty identifiers
        for identifier in identifiers_to_remove:
            del self.requests[identifier]
        
        if identifiers_to_remove:
            logging.info(f"Cleaned up {len(identifiers_to_remove)} rate limit records")
    
    def set_user_rate_limit(self, api_key: str, limit: int):
        """Set custom rate limit for a specific API key"""
        self.user_limits[api_key] = limit
    
    def get_rate_limit_stats(self) -> dict:
        """Get rate limiting statistics"""
        
        total_identifiers = len(self.requests)
        total_requests = sum(len(requests) for requests in self.requests.values())
        
        # Calculate top requesters
        top_requesters = sorted(
            [(id, len(reqs)) for id, reqs in self.requests.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_identifiers": total_identifiers,
            "total_requests_last_hour": total_requests,
            "default_limit": self.default_limit,
            "custom_limits": len(self.user_limits),
            "top_requesters": top_requesters
        }