# middleware/security.py - Security Middleware
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Security middleware manager"""

    def __init__(self, settings):
        self.settings = settings
        self.rate_limiter = RateLimiter(settings)

    def add_security_middleware(self, app):
        """Add all security middleware to the app"""

        # HTTPS redirect in production
        if self.settings.ENVIRONMENT == "production":
            app.add_middleware(HTTPSRedirectMiddleware)

        # Trusted host middleware
        if self.settings.ENVIRONMENT == "production":
            # In production, configure with actual allowed hosts
            allowed_hosts = ["*"]  # Configure with actual hosts
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

        # Security headers middleware
        app.add_middleware(SecurityHeadersMiddleware)

        # Rate limiting middleware
        if self.settings.ENABLE_RATE_LIMITING:
            app.add_middleware(RateLimitingMiddleware, rate_limiter=self.rate_limiter)

        # Request logging middleware
        app.add_middleware(RequestLoggingMiddleware, settings=self.settings)

        logger.info("✅ Security middleware configured")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # HSTS header for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Remove server information
        response.headers.pop("server", None)

        return response


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, settings):
        self.settings = settings
        self.requests = {}  # {ip: [(timestamp, count), ...]}
        self.window_size = 60  # 1 minute window

    def is_allowed(self, client_ip: str, limit: int = None) -> tuple[bool, dict]:
        """Check if request is allowed and return rate limit info"""
        if limit is None:
            limit = self.settings.DEFAULT_RATE_LIMIT

        current_time = time.time()
        window_start = current_time - self.window_size

        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (timestamp, count)
                for timestamp, count in self.requests[client_ip]
                if timestamp > window_start
            ]
        else:
            self.requests[client_ip] = []

        # Count requests in current window
        current_count = sum(count for _, count in self.requests[client_ip])

        # Check if limit exceeded
        allowed = current_count < limit

        if allowed:
            # Add current request
            self.requests[client_ip].append((current_time, 1))

        rate_limit_info = {
            "limit": limit,
            "remaining": max(0, limit - current_count - (1 if allowed else 0)),
            "reset": int(window_start + self.window_size),
            "current": current_count,
        }

        return allowed, rate_limit_info


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""

    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

        # Paths to exclude from rate limiting
        self.excluded_paths = {"/health", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check rate limit
        allowed, rate_info = self.rate_limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning(f"🚫 Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "rate_limit": rate_info,
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": "60",
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support"""
        # Check for forwarded IP headers (from load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware"""

    def __init__(self, app, settings):
        super().__init__(app)
        self.settings = settings

        # Paths to exclude from detailed logging
        self.excluded_paths = (
            {"/health", "/metrics"} if settings.ENVIRONMENT == "production" else set()
        )

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Get request info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")

        # Process request
        try:
            response = await call_next(request)

            # Calculate response time
            process_time = time.time() - start_time

            # Log request (skip health checks in production)
            if request.url.path not in self.excluded_paths:
                logger.info(
                    f"📝 {request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.3f}s - "
                    f"IP: {client_ip} - "
                    f"UA: {user_agent[:50]}..."
                )

            # Add response headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = self._generate_request_id()

            return response

        except Exception as e:
            process_time = time.time() - start_time

            logger.error(
                f"❌ {request.method} {request.url.path} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.3f}s - "
                f"IP: {client_ip}"
            )

            raise

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid

        return str(uuid.uuid4())[:8]


class CORSConfig:
    """CORS configuration helper"""

    def __init__(self, settings):
        self.settings = settings

    def add_cors_middleware(self, app):
        """Add CORS middleware with proper configuration"""
        from fastapi.middleware.cors import CORSMiddleware

        # Get CORS origins
        if isinstance(self.settings.CORS_ORIGINS, str):
            origins = [
                origin.strip() for origin in self.settings.CORS_ORIGINS.split(",")
            ]
        else:
            origins = self.settings.CORS_ORIGINS

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=self.settings.CORS_ALLOW_CREDENTIALS,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
            expose_headers=[
                "X-Process-Time",
                "X-Request-ID",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ],
        )

        logger.info(f"✅ CORS configured with origins: {origins}")


# Export middleware classes
__all__ = [
    "SecurityMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimiter",
    "RateLimitingMiddleware",
    "RequestLoggingMiddleware",
    "CORSConfig",
]
