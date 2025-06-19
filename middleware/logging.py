# middleware/logging.py
import logging
import time
import uuid
import json
from datetime import datetime
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, enable_detailed_logging: bool = True):
        super().__init__(app)
        self.enable_detailed_logging = enable_detailed_logging
        self.request_counts = {}
        self.error_counts = {}

        # Configure structured logging
        self.setup_logging()

    def setup_logging(self):
        """Set up structured logging format"""

        # Create custom formatter for structured logs
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

                # Add extra fields if present
                if hasattr(record, "request_id"):
                    log_entry["request_id"] = record.request_id
                if hasattr(record, "user_id"):
                    log_entry["user_id"] = record.user_id
                if hasattr(record, "duration"):
                    log_entry["duration"] = record.duration

                return json.dumps(log_entry)

        # Apply formatter to existing handlers
        for handler in logging.getLogger().handlers:
            if self.enable_detailed_logging:
                handler.setFormatter(StructuredFormatter())

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Start timing
        start_time = time.time()

        # Extract request info
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        client_ip = self.get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")

        # Get user info if available
        user_id = "anonymous"
        if hasattr(request.state, "user") and request.state.user:
            user_id = request.state.user.get("user_id", "unknown")

        # Log incoming request
        self.log_request_start(
            request_id, method, path, query_params, client_ip, user_agent, user_id
        )

        # Track request count
        endpoint = f"{method} {path}"
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1

        # Process request
        try:
            # Add request ID to request state for downstream use
            request.state.request_id = request_id

            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log successful response
            self.log_request_success(
                request_id, response.status_code, duration, user_id
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"

            return response

        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time

            # Log error
            self.log_request_error(request_id, str(e), duration, user_id)

            # Track error count
            error_type = type(e).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

            # Re-raise the exception
            raise

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""

        # Check X-Forwarded-For header first (for load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    def log_request_start(
        self,
        request_id: str,
        method: str,
        path: str,
        query_params: str,
        client_ip: str,
        user_agent: str,
        user_id: str,
    ):
        """Log incoming request"""

        extra = {"request_id": request_id, "user_id": user_id}

        message = f"[{request_id}] {method} {path}"
        if query_params:
            message += f"?{query_params}"
        message += f" - Client: {client_ip}"

        if self.enable_detailed_logging:
            message += f" - User-Agent: {user_agent[:100]}..."

        logging.info(message, extra=extra)

    def log_request_success(
        self, request_id: str, status_code: int, duration: float, user_id: str
    ):
        """Log successful request completion"""

        extra = {"request_id": request_id, "user_id": user_id, "duration": duration}

        # Color code by response time
        if duration < 1.0:
            level = logging.INFO
        elif duration < 5.0:
            level = logging.WARNING
        else:
            level = logging.ERROR

        message = f"[{request_id}] {status_code} - Duration: {duration:.3f}s"

        logging.log(level, message, extra=extra)

    def log_request_error(
        self, request_id: str, error: str, duration: float, user_id: str
    ):
        """Log request error"""

        extra = {"request_id": request_id, "user_id": user_id, "duration": duration}

        message = f"[{request_id}] ERROR - Duration: {duration:.3f}s - Error: {error}"

        logging.error(message, extra=extra)

    def get_logging_stats(self) -> dict:
        """Get logging statistics"""

        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / max(1, total_requests)) * 100

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "top_endpoints": sorted(
                self.request_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "error_breakdown": dict(self.error_counts),
        }
