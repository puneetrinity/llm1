# middleware/cors.py
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import logging


class EnhancedCORSMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app, allowed_origins: list = None, enable_credentials: bool = True
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.enable_credentials = enable_credentials

        # Add CORS middleware to the app
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.allowed_origins,
            allow_credentials=self.enable_credentials,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Response-Time", "X-RateLimit-*"],
        )

    async def dispatch(self, request: Request, call_next):
        # Log CORS requests if needed
        origin = request.headers.get("Origin")

        if origin and self.allowed_origins != ["*"]:
            if origin not in self.allowed_origins:
                logging.warning(f"CORS request from unauthorized origin: {origin}")

        return await call_next(request)
