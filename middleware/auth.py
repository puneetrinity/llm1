# middleware/auth.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth_service):
        super().__init__(app)
        self.auth_service = auth_service

        # Public endpoints that don't require auth
        self.public_endpoints = {
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/favicon.ico",
        }

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)

        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check if auth is enabled
        if not self.auth_service.settings.ENABLE_AUTH:
            return await call_next(request)

        # Extract API key
        api_key = request.headers.get(self.auth_service.settings.API_KEY_HEADER)

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "API key required",
                    "message": f"Please provide API key in {self.auth_service.settings.API_KEY_HEADER} header",
                },
            )

        # Validate API key
        user_info = self.auth_service.validate_api_key(api_key)

        if not user_info:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid",
                },
            )

        # Add user info to request state
        request.state.user = user_info

        # Log authenticated request
        logging.info(
            f"Authenticated request: {user_info['user_id']} - {request.method} {request.url.path}"
        )

        return await call_next(request)
