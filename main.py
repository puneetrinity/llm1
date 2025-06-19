#!/usr/bin/env python3
"""
Enhanced LLM Proxy - Main Application (Auto-Generated)
"""
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi import status
import traceback

from config import Settings
from models.requests import ChatCompletionRequest, CompletionRequest
from models.responses import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionResponse,
    ModelResponse,
    ModelListResponse,
)
from services.ollama_client import OllamaClient
from services.router import LLMRouter
from services.smart_cache import SmartCache
from middleware.auth import AuthMiddleware
from middleware.rate_limit import RateLimitMiddleware

# Initialize settings
settings = Settings()

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("llm-proxy")

# Create FastAPI app
app = FastAPI(title="Enhanced LLM Proxy", version="4.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware (if enabled)
try:
    from services.auth import AuthService

    auth_service = AuthService(settings)
    if settings.ENABLE_AUTH:
        app.add_middleware(AuthMiddleware, auth_service=auth_service)
except ImportError:
    logger.warning("AuthService not available; skipping auth middleware.")

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, default_limit=60)

# Initialize router and cache
ollama_client = OllamaClient()
router = LLMRouter(ollama_client=ollama_client)
model_router = router
cache = SmartCache()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest):
    # Example: route request, use cache, etc.
    # This is a stub; implement your logic here
    return JSONResponse(content={"message": "Chat completion endpoint stub."})


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(request: CompletionRequest):
    return JSONResponse(content={"message": "Completion endpoint stub."})


@app.get("/v1/models", response_model=ModelListResponse)
def list_models():
    # Use the model_router to get available models
    import asyncio

    models = []
    try:
        # If get_available_models is async, run it in the event loop
        if hasattr(model_router, "get_available_models"):
            coro = model_router.get_available_models()
            if asyncio.iscoroutine(coro):
                models = asyncio.get_event_loop().run_until_complete(coro)
            else:
                models = coro
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
    return {"data": models}


# --- Enhanced Features Activation ---

# Enhanced Router (semantic classification)
if getattr(settings, "ENABLE_ENHANCED_FEATURES", False) or getattr(
    settings, "ENABLE_SEMANTIC_CLASSIFICATION", False
):
    try:
        from services.enhanced_router import EnhancedLLMRouter

        router = EnhancedLLMRouter(ollama_client=ollama_client)
        logger.info("EnhancedLLMRouter enabled.")
    except ImportError as e:
        logger.warning(f"Could not import EnhancedLLMRouter: {e}")

# Streaming Support
if getattr(settings, "ENABLE_STREAMING", False):
    try:
        from services.streaming import StreamingService

        streaming_service = StreamingService(ollama_client)

        @app.post("/v1/chat/completions/stream")
        async def chat_completions_stream(request: Request):
            req_data = await request.json()
            model = req_data.get("model", settings.DEFAULT_MODEL)
            return await streaming_service.stream_chat_completion(req_data, model)

        logger.info("Streaming endpoint enabled.")
    except ImportError as e:
        logger.warning(f"Could not import StreamingService: {e}")

# Dashboard (React or Enhanced)
if getattr(settings, "ENABLE_DASHBOARD", False) or getattr(
    settings, "ENABLE_REACT_DASHBOARD", False
):
    try:
        from utils.dashboard import EnhancedDashboard
        from utils.metrics import MetricsCollector
        from utils.performance_monitor import PerformanceMonitor

        dashboard = EnhancedDashboard(
            metrics_collector=MetricsCollector(),
            performance_monitor=PerformanceMonitor(),
            cache_service=cache,
            warmup_service=None,
            semantic_classifier=None,
        )

        @app.get("/dashboard")
        async def dashboard_endpoint():
            return await dashboard.get_comprehensive_dashboard()

        logger.info("Dashboard endpoint enabled.")
    except ImportError as e:
        logger.warning(f"Could not import EnhancedDashboard: {e}")

# WebSocket Dashboard
if getattr(settings, "ENABLE_WEBSOCKET_DASHBOARD", False) or getattr(
    settings, "ENABLE_WEBSOCKET", False
):
    try:
        from utils.websocket_dashboard import WebSocketDashboard
        from fastapi import WebSocket

        ws_dashboard = WebSocketDashboard(dashboard)

        @app.websocket("/ws/dashboard")
        async def websocket_dashboard_endpoint(websocket: WebSocket):
            await ws_dashboard.connect(websocket)

        logger.info("WebSocket dashboard endpoint enabled.")
    except ImportError as e:
        logger.warning(f"Could not import WebSocketDashboard: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error.", "error": str(exc)},
    )


@app.on_event("startup")
async def wait_for_ollama():
    max_attempts = 30  # e.g., wait up to 30 x 2s = 60 seconds
    delay = 2
    for attempt in range(1, max_attempts + 1):
        try:
            healthy = await ollama_client.health_check()
            if healthy:
                logger.info(f"Ollama is ready at {ollama_client.base_url}")
                return
            else:
                logger.warning(f"Ollama not ready (attempt {attempt}/{max_attempts})")
        except Exception as e:
            logger.warning(
                f"Ollama health check failed: {e} (attempt {attempt}/{max_attempts})"
            )
        await asyncio.sleep(delay)
    logger.error(
        f"Ollama did not become ready after {max_attempts * delay} seconds. Exiting."
    )
    import sys

    sys.exit(1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
