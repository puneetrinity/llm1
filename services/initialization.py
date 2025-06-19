# services/initialization.py
import asyncio
import logging
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from config import settings
from services.ollama_client import OllamaClient
from services.router import LLMRouter
from services.model_warmup import ModelWarmupService

logger = logging.getLogger(__name__)


class InitializationService:
    """Service to handle application initialization with proper retry logic"""

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.max_retries = 60
        self.retry_delay = 1.0
        self.initialization_state = {
            "ollama_connected": False,
            "models_available": False,
            "services_ready": False,
            "timestamp": None,
        }

    async def wait_for_ollama(self) -> bool:
        """Wait for Ollama service to be ready with exponential backoff"""
        logger.info(f"Waiting for Ollama service at {self.ollama_base_url}...")

        retry_count = 0
        backoff_delay = self.retry_delay

        while retry_count < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.ollama_base_url}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(
                                f"✅ Ollama is ready! Found {len(data.get('models', []))} models"
                            )
                            self.initialization_state["ollama_connected"] = True
                            self.initialization_state["timestamp"] = (
                                datetime.now().isoformat()
                            )
                            return True

            except aiohttp.ClientError as e:
                logger.debug(
                    f"Ollama not ready yet (attempt {retry_count + 1}/{self.max_retries}): {str(e)}"
                )
            except Exception as e:
                logger.error(f"Unexpected error checking Ollama: {str(e)}")

            retry_count += 1
            if retry_count < self.max_retries:
                await asyncio.sleep(backoff_delay)
                # Exponential backoff with max delay of 5 seconds
                backoff_delay = min(backoff_delay * 1.5, 5.0)

        logger.error(f"❌ Ollama failed to start after {self.max_retries} attempts")
        return False

    async def check_models(self) -> Dict[str, Any]:
        """Check available models and their status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])

                        # Extract model names
                        model_names = [model.get("name", "") for model in models]

                        # Check for required models
                        required_models = [
                            "phi3.5",
                            "mistral:7b-instruct-q4_0",
                            "gemma:7b-instruct",
                            "llama3:8b-instruct-q4_0",
                        ]

                        available_required = [
                            m for m in required_models if m in model_names
                        ]
                        missing_models = [
                            m for m in required_models if m not in model_names
                        ]

                        self.initialization_state["models_available"] = (
                            len(available_required) > 0
                        )

                        return {
                            "total_models": len(models),
                            "available_models": model_names,
                            "required_models_available": available_required,
                            "missing_models": missing_models,
                            "all_required_available": len(missing_models) == 0,
                        }

        except Exception as e:
            logger.error(f"Error checking models: {str(e)}")
            return {
                "error": str(e),
                "total_models": 0,
                "available_models": [],
                "required_models_available": [],
                "missing_models": [],
            }

    async def initialize_with_retry(self, services: Dict[str, Any]) -> bool:
        """Initialize all services with proper retry logic"""
        logger.info("🚀 Starting initialization sequence...")

        # Step 1: Wait for Ollama
        if not await self.wait_for_ollama():
            logger.error("Failed to connect to Ollama service")
            return False

        # Step 2: Check models
        model_status = await self.check_models()
        logger.info(f"Model status: {model_status}")

        if model_status.get("total_models", 0) == 0:
            logger.warning("No models available yet, but continuing...")

        # Step 3: Initialize other services
        try:
            # Initialize Ollama client if provided
            if "ollama_client" in services and services["ollama_client"]:
                await services["ollama_client"].initialize()
                logger.info("✅ Ollama client initialized")

            # Initialize router if provided
            if "router" in services and services["router"]:
                await services["router"].initialize()
                logger.info("✅ Router initialized")

            # Initialize model warmup if provided
            if "warmup_service" in services and services["warmup_service"]:
                asyncio.create_task(services["warmup_service"].start_warmup_service())
                logger.info("✅ Model warmup service started")

            self.initialization_state["services_ready"] = True
            logger.info("✅ All services initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current initialization status"""
        return {
            **self.initialization_state,
            "ready": all(
                [
                    self.initialization_state["ollama_connected"],
                    self.initialization_state["services_ready"],
                ]
            ),
        }


# Usage in main.py:
async def initialize_services_with_retry():
    """Enhanced initialization with retry logic"""
    init_service = InitializationService(settings.OLLAMA_BASE_URL)

    ollama_client = OllamaClient()
    model_router = LLMRouter(ollama_client=ollama_client)
    warmup_service = ModelWarmupService(ollama_client, model_router)

    services = {
        "ollama_client": ollama_client,
        "router": model_router,
        "warmup_service": warmup_service if settings.ENABLE_MODEL_WARMUP else None,
    }

    success = await init_service.initialize_with_retry(services)

    global services_state
    services_state = {}
    services_state.update(init_service.get_status())

    return success
