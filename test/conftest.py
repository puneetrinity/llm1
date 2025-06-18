# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    from main import app

    class DummyModelRouter:
        async def get_available_models(self):
            return [
                {"id": "phi3.5", "object": "model"},
                {"id": "mistral:7b-instruct-q4_0", "object": "model"},
                {"id": "gemma:7b-instruct", "object": "model"},
                {"id": "llama3:8b-instruct-q4_0", "object": "model"},
            ]

    with patch("main.model_router", new=DummyModelRouter()):
        yield TestClient(app)


@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
