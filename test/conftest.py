# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from main_enhanced import app
    return TestClient(app)

@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
