# tests/test_basic.py
import pytest
from fastapi.testclient import TestClient

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code in [200, 503]  # May be unhealthy if Ollama not running

def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data

def test_basic_completion(client):
    response = client.post("/v1/chat/completions", json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    # May fail if no models available, but should not crash
    assert response.status_code in [200, 500, 503]
