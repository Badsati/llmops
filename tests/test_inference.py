from unittest.mock import AsyncMock, patch

import pytest


def test_health_endpoint_ollama_up(client):
    with patch("app.main.check_ollama_health", new_callable=AsyncMock, return_value=True):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ollama"] == "up"


def test_health_endpoint_ollama_down(client):
    with patch("app.main.check_ollama_health", new_callable=AsyncMock, return_value=False):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["ollama"] == "down"


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"llmops_app_info" in response.content


def test_inference_success(client, inference_response):
    with patch("app.main.generate", new_callable=AsyncMock, return_value=inference_response):
        response = client.post(
            "/api/v1/inference",
            json={"prompt": "What is the capital of France?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == inference_response["text"]
    assert data["model"] == inference_response["model"]
    assert data["tokens_generated"] == inference_response["tokens_generated"]


def test_inference_empty_prompt(client):
    response = client.post("/api/v1/inference", json={"prompt": ""})
    assert response.status_code == 422


def test_inference_service_unavailable(client):
    from app.inference import InferenceError

    with patch(
        "app.main.generate",
        new_callable=AsyncMock,
        side_effect=InferenceError("Ollama is down"),
    ):
        response = client.post(
            "/api/v1/inference", json={"prompt": "Hello"}
        )
    assert response.status_code == 503
    assert "Ollama is down" in response.json()["detail"]


def test_models_endpoint(client):
    with patch(
        "app.main.list_models",
        new_callable=AsyncMock,
        return_value=["qwen2.5:3b", "nomic-embed-text"],
    ):
        response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert "qwen2.5:3b" in response.json()["models"]


def test_inference_temperature_validation(client):
    response = client.post(
        "/api/v1/inference",
        json={"prompt": "Hello", "temperature": 5.0},
    )
    assert response.status_code == 422


def test_inference_max_tokens_validation(client):
    response = client.post(
        "/api/v1/inference",
        json={"prompt": "Hello", "max_tokens": 0},
    )
    assert response.status_code == 422
