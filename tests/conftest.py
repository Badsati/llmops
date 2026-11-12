from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def mock_ollama_health():
    """Patch Ollama health check to always return True."""
    with patch("app.inference.check_ollama_health", new_callable=AsyncMock, return_value=True):
        yield


@pytest.fixture
def inference_response():
    return {
        "text": "The capital of France is Paris.",
        "model": "qwen2.5:3b",
        "tokens_generated": 8,
        "prompt_tokens": 6,
        "duration_seconds": 0.42,
        "tokens_per_second": 19.0,
    }


@pytest.fixture
def client(mock_ollama_health):
    # Import app after patches so lifespan health check is mocked
    from app.main import app

    with TestClient(app) as c:
        yield c
