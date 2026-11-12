import logging
import time
from typing import Optional

import httpx

from app.config import settings
from app.metrics import (
    inference_duration_seconds,
    inference_requests_total,
    ollama_available,
    tokens_generated_total,
)

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    pass


async def check_ollama_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            is_up = response.status_code == 200
            ollama_available.set(1 if is_up else 0)
            return is_up
    except Exception:
        ollama_available.set(0)
        return False


async def generate(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict:
    model = model or settings.ollama_model
    start_time = time.time()

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            response = await client.post(
                f"{settings.ollama_host}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        duration = time.time() - start_time
        tokens = data.get("eval_count", 0)

        inference_requests_total.labels(model=model, status="success").inc()
        inference_duration_seconds.labels(model=model).observe(duration)
        if tokens:
            tokens_generated_total.labels(model=model).inc(tokens)

        return {
            "text": data.get("response", ""),
            "model": model,
            "tokens_generated": tokens,
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "duration_seconds": duration,
            "tokens_per_second": tokens / duration if duration > 0 and tokens > 0 else 0.0,
        }

    except httpx.TimeoutException:
        duration = time.time() - start_time
        inference_requests_total.labels(model=model, status="timeout").inc()
        inference_duration_seconds.labels(model=model).observe(duration)
        raise InferenceError(f"Ollama request timed out after {settings.ollama_timeout}s")

    except httpx.HTTPStatusError as exc:
        duration = time.time() - start_time
        inference_requests_total.labels(model=model, status="error").inc()
        inference_duration_seconds.labels(model=model).observe(duration)
        raise InferenceError(
            f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
        )

    except Exception as exc:
        duration = time.time() - start_time
        inference_requests_total.labels(model=model, status="error").inc()
        inference_duration_seconds.labels(model=model).observe(duration)
        raise InferenceError(f"Inference failed: {exc}") from exc


async def list_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        raise InferenceError(f"Failed to list models: {exc}") from exc
