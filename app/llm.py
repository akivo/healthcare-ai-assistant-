"""
LLM client for Ollama integration.
Connects to the locally running Ollama service and provides
health check functionality.
"""

import logging
import httpx
from functools import lru_cache
from langchain_ollama import ChatOllama
from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """
    Initialize and return a cached ChatOllama LLM instance.
    Low temperature (0.1) ensures factual, consistent responses.
    """
    settings = get_settings()
    logger.info(f"Initializing LLM: {settings.llm_model} @ {settings.ollama_base_url}")

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=512,
    )


def check_ollama_health() -> dict:
    """
    Check if Ollama service is running and the configured model is available.

    Returns:
        dict with connection status and available models.
    """
    settings = get_settings()

    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()

        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        model_available = any(settings.llm_model in m for m in models)

        return {
            "status": "connected",
            "model": settings.llm_model,
            "model_available": model_available,
            "available_models": models,
        }

    except httpx.ConnectError:
        return {
            "status": "disconnected",
            "error": "Ollama service not running. Start with: ollama serve",
            "model": settings.llm_model,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": settings.llm_model,
        }
