from __future__ import annotations

from ..config import settings
from .client import LLMClient
from .mock_client import MockLLMClient
from .openai_compatible import OpenAICompatibleLLMClient


def get_llm_backend_name() -> str:
    """Return the active adapter without exposing any connection details."""
    if settings.llm_provider == "mock":
        return "mock"
    if settings.llm_base_url and settings.llm_api_key and settings.llm_model != "mock-growth-rag":
        return "openai-compatible"
    return "mock"


def get_default_llm_client() -> LLMClient:
    if get_llm_backend_name() == "openai-compatible":
        return OpenAICompatibleLLMClient(settings)
    return MockLLMClient()
