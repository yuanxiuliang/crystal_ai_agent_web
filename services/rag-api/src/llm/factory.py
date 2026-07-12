from __future__ import annotations

from ..config import settings
from .client import LLMClient
from .mock_client import MockLLMClient
from .openai_compatible import OpenAICompatibleLLMClient


def get_default_llm_client() -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLMClient()
    if settings.llm_base_url and settings.llm_api_key and settings.llm_model != "mock-growth-rag":
        return OpenAICompatibleLLMClient(settings)
    return MockLLMClient()

