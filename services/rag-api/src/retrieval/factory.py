from __future__ import annotations

from ..config import settings
from .local_jsonl import LocalJsonlRetrievalService
from .milvus_hybrid import MilvusHybridRetrievalService
from .mock_retrieval import MockRetrievalService
from .service import RetrievalService


def get_default_retrieval_service() -> RetrievalService:
    if settings.retrieval_provider == "mock":
        return MockRetrievalService()
    if settings.retrieval_provider == "local-jsonl":
        return LocalJsonlRetrievalService(settings.growth_records_path)
    if settings.retrieval_provider == "milvus-hybrid":
        return MilvusHybridRetrievalService()
    raise ValueError(f"Unsupported RETRIEVAL_PROVIDER: {settings.retrieval_provider}")
