from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MINILM_INT8_DIR = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2-int8"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_MEMORY_DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'runtime' / 'rag-memory.sqlite3'}"


def _project_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _memory_database_url() -> str:
    value = os.getenv("MEMORY_DATABASE_URL", DEFAULT_MEMORY_DATABASE_URL)
    if not value.startswith("sqlite:///"):
        return value
    location = value.removeprefix("sqlite:///")
    if location == ":memory:" or Path(location).is_absolute():
        return value
    return f"sqlite:///{_project_path(location)}"


def _embedding_dimension() -> int:
    """Keep stale BGE-era EMBEDDING_DIM values from changing the MiniLM schema."""
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    model_path = os.getenv(
        "EMBEDDING_ONNX_MODEL_PATH",
        str(DEFAULT_MINILM_INT8_DIR / "onnx" / "model_quint8_avx2.onnx"),
    )
    if "all-minilm-l6-v2" in model.lower() or "all-minilm-l6-v2-int8" in model_path.lower():
        return DEFAULT_EMBEDDING_DIM
    return int(os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("RAG_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("RAG_API_PORT", "8003"))
    cors_origins: tuple[str, ...] = tuple(
        _split_csv(os.getenv("RAG_API_CORS_ORIGINS", "http://localhost:3003,http://localhost:3000"))
    )

    llm_base_url: str = os.getenv("LLM_BASE_URL") or os.getenv("IPHY_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("IPHY_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL") or os.getenv("IPHY_MODEL", "mock-growth-rag")
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "onnx")
    embedding_onnx_model_path: str = _project_path(
        os.getenv(
            "EMBEDDING_ONNX_MODEL_PATH",
            str(DEFAULT_MINILM_INT8_DIR / "onnx" / "model_quint8_avx2.onnx"),
        )
    )
    embedding_tokenizer_path: str = _project_path(
        os.getenv(
            "EMBEDDING_TOKENIZER_PATH",
            str(DEFAULT_MINILM_INT8_DIR),
        )
    )
    embedding_dim: int = _embedding_dimension()
    embedding_timeout_seconds: int = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "256"))

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "growth_records")

    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_token: str = os.getenv("MILVUS_TOKEN", "")
    milvus_db_name: str = os.getenv("MILVUS_DB_NAME", "default")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "growth_records_minilm")
    milvus_text_field: str = os.getenv("MILVUS_TEXT_FIELD", "text")
    milvus_sparse_field: str = os.getenv("MILVUS_SPARSE_FIELD", "text_sparse")
    milvus_dense_field: str = os.getenv("MILVUS_DENSE_FIELD", "text_dense")
    milvus_bm25_top_k: int = int(os.getenv("MILVUS_BM25_TOP_K", "50"))
    milvus_dense_top_k: int = int(os.getenv("MILVUS_DENSE_TOP_K", "50"))
    milvus_rrf_k: int = int(os.getenv("MILVUS_RRF_K", "60"))
    milvus_consistency_level: str = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Bounded")

    retrieval_provider: str = os.getenv("RETRIEVAL_PROVIDER", "milvus-hybrid")
    growth_records_path: str = os.getenv("GROWTH_RECORDS_PATH", str(PROJECT_ROOT / "rawData.jsonl"))

    # The local SQLite default keeps the RAG host self-contained on a 1 vCPU / 1 GiB machine.
    # Production can point this at an external PostgreSQL instance with the same schema.
    memory_database_url: str = _memory_database_url()
    memory_checkpoint_backend: str = os.getenv("MEMORY_CHECKPOINT_BACKEND", "auto")
    memory_postgres_connect_timeout_seconds: int = int(
        os.getenv("MEMORY_POSTGRES_CONNECT_TIMEOUT_SECONDS", "3")
    )
    memory_short_max_messages: int = int(os.getenv("MEMORY_SHORT_MAX_MESSAGES", "10"))
    memory_summary_max_chars: int = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "1200"))
    memory_message_max_chars: int = int(os.getenv("MEMORY_MESSAGE_MAX_CHARS", "4000"))
    memory_active_context_max_items: int = int(os.getenv("MEMORY_ACTIVE_CONTEXT_MAX_ITEMS", "8"))
    memory_long_max_items_per_user: int = int(os.getenv("MEMORY_LONG_MAX_ITEMS_PER_USER", "200"))
    memory_long_prompt_max_items: int = int(os.getenv("MEMORY_LONG_PROMPT_MAX_ITEMS", "8"))
    memory_long_prompt_max_chars: int = int(os.getenv("MEMORY_LONG_PROMPT_MAX_CHARS", "1800"))
    memory_session_ttl_days: int = int(os.getenv("MEMORY_SESSION_TTL_DAYS", "30"))
    memory_event_ttl_days: int = int(os.getenv("MEMORY_EVENT_TTL_DAYS", "90"))
    memory_cleanup_interval_seconds: int = int(os.getenv("MEMORY_CLEANUP_INTERVAL_SECONDS", "21600"))
    memory_thread_rollover_turns: int = int(os.getenv("MEMORY_THREAD_ROLLOVER_TURNS", "100"))
    memory_semantic_search_enabled: str = os.getenv("MEMORY_SEMANTIC_SEARCH_ENABLED", "auto")
    memory_semantic_candidate_limit: int = int(os.getenv("MEMORY_SEMANTIC_CANDIDATE_LIMIT", "24"))
    memory_worker_poll_seconds: int = int(os.getenv("MEMORY_WORKER_POLL_SECONDS", "3"))


settings = Settings()
