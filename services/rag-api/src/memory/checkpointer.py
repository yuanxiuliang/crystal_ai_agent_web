from __future__ import annotations

import asyncio
import threading
from typing import Any

from ..config import Settings, settings


class CheckpointerRuntime:
    """Own the optional PostgreSQL LangGraph checkpointer for one application process."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._pool: Any | None = None
        self._checkpointer: Any | None = None

    @property
    def enabled(self) -> bool:
        backend = self.config.memory_checkpoint_backend.strip().lower()
        is_postgres = self.config.memory_database_url.startswith(("postgresql://", "postgres://"))
        if backend == "auto":
            return is_postgres
        if backend == "postgres":
            return True
        if backend in {"none", "sqlite", "store"}:
            return False
        raise ValueError("MEMORY_CHECKPOINT_BACKEND must be auto, postgres, none, sqlite, or store.")

    async def get(self) -> Any | None:
        if not self.enabled:
            return None
        if not self.config.memory_database_url.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("MEMORY_CHECKPOINT_BACKEND=postgres requires a PostgreSQL MEMORY_DATABASE_URL.")
        if self._checkpointer is not None:
            return self._checkpointer

        async with self._lock:
            if self._checkpointer is not None:
                return self._checkpointer
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from psycopg.rows import dict_row
                from psycopg_pool import AsyncConnectionPool
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL short-term memory requires the rag-api [postgres] dependencies."
                ) from exc

            self._pool = AsyncConnectionPool(
                self.config.memory_database_url,
                min_size=1,
                max_size=4,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            )
            await self._pool.open()
            self._checkpointer = AsyncPostgresSaver(self._pool)
            await self._checkpointer.setup()
            return self._checkpointer

    async def close(self) -> None:
        async with self._lock:
            if self._pool is not None:
                await self._pool.close()
            self._pool = None
            self._checkpointer = None


_default_runtime: CheckpointerRuntime | None = None
_default_runtime_lock = threading.Lock()


def get_default_checkpointer_runtime() -> CheckpointerRuntime:
    global _default_runtime
    if _default_runtime is None:
        with _default_runtime_lock:
            if _default_runtime is None:
                _default_runtime = CheckpointerRuntime()
    return _default_runtime


async def close_default_checkpointer_runtime() -> None:
    if _default_runtime is not None:
        await _default_runtime.close()
