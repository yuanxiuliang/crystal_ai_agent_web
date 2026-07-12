from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ..config import Settings, settings
from ..retrieval.embedding import EmbeddingClient, get_default_embedding_client
from .checkpointer import CheckpointerRuntime
from .store import MemoryLimits, MemoryStore


@dataclass
class MemoryWorker:
    store: MemoryStore
    embedding_client: EmbeddingClient
    poll_seconds: int
    checkpointer_runtime: CheckpointerRuntime | None = None
    _last_checkpoint_cleanup: float = field(default=0.0, init=False)

    @classmethod
    def from_settings(cls, config: Settings = settings) -> "MemoryWorker":
        return cls(
            store=MemoryStore(config.memory_database_url, MemoryLimits.from_settings(config)),
            embedding_client=get_default_embedding_client(config),
            poll_seconds=max(1, config.memory_worker_poll_seconds),
            checkpointer_runtime=CheckpointerRuntime(config),
        )

    async def run_once(self, limit: int = 4) -> int:
        await self._cleanup_expired_checkpoint_threads()
        jobs = await asyncio.to_thread(self.store.claim_memory_jobs, limit)
        for job in jobs:
            try:
                await self._run_job(job)
            except Exception as exc:  # noqa: BLE001 - a failed job must not stop the worker loop.
                await asyncio.to_thread(self.store.fail_memory_job, job["id"], f"{type(exc).__name__}: {exc}")
            else:
                await asyncio.to_thread(self.store.complete_memory_job, job["id"])
        return len(jobs)

    async def close(self) -> None:
        if self.checkpointer_runtime is not None:
            await self.checkpointer_runtime.close()

    async def run_forever(self) -> None:
        while True:
            processed = await self.run_once()
            if processed == 0:
                await asyncio.sleep(self.poll_seconds)

    async def _run_job(self, job: dict) -> None:
        if job["job_type"] != "embed_memory":
            raise RuntimeError(f"Unsupported memory job type: {job['job_type']}")
        memory_id = job.get("memory_id")
        if not memory_id:
            raise RuntimeError("embed_memory job has no memory_id")
        content = await asyncio.to_thread(self.store.get_memory_content, memory_id)
        if not content:
            return
        embedding = await asyncio.to_thread(self.embedding_client.embed_query, content)
        result = await asyncio.to_thread(self.store.store_memory_embedding, memory_id, embedding)
        if not result.written:
            raise RuntimeError(result.reason)

    async def _cleanup_expired_checkpoint_threads(self) -> None:
        runtime = self.checkpointer_runtime
        if runtime is None or not runtime.enabled:
            return
        now = time.monotonic()
        if now - self._last_checkpoint_cleanup < self.store.limits.cleanup_interval_seconds:
            return
        checkpointer = await runtime.get()
        if checkpointer is None:
            return
        thread_ids = await asyncio.to_thread(self.store.take_expired_checkpoint_threads)
        for thread_id in thread_ids:
            await checkpointer.adelete_thread(thread_id)
        self._last_checkpoint_cleanup = now
