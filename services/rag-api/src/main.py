from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.chat import router as chat_router
from .config import settings
from .memory.checkpointer import close_default_checkpointer_runtime, get_default_checkpointer_runtime
from .memory.store import get_memory_store


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast on the ThinkPad PostgreSQL profile instead of discovering a missing
    # checkpointer during the first user request. SQLite fallback returns None here.
    await get_default_checkpointer_runtime().get()
    try:
        yield
    finally:
        await close_default_checkpointer_runtime()


app = FastAPI(title="AgentWeb RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/rag/health")
async def health() -> dict[str, str]:
    checkpointer_runtime = get_default_checkpointer_runtime()
    return {
        "status": "ok",
        "service": "rag-api",
        "memory_database": get_memory_store().kind,
        "short_term_backend": "postgres-checkpointer"
        if checkpointer_runtime.enabled
        else "bounded-store",
    }


app.include_router(chat_router, prefix="/api/rag", tags=["rag"])
