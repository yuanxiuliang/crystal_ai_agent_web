from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .accounts.store import get_default_account_store
from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.conversations import router as conversations_router
from .api.prediction import router as prediction_router
from .config import settings
from .memory.checkpointer import (
    close_default_checkpointer_runtime,
    get_default_checkpointer_runtime,
)
from .memory.store import get_memory_store
from .conversations.store import get_default_conversation_store


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast on the ThinkPad PostgreSQL profile instead of discovering a missing
    # checkpointer during the first user request. SQLite fallback returns None here.
    await get_default_checkpointer_runtime().get()
    await asyncio.to_thread(get_default_account_store().ensure_schema)
    await asyncio.to_thread(get_default_conversation_store().ensure_schema)
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
        "prediction": "enabled" if settings.prediction_enabled else "disabled",
    }


app.include_router(chat_router, prefix="/api/rag", tags=["rag"])
app.include_router(prediction_router, prefix="/api/rag", tags=["rag"])
app.include_router(conversations_router, prefix="/api/rag", tags=["rag"])
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
