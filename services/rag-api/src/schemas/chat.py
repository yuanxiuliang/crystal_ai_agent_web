from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RetrievalMode = Literal["dense", "sparse", "hybrid"]


class ChatOptions(BaseModel):
    force_retrieve: bool = False
    top_k: int = Field(default=12, ge=1, le=30)
    retrieval_mode: RetrievalMode = "hybrid"
    model: str | None = None
    stream_trace: bool = True
    temperature: float | None = None


class ChatStreamRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message_id: str | None = None
    replace_message_id: str | None = Field(default=None, min_length=1, max_length=64)
    message: str
    options: ChatOptions = Field(default_factory=ChatOptions)
