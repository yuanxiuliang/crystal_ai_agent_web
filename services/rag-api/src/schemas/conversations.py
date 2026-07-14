from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class RenameChatSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=72)


class ChatMessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    response: dict[str, Any] | None = None
