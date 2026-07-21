from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .auth import CurrentUserResponse


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


class ChatBootstrapRequest(BaseModel):
    """Select the workspace to prepare, or let the API choose the most recent one."""

    requested_session_id: str | None = Field(default=None, min_length=1, max_length=64)


class ChatBootstrapResponse(BaseModel):
    """All authenticated data required before mounting the chat workbench."""

    user: CurrentUserResponse
    sessions: list[ChatSessionResponse]
    active_session: ChatSessionResponse
    messages: list[ChatMessageResponse]
