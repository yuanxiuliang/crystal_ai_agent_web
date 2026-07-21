from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response, status

from ..accounts.dependencies import require_current_account
from ..accounts.store import Account
from ..conversations.store import get_default_conversation_store
from ..schemas.conversations import (
    ChatBootstrapRequest,
    ChatBootstrapResponse,
    ChatMessageResponse,
    ChatSessionResponse,
    RenameChatSessionRequest,
)


router = APIRouter()


@router.post("/bootstrap", response_model=ChatBootstrapResponse)
async def bootstrap_chat_workspace(
    body: ChatBootstrapRequest,
    account: Account = Depends(require_current_account),
) -> dict:
    """Return a complete, owner-scoped workspace before the web client mounts its UI."""
    store = get_default_conversation_store()
    sessions = await asyncio.to_thread(store.list_sessions, user_id=account.id)

    if body.requested_session_id:
        active_session = next(
            (session for session in sessions if session["id"] == body.requested_session_id), None
        )
        if active_session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在。")
    else:
        active_session = sessions[0] if sessions else await asyncio.to_thread(
            store.create_session, user_id=account.id
        )
        if not sessions:
            sessions = [active_session]

    messages = await asyncio.to_thread(
        store.list_messages, user_id=account.id, session_id=active_session["id"]
    )
    return {
        "user": {"id": account.id, "email": account.email},
        "sessions": sessions,
        "active_session": active_session,
        "messages": messages,
    }


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    account: Account = Depends(require_current_account),
) -> list[dict[str, str]]:
    return await asyncio.to_thread(
        get_default_conversation_store().list_sessions, user_id=account.id
    )


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(account: Account = Depends(require_current_account)) -> dict[str, str]:
    return await asyncio.to_thread(
        get_default_conversation_store().create_session, user_id=account.id
    )


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
async def rename_session(
    session_id: str,
    body: RenameChatSessionRequest,
    account: Account = Depends(require_current_account),
) -> dict[str, str]:
    try:
        session = await asyncio.to_thread(
            get_default_conversation_store().rename_session,
            user_id=account.id,
            session_id=session_id,
            title=body.title,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在。")
    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    account: Account = Depends(require_current_account),
) -> Response:
    deleted = await asyncio.to_thread(
        get_default_conversation_store().delete_session,
        user_id=account.id,
        session_id=session_id,
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在。")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def list_messages(
    session_id: str,
    account: Account = Depends(require_current_account),
) -> list[dict]:
    store = get_default_conversation_store()
    session = await asyncio.to_thread(store.get_session, user_id=account.id, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在。")
    return await asyncio.to_thread(store.list_messages, user_id=account.id, session_id=session_id)
