from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..accounts.dependencies import require_current_account
from ..accounts.store import Account
from ..agent.graph import GrowthRAGGraph
from ..conversations.store import get_default_conversation_store
from ..schemas.chat import ChatStreamRequest
from ..streaming.sse import encode_sse


router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    account: Account = Depends(require_current_account),
) -> StreamingResponse:
    store = get_default_conversation_store()
    session = await asyncio.to_thread(
        store.get_session, user_id=account.id, session_id=request.session_id
    )
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    await asyncio.to_thread(
        store.append_message,
        user_id=account.id,
        session_id=request.session_id,
        role="user",
        content=request.message,
    )

    graph = GrowthRAGGraph()

    async def persisted_events():
        final: dict | None = None
        async for event in graph.stream({**request.model_dump(), "user_id": account.id}):
            if event.event == "final":
                final = event.data
            yield event
        if final and final.get("answer"):
            await asyncio.to_thread(
                store.append_message,
                user_id=account.id,
                session_id=request.session_id,
                role="assistant",
                content=str(final["answer"]),
                response=final,
            )

    # Keep LangGraph progress events observable through every reverse-proxy layer.
    return StreamingResponse(
        encode_sse(persisted_events()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/retrieve")
async def retrieve_debug(
    request: ChatStreamRequest,
    account: Account = Depends(require_current_account),
) -> dict:
    session = await asyncio.to_thread(
        get_default_conversation_store().get_session,
        user_id=account.id,
        session_id=request.session_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    graph = GrowthRAGGraph()
    final = None
    async for event in graph.stream(
        {
            **request.model_dump(),
            "user_id": account.id,
            "options": {
                **request.options.model_dump(),
                "force_retrieve": True,
                "evidence_only": True,
            },
        }
    ):
        if event.event == "final":
            final = event.data
    return final or {}
