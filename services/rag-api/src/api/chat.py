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


def _graph_history(messages: list[dict]) -> list[dict]:
    return [
        {
            "role": message["role"],
            "content": message["content"],
            "message_id": message["id"],
            "created_at": message["created_at"],
            "metadata": {},
        }
        for message in messages
    ]


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
    graph = GrowthRAGGraph()
    if request.replace_message_id:
        retained_messages = await asyncio.to_thread(
            store.replace_user_message_and_truncate,
            user_id=account.id,
            session_id=request.session_id,
            message_id=request.replace_message_id,
            content=request.message,
        )
        if retained_messages is None:
            raise HTTPException(status_code=404, detail="可编辑的用户消息不存在。")
        await graph.reset_session_context(user_id=account.id, session_id=request.session_id)
        history = retained_messages[:-1]
    else:
        history = await asyncio.to_thread(
            store.list_messages,
            user_id=account.id,
            session_id=request.session_id,
        )
        await asyncio.to_thread(
            store.append_message,
            user_id=account.id,
            session_id=request.session_id,
            role="user",
            content=request.message,
        )

    async def persisted_events():
        final: dict | None = None
        async for event in graph.stream(
            {
                **request.model_dump(exclude={"replace_message_id"}),
                "messages": _graph_history(history),
                "user_id": account.id,
            }
        ):
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
