from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..agent.graph import GrowthRAGGraph
from ..schemas.chat import ChatStreamRequest
from ..streaming.sse import encode_sse


router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
    graph = GrowthRAGGraph()
    events = graph.stream(request.model_dump())
    return StreamingResponse(encode_sse(events), media_type="text/event-stream")


@router.post("/retrieve")
async def retrieve_debug(request: ChatStreamRequest) -> dict:
    graph = GrowthRAGGraph()
    final = None
    async for event in graph.stream({**request.model_dump(), "options": {**request.options.model_dump(), "force_retrieve": True}}):
        if event.event == "final":
            final = event.data
    return final or {}

