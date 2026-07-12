from __future__ import annotations

from ...llm.client import LLMClient
from ..state import GrowthRAGState
from ..utils import error, trace


async def route_query(state: GrowthRAGState, llm: LLMClient) -> dict:
    understanding = state["understanding"]
    if understanding is None:
        return {
            "errors": [error("route_query", "missing_understanding", "Understanding is missing.", False)]
        }
    route = await llm.route(understanding, state["runtime"]["force_retrieve"])
    return {
        "route": route,
        "trace": [
            trace(
                "route_query",
                "routed",
                {"intent": route["intent"], "should_retrieve": route["should_retrieve"]},
            )
        ],
    }

