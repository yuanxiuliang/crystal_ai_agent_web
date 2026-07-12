from __future__ import annotations

from ...llm.client import LLMClient
from ..state import GrowthRAGState
from ..utils import trace


async def analyze_and_route(state: GrowthRAGState, llm: LLMClient) -> dict:
    understanding, route = await llm.analyze_and_route(
        state["user_message"],
        state["messages"],
        state["long_memories"],
        state["runtime"]["force_retrieve"],
        state["conversation_summary"],
        state["active_context"],
    )
    active_context = dict(state["active_context"])
    if understanding["formulas"]:
        active_context["active_formulas"] = understanding["formulas"]
        active_context["active_materials"] = understanding["materials"]
    return {
        "understanding": understanding,
        "route": route,
        "active_context": active_context,
        "trace": [
            trace(
                "analyze_and_route",
                "analyzed",
                {
                    "intent": route["intent"],
                    "should_retrieve": route["should_retrieve"],
                    "task_type": understanding["task_type"],
                    "formulas": understanding["formulas"],
                    "missing_slots": route["missing_slots"],
                },
            )
        ],
    }
