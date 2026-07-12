from __future__ import annotations

from ...llm.client import LLMClient
from ..state import GrowthRAGState
from ..utils import trace


async def understand_query(state: GrowthRAGState, llm: LLMClient) -> dict:
    understanding = await llm.understand(
        state["user_message"],
        state["messages"],
        state["long_memories"],
        state["conversation_summary"],
        state["active_context"],
    )
    active_context = dict(state["active_context"])
    if understanding["formulas"]:
        active_context["active_formulas"] = understanding["formulas"]
        active_context["active_materials"] = understanding["materials"]
    return {
        "understanding": understanding,
        "active_context": active_context,
        "trace": [
            trace(
                "understand_query",
                "understood",
                {
                    "task_type": understanding["task_type"],
                    "formulas": understanding["formulas"],
                    "missing_slots": understanding["missing_slots"],
                },
            )
        ],
    }
