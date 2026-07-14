from __future__ import annotations

from ...llm.client import LLMClient
from ..short_term_policy import is_material_history_request, material_history_answer
from ..state import GrowthRAGState
from ..utils import error, trace


async def answer_direct(state: GrowthRAGState, llm: LLMClient) -> dict:
    understanding = state["understanding"]
    if understanding is None:
        return {
            "errors": [
                error("answer_direct", "missing_understanding", "Understanding is missing.", False)
            ]
        }
    if is_material_history_request(state["user_message"], state["short_memory"]):
        answer = material_history_answer(
            state["short_memory"],
            is_follow_up=state["short_memory"].get("last_turn_kind") == "material_history",
        )
        return {
            "final_answer": answer,
            "trace": [trace("answer_direct", "answered_from_short_memory", {"chars": len(answer)})],
        }
    answer = await llm.answer_direct(
        state["user_message"],
        understanding,
        state["messages"],
        state["long_memories"],
        state["conversation_summary"],
        state["active_context"],
    )
    return {
        "final_answer": answer,
        "trace": [trace("answer_direct", "answered", {"chars": len(answer)})],
    }
