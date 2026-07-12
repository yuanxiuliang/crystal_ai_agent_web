from __future__ import annotations

from ...llm.client import LLMClient
from ..state import GrowthRAGState
from ..utils import error, trace


async def answer_direct(state: GrowthRAGState, llm: LLMClient) -> dict:
    understanding = state["understanding"]
    if understanding is None:
        return {
            "errors": [error("answer_direct", "missing_understanding", "Understanding is missing.", False)]
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
