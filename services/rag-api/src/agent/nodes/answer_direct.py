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
    try:
        answer = await llm.answer_direct(
            state["user_message"],
            understanding,
            state["messages"],
            state["long_memories"],
            state["conversation_summary"],
            state["active_context"],
        )
        trace_event = "answered"
    except Exception:  # noqa: BLE001 - direct chat remains available during transient LLM outages.
        answer = (
            "当前生成式回答服务暂时不可用，因此我不能给出未经验证的解释。"
            "本次对话中的明确偏好或约束仍会按记忆规则处理；请稍后重试。"
        )
        trace_event = "llm_unavailable_fallback"
    return {
        "final_answer": answer,
        "trace": [trace("answer_direct", trace_event, {"chars": len(answer)})],
    }
