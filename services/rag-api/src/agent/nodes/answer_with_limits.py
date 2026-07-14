from __future__ import annotations

from ...llm.client import LLMClient
from ..state import GrowthRAGState
from ..utils import error, trace


async def answer_with_limits(state: GrowthRAGState, llm: LLMClient) -> dict:
    understanding = state["understanding"]
    if understanding is None:
        return {
            "errors": [
                error(
                    "answer_with_limits",
                    "missing_understanding",
                    "Understanding is missing.",
                    False,
                )
            ]
        }
    reason = state["evidence_grade"]["reason"] if state["evidence_grade"] else "未生成证据评估。"
    eligibility = state["prediction_eligibility"]
    if eligibility and not eligibility["eligible"]:
        reason += " " + eligibility["reason"]
    if state["prediction_error"]:
        reason += " 预测回退未完成：" + state["prediction_error"]
    answer = await llm.answer_with_limits(understanding, state["evidence_pack"], reason)
    return {
        "draft_answer": answer,
        "final_answer": answer,
        "citations": [],
        "trace": [trace("answer_with_limits", "answered", {"reason": reason})],
    }
