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
    aggregate_query = state["aggregate_query"]
    if aggregate_query is not None:
        answer = (
            "## 真实记录统计\n\n"
            f"当前知识库中没有找到满足 **{aggregate_query['label']}** 的严格结构化真实记录。\n\n"
            "本次查询只接受目标化学式元素、标准化方法、原料名称和原料角色的精确匹配；"
            "不会以语义相近记录替代事实，也不会调用单晶生长路线预测模型。\n\n"
            "你可以改为查询更宽的元素体系、明确原料角色（起始原料、助熔剂或传输剂），"
            "或指定是否接受含该元素/试剂的扩展匹配。"
        )
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "citations": [],
            "trace": [
                trace(
                    "answer_with_limits",
                    "answered_aggregate_limit",
                    {"kind": aggregate_query["kind"], "reason": reason},
                )
            ],
        }
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
