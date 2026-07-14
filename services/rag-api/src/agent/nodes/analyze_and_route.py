from __future__ import annotations

from ...llm.client import LLMClient
from ..prediction_policy import is_candidate_growth_request
from ..short_term_policy import (
    can_inherit_active_formula,
    has_unresolved_material_target_attempt,
    is_material_history_request,
)
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
    if is_material_history_request(state["user_message"], state["short_memory"]):
        return {
            "understanding": {**understanding, "task_type": "summarize", "missing_slots": []},
            "route": {
                "intent": "direct_answer",
                "should_retrieve": False,
                "reason": "用户请求回顾当前会话中已询问的材料。",
                "answer_mode": "direct",
                "required_slots": [],
                "missing_slots": [],
                "confidence": 1.0,
            },
            "active_context": active_context,
            "trace": [
                trace(
                    "analyze_and_route",
                    "analyzed",
                    {"intent": "direct_answer", "task_type": "summarize", "formulas": []},
                )
            ],
        }

    if has_unresolved_material_target_attempt(state["user_message"], understanding["formulas"]):
        active_context["active_formulas"] = []
        active_context["active_materials"] = []
        active_context["active_growth_methods"] = []
        active_context["current_task"] = "clarify"
        return {
            "understanding": {
                **understanding,
                "task_type": "clarify",
                "materials": [],
                "formulas": [],
                "missing_slots": list(
                    dict.fromkeys([*understanding["missing_slots"], "target_material"])
                ),
            },
            "route": {
                "intent": "clarify",
                "should_retrieve": False,
                "reason": "检测到新的材料输入，但化学式无法解析，不能继承上一轮材料。",
                "answer_mode": "ask_clarification",
                "required_slots": ["target_material"],
                "missing_slots": ["target_material"],
                "confidence": understanding["confidence"],
            },
            "active_context": active_context,
            "trace": [
                trace(
                    "analyze_and_route",
                    "analyzed",
                    {"intent": "clarify", "task_type": "clarify", "formulas": []},
                )
            ],
        }

    if understanding["formulas"]:
        active_context["active_formulas"] = understanding["formulas"]
        active_context["active_materials"] = understanding["materials"]
    candidate_formulas = understanding["formulas"] or (
        active_context.get("active_formulas", [])
        if can_inherit_active_formula(state["user_message"], understanding["formulas"])
        else []
    )
    if (
        not route["should_retrieve"]
        and len(candidate_formulas) == 1
        and is_candidate_growth_request(state["user_message"], understanding["normalized_question"])
    ):
        route = {
            "intent": "retrieve",
            "should_retrieve": True,
            "reason": "用户请求候选生长路线；先检索真实记录，证据不足时允许模型回退。",
            "answer_mode": "evidence_grounded",
            "required_slots": ["target_material"],
            "missing_slots": [],
            "confidence": understanding["confidence"],
        }
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
