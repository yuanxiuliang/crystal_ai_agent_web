from __future__ import annotations

from ...llm.client import LLMClient
from ...retrieval.catalog_query import detect_aggregate_query
from ..prediction_policy import is_candidate_growth_request, is_material_growth_request
from ..short_term_policy import (
    can_inherit_active_formula,
    extract_formula_candidates,
    has_unresolved_material_target_attempt,
    is_material_history_request,
)
from ..state import GrowthRAGState
from ..utils import trace


async def analyze_and_route(state: GrowthRAGState, llm: LLMClient) -> dict:
    active_context = dict(state["active_context"])
    aggregate_query = detect_aggregate_query(state["user_message"])
    if aggregate_query is not None:
        methods = (
            ["Flux"]
            if aggregate_query["growth_method"] == "flux growth"
            else (
                ["CVT"]
                if aggregate_query["growth_method"] == "chemical vapor transport"
                else []
            )
        )
        active_context["current_task"] = "aggregate_retrieval"
        return {
            "understanding": {
                "normalized_question": state["user_message"],
                "task_type": "retrieve",
                "materials": [],
                "formulas": [],
                "growth_methods": methods,
                "temperature_mentions": [],
                "atmosphere_mentions": [],
                "precursor_mentions": [item["name"] for item in aggregate_query["reactants"]],
                "constraints": [],
                "missing_slots": [],
                "confidence": 1.0,
            },
            "route": {
                "intent": "retrieve",
                "should_retrieve": True,
                "reason": "检测到元素体系、方法、传输剂或原料的结构化真实记录查询。",
                "answer_mode": "evidence_grounded",
                "required_slots": [],
                "missing_slots": [],
                "confidence": 1.0,
            },
            "aggregate_query": aggregate_query,
            "active_context": active_context,
            "trace": [
                trace(
                    "analyze_and_route",
                    "analyzed",
                    {
                        "intent": "retrieve",
                        "task_type": "retrieve",
                        "aggregate_kind": aggregate_query["kind"],
                        "formulas": [],
                    },
                )
            ],
        }

    understanding, route = await llm.analyze_and_route(
        state["user_message"],
        state["messages"],
        state["long_memories"],
        state["runtime"]["force_retrieve"],
        state["conversation_summary"],
        state["active_context"],
    )
    # A formula that is present in the user message is a deterministic input fact.
    # Keep it when an upstream LLM misses or normalizes it incorrectly, so the
    # retrieval-first policy cannot be bypassed by one routing classification.
    message_formulas = extract_formula_candidates(state["user_message"])
    if message_formulas:
        formulas = list(dict.fromkeys([*understanding["formulas"], *message_formulas]))
        materials = list(dict.fromkeys([*understanding["materials"], *message_formulas]))
        understanding = {
            **understanding,
            "materials": materials,
            "formulas": formulas,
            "missing_slots": [
                slot for slot in understanding["missing_slots"] if slot != "target_material"
            ],
        }
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
            "aggregate_query": None,
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
            "aggregate_query": None,
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
    if len(candidate_formulas) == 1 and (
        is_material_growth_request(state["user_message"], understanding["normalized_question"])
        or is_candidate_growth_request(state["user_message"], understanding["normalized_question"])
    ):
        route = {
            "intent": "retrieve",
            "should_retrieve": True,
            "reason": "检测到明确材料或候选生长请求；先检索真实记录。",
            "answer_mode": "evidence_grounded",
            "required_slots": ["target_material"],
            "missing_slots": [],
            "confidence": understanding["confidence"],
        }
    return {
        "understanding": understanding,
        "route": route,
        "active_context": active_context,
        "aggregate_query": None,
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
