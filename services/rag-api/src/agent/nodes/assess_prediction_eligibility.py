from __future__ import annotations

from ..prediction_policy import is_candidate_growth_request
from ..short_term_policy import can_inherit_active_formula
from ..state import GrowthRAGState, PredictionEligibility
from ..utils import trace


def _candidate_formulas(state: GrowthRAGState) -> list[str]:
    understanding = state["understanding"]
    values = list(understanding["formulas"] if understanding else [])
    if not values and can_inherit_active_formula(state["user_message"], values):
        values = list(state["active_context"].get("active_formulas", []))
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


async def assess_prediction_eligibility(state: GrowthRAGState) -> dict:
    if state["runtime"]["evidence_only"]:
        eligibility: PredictionEligibility = {
            "eligible": False,
            "reason": "当前请求属于仅检索模式，不能调用路线预测模型。",
            "formula": None,
        }
        return _result(eligibility)

    outcome = state["retrieval_outcome"]
    if outcome is None or not outcome["fallback_allowed"]:
        eligibility: PredictionEligibility = {
            "eligible": False,
            "reason": "检索结果不允许模型回退。",
            "formula": None,
        }
        return _result(eligibility)

    understanding = state["understanding"]
    question = understanding["normalized_question"] if understanding else state["user_message"]
    if not is_candidate_growth_request(state["user_message"], question):
        eligibility = {
            "eligible": False,
            "reason": "当前问题是在索要文献事实，而不是候选生长路线，不能用模型预测替代文献事实。",
            "formula": None,
        }
        return _result(eligibility)

    formulas = _candidate_formulas(state)
    if len(formulas) != 1:
        eligibility = {
            "eligible": False,
            "reason": "模型回退需要唯一、明确的目标化学式。",
            "formula": None,
        }
        return _result(eligibility)

    eligibility = {
        "eligible": True,
        "reason": "检索已完成但证据不足；问题请求候选路线且存在唯一化学式。",
        "formula": formulas[0],
    }
    return _result(eligibility)


def _result(eligibility: PredictionEligibility) -> dict:
    return {
        "prediction_eligibility": eligibility,
        "trace": [
            trace(
                "assess_prediction_eligibility",
                "assessed",
                {
                    "eligible": eligibility["eligible"],
                    "reason": eligibility["reason"],
                    "formula": eligibility["formula"],
                },
            )
        ],
    }
