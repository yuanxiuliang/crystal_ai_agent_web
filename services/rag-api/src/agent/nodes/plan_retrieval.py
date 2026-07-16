from __future__ import annotations

from ..short_term_policy import can_inherit_active_formula
from ..state import GrowthRAGState, RetrievalFilters, RetrievalPlan
from ..utils import error, trace


async def plan_retrieval(state: GrowthRAGState) -> dict:
    understanding = state["understanding"]
    if understanding is None:
        return {
            "errors": [
                error("plan_retrieval", "missing_understanding", "Understanding is missing.", False)
            ]
        }

    active_formulas = state["active_context"].get("active_formulas", [])
    formula = (
        understanding["formulas"][0]
        if understanding["formulas"]
        else (
            active_formulas[0]
            if active_formulas
            and can_inherit_active_formula(state["user_message"], understanding["formulas"])
            else None
        )
    )
    method = understanding["growth_methods"][0] if understanding["growth_methods"] else None
    filters: RetrievalFilters = {
        "material_formula": formula,
        "material_name": None,
        "growth_method": method,
        "temperature_min": None,
        "temperature_max": None,
        "atmosphere": understanding["atmosphere_mentions"][0]
        if understanding["atmosphere_mentions"]
        else None,
        "doi": None,
    }
    query = understanding["normalized_question"]
    plan: RetrievalPlan = {
        "query_kind": "material_record",
        "query_text": query,
        "dense_query": query,
        "sparse_query": " ".join([query, formula or "", method or ""]).strip(),
        "filters": filters,
        "top_k": state["runtime"]["top_k"],
        "retrieval_mode": state["runtime"]["retrieval_mode"],
        "relax_filters_if_empty": True,
        "must_have": ["material_formula"] if formula else [],
        "nice_to_have": ["temperature_program", "growth_method", "atmosphere"],
    }
    return {
        "retrieval_plan": plan,
        "trace": [
            trace(
                "plan_retrieval",
                "planned",
                {"query": plan["query_text"], "filters": plan["filters"], "top_k": plan["top_k"]},
            )
        ],
    }
