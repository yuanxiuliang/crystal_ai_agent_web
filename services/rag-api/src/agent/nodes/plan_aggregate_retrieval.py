from __future__ import annotations

from ..state import GrowthRAGState, RetrievalFilters, RetrievalPlan
from ..utils import error, trace


async def plan_aggregate_retrieval(state: GrowthRAGState) -> dict:
    query = state["aggregate_query"]
    if query is None:
        return {
            "errors": [
                error(
                    "plan_aggregate_retrieval",
                    "missing_aggregate_query",
                    "Aggregate query is missing.",
                    False,
                )
            ]
        }
    filters: RetrievalFilters = {
        "material_formula": None,
        "material_name": None,
        "growth_method": query["growth_method"],
        "temperature_min": None,
        "temperature_max": None,
        "atmosphere": None,
        "doi": None,
    }
    plan: RetrievalPlan = {
        "query_kind": "aggregate_fact",
        "query_text": state["understanding"]["normalized_question"]
        if state["understanding"]
        else state["user_message"],
        "dense_query": "",
        "sparse_query": "",
        "filters": filters,
        "top_k": state["runtime"]["top_k"],
        "retrieval_mode": "sparse",
        "relax_filters_if_empty": False,
        "must_have": [],
        "nice_to_have": ["growth_method", "precursors", "temperature_program"],
    }
    return {
        "retrieval_plan": plan,
        "trace": [
            trace(
                "plan_aggregate_retrieval",
                "planned",
                {"kind": query["kind"], "label": query["label"], "filters": filters},
            )
        ],
    }
