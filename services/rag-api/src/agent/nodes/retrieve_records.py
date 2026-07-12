from __future__ import annotations

from ...retrieval.service import RetrievalService
from ..state import GrowthRAGState
from ..utils import error, trace


async def retrieve_records(state: GrowthRAGState, retrieval: RetrievalService) -> dict:
    plan = state["retrieval_plan"]
    if plan is None:
        return {"errors": [error("retrieve_records", "missing_plan", "Retrieval plan is missing.", False)]}
    records = await retrieval.retrieve(plan)
    return {
        "retrieved_records": records,
        "trace": [
            trace(
                "retrieve_records",
                "retrieved",
                {"count": len(records), "record_ids": [record["record_id"] for record in records]},
            )
        ],
    }

