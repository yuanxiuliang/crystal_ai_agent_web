from __future__ import annotations

from ...retrieval.service import RetrievalService
from ..state import GrowthRAGState
from ..utils import error, trace


async def retrieve_records(state: GrowthRAGState, retrieval: RetrievalService) -> dict:
    plan = state["retrieval_plan"]
    if plan is None:
        return {
            "errors": [
                error("retrieve_records", "missing_plan", "Retrieval plan is missing.", False)
            ]
        }
    try:
        records = await retrieval.retrieve(plan)
    except Exception as exc:  # noqa: BLE001 - distinguish an outage from an empty corpus result.
        retrieval_error = error(
            "retrieve_records",
            "retrieval_unavailable",
            f"Retrieval service failed: {type(exc).__name__}",
            True,
        )
        return {
            "retrieved_records": [],
            "retrieval_error": retrieval_error,
            "errors": [retrieval_error],
            "trace": [trace("retrieve_records", "unavailable", {"error": type(exc).__name__})],
        }
    return {
        "retrieved_records": records,
        "retrieval_error": None,
        "trace": [
            trace(
                "retrieve_records",
                "retrieved",
                {"count": len(records), "record_ids": [record["record_id"] for record in records]},
            )
        ],
    }
