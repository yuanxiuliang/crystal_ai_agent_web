from __future__ import annotations

import asyncio

from ...retrieval.fact_catalog import FactCatalog
from ..state import GrowthRAGState
from ..utils import error, trace


async def retrieve_aggregate_records(state: GrowthRAGState, catalog: FactCatalog) -> dict:
    query = state["aggregate_query"]
    if query is None:
        return {
            "errors": [
                error(
                    "retrieve_aggregate_records",
                    "missing_aggregate_query",
                    "Aggregate query is missing.",
                    False,
                )
            ]
        }
    try:
        result = await asyncio.to_thread(catalog.aggregate, query)
    except Exception as exc:  # noqa: BLE001 - distinguishes a catalog outage from no matches.
        retrieval_error = error(
            "retrieve_aggregate_records",
            "catalog_unavailable",
            f"Structured catalog failed: {type(exc).__name__}",
            True,
        )
        return {
            "aggregate_result": None,
            "retrieved_records": [],
            "retrieval_error": retrieval_error,
            "errors": [retrieval_error],
            "trace": [
                trace("retrieve_aggregate_records", "unavailable", {"error": type(exc).__name__})
            ],
        }
    records = result["representatives"]
    return {
        "aggregate_result": result,
        "retrieved_records": records,
        "retrieval_error": None,
        "trace": [
            trace(
                "retrieve_aggregate_records",
                "retrieved",
                {
                    "kind": query["kind"],
                    "records": result["total_records"],
                    "formulas": result["total_formulas"],
                    "dois": result["total_dois"],
                    "representative_ids": [record["record_id"] for record in records],
                },
            )
        ],
    }
