from __future__ import annotations

from ..state import FinalResponse, GrowthRAGState
from ..utils import new_message_id, trace


async def finalize_response(state: GrowthRAGState) -> dict:
    retrieval = None
    if state["retrieval_plan"]:
        retrieval = {
            "query": state["retrieval_plan"]["query_text"],
            "mode": state["retrieval_plan"]["query_kind"],
            "filters": state["retrieval_plan"]["filters"],
            "top_k": state["retrieval_plan"]["top_k"],
            "result_count": len(state["retrieved_records"]),
            "sufficient": state["evidence_grade"]["is_sufficient"]
            if state["evidence_grade"]
            else None,
            "outcome": state["retrieval_outcome"],
        }
    response: FinalResponse = {
        "message_id": new_message_id("assistant"),
        "session_id": state["session_id"],
        "answer": state["final_answer"] or "",
        "citations": state["citations"],
        "evidence_records": (
            state["evidence_pack"]["records"]
            if state["selected_evidence_kind"] == "literature_record" and state["evidence_pack"]
            else []
        ),
        "route": state["route"],
        "retrieval": retrieval,
        "aggregate": state["aggregate_result"],
        "evidence_kind": state["selected_evidence_kind"],
        "prediction": state["prediction_result"],
        "memory": {
            "short_term_updated": state["short_term_persisted"],
            "long_term_written": any(item["written"] for item in state["memory_writes"]),
        },
        "errors": state["errors"],
    }
    return {
        "final_response": response,
        "trace": [
            trace("finalize_response", "finalized", {"has_answer": bool(response["answer"])})
        ],
    }
