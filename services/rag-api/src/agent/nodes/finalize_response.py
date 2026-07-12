from __future__ import annotations

from ..state import FinalResponse, GrowthRAGState
from ..utils import new_message_id, trace


async def finalize_response(state: GrowthRAGState) -> dict:
    retrieval = None
    if state["retrieval_plan"]:
        retrieval = {
            "query": state["retrieval_plan"]["query_text"],
            "filters": state["retrieval_plan"]["filters"],
            "top_k": state["retrieval_plan"]["top_k"],
            "result_count": len(state["retrieved_records"]),
            "sufficient": state["evidence_grade"]["is_sufficient"] if state["evidence_grade"] else None,
        }
    response: FinalResponse = {
        "message_id": new_message_id("assistant"),
        "session_id": state["session_id"],
        "answer": state["final_answer"] or "",
        "citations": state["citations"],
        "route": state["route"],
        "retrieval": retrieval,
        "memory": {
            "short_term_updated": state["short_term_persisted"],
            "long_term_written": any(item["written"] for item in state["memory_writes"]),
        },
        "errors": state["errors"],
    }
    return {
        "final_response": response,
        "trace": [trace("finalize_response", "finalized", {"has_answer": bool(response["answer"])})],
    }
