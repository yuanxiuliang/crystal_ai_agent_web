from __future__ import annotations

from ..state import GrowthRAGState
from ..utils import trace


async def compact_persistent_state(state: GrowthRAGState) -> dict:
    """Remove turn-only retrieval artifacts before the next checkpoint is written."""
    return {
        "input_payload": {},
        "memory_query_embedding": None,
        "long_memories": [],
        "memory_candidates": [],
        "memory_writes": [],
        "understanding": None,
        "route": None,
        "retrieval_plan": None,
        "retrieved_records": [],
        "evidence_pack": None,
        "evidence_grade": None,
        "answer_plan": None,
        "draft_answer": None,
        "final_answer": None,
        "citations": [],
        "final_response": None,
        "trace": [trace("compact_persistent_state", "compacted", {"message_count": len(state["messages"])})],
        "errors": [],
    }
