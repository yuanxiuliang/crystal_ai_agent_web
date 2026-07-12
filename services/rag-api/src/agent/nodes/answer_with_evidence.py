from __future__ import annotations

from ...llm.client import LLMClient
from ..state import GrowthRAGState
from ..utils import error, trace


async def answer_with_evidence(state: GrowthRAGState, llm: LLMClient) -> dict:
    understanding = state["understanding"]
    evidence_pack = state["evidence_pack"]
    if understanding is None or evidence_pack is None:
        return {
            "errors": [
                error("answer_with_evidence", "missing_evidence", "Understanding or evidence is missing.", False)
            ]
        }
    answer = await llm.answer_with_evidence(understanding, evidence_pack, state["long_memories"])
    return {
        "draft_answer": answer,
        "final_answer": answer,
        "trace": [trace("answer_with_evidence", "answered", {"citation_count": len(state["citations"])})],
    }
