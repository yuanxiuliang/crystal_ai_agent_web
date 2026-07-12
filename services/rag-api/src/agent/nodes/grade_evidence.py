from __future__ import annotations

from ..state import EvidenceGrade, GrowthRAGState
from ..utils import trace


async def grade_evidence(state: GrowthRAGState) -> dict:
    pack = state["evidence_pack"]
    understanding = state["understanding"]
    records = pack["records"] if pack else []
    missing: list[str] = []
    if not records:
        missing.append("retrieval_results")
    if understanding and "温度" in understanding["normalized_question"] and pack:
        if "temperature_program" in pack["missing_fields"]:
            missing.append("temperature_program")

    is_sufficient = bool(records) and not missing
    grade: EvidenceGrade = {
        "is_sufficient": is_sufficient,
        "reason": "检索证据足以支持回答。" if is_sufficient else "检索结果不足或缺少关键字段。",
        "usable_record_ids": [record["record_id"] for record in records],
        "missing_evidence": missing,
        "answer_strategy": "single_record" if is_sufficient and len(records) == 1 else ("compare_records" if is_sufficient else "insufficient"),
        "confidence": 0.84 if is_sufficient else 0.45,
    }
    return {
        "evidence_grade": grade,
        "trace": [trace("grade_evidence", "graded", {"is_sufficient": is_sufficient, "missing": missing})],
    }

