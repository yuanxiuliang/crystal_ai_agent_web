from __future__ import annotations

from ..state import EvidenceGrade, GrowthRAGState, RetrievalOutcome
from ..utils import trace


async def assess_aggregate_sufficiency(state: GrowthRAGState) -> dict:
    result = state["aggregate_result"]
    if state["retrieval_error"] is not None:
        outcome: RetrievalOutcome = {
            "status": "unavailable",
            "reason_codes": ["catalog_unavailable"],
            "requested_slots": ["structured_catalog"],
            "covered_slots": [],
            "usable_record_ids": [],
            "fallback_allowed": False,
        }
        return _result(outcome, [], "结构化真实记录目录暂时不可用。")
    if result is None or result["total_records"] == 0:
        outcome = {
            "status": "empty",
            "reason_codes": ["no_exact_catalog_match"],
            "requested_slots": ["structured_catalog"],
            "covered_slots": [],
            "usable_record_ids": [],
            "fallback_allowed": False,
        }
        return _result(outcome, [], "当前语料中没有满足该精确条件的真实记录。")

    usable = [record for record in state["retrieved_records"] if record["record_id"] and record["source_text"]]
    if not usable:
        outcome = {
            "status": "insufficient",
            "reason_codes": ["representative_source_missing"],
            "requested_slots": ["structured_catalog"],
            "covered_slots": [],
            "usable_record_ids": [],
            "fallback_allowed": False,
        }
        return _result(outcome, [], "匹配结果缺少可展示的来源记录。")
    outcome = {
        "status": "sufficient",
        "reason_codes": [],
        "requested_slots": ["structured_catalog"],
        "covered_slots": ["structured_catalog"],
        "usable_record_ids": [record["record_id"] for record in usable],
        "fallback_allowed": False,
    }
    return _result(outcome, usable, "结构化目录返回了可追溯的精确匹配记录。")


def _result(outcome: RetrievalOutcome, usable: list[dict], reason: str) -> dict:
    sufficient = outcome["status"] == "sufficient"
    grade: EvidenceGrade = {
        "is_sufficient": sufficient,
        "reason": reason,
        "usable_record_ids": outcome["usable_record_ids"],
        "missing_evidence": outcome["reason_codes"],
        "answer_strategy": "compare_records" if sufficient else "insufficient",
        "confidence": 0.9 if sufficient else 0.0,
    }
    return {
        "usable_retrieved_records": usable,
        "retrieval_outcome": outcome,
        "evidence_grade": grade,
        "trace": [
            trace(
                "assess_aggregate_sufficiency",
                "assessed",
                {
                    "status": outcome["status"],
                    "reason_codes": outcome["reason_codes"],
                    "usable_record_ids": outcome["usable_record_ids"],
                },
            )
        ],
    }
