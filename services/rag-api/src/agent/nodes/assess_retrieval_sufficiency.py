from __future__ import annotations

import re

from ..prediction_policy import needs_explicit_material_formula
from ..state import EvidenceGrade, GrowthRAGState, RetrievedRecord, RetrievalOutcome
from ..utils import trace


def _formula_key(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "")


def _method_matches(record_method: str | None, expected_method: str | None) -> bool:
    if not expected_method:
        return True
    value = (record_method or "").strip().lower()
    expected = expected_method.strip().lower()
    if expected == "flux":
        return "flux" in value or "助熔剂" in value
    if expected == "cvt":
        return "cvt" in value or "chemical vapor transport" in value or "气相输运" in value
    return value == expected


def _requested_slots(state: GrowthRAGState) -> list[str]:
    plan = state["retrieval_plan"]
    understanding = state["understanding"]
    if plan is None:
        return []
    requested: list[str] = []
    if plan["filters"].get("material_formula"):
        requested.append("material_formula")
    if plan["filters"].get("growth_method"):
        requested.append("growth_method")
    question = understanding["normalized_question"] if understanding else plan["query_text"]
    lowered = question.lower()
    if "温度" in question or "temperature" in lowered:
        requested.append("temperature_program")
    if "气氛" in question or "atmosphere" in lowered:
        requested.append("atmosphere")
    if any(token in question for token in ("原料", "前驱体", "助熔剂", "运输剂")):
        requested.append("precursors")
    return requested


def _has_slot(record: RetrievedRecord, slot: str) -> bool:
    if slot == "material_formula":
        return bool(record.get("material_formula"))
    if slot == "growth_method":
        return bool(record.get("growth_method"))
    if slot == "temperature_program":
        return bool(record.get("temperature_program"))
    if slot == "atmosphere":
        return bool(record.get("atmosphere"))
    if slot == "precursors":
        return bool(record.get("precursors"))
    return False


def _source_is_traceable(record: RetrievedRecord) -> bool:
    return bool(record.get("record_id") and str(record.get("source_text") or "").strip())


async def assess_retrieval_sufficiency(state: GrowthRAGState) -> dict:
    """Select usable literature records with deterministic, field-aware retrieval gates."""
    requested_slots = _requested_slots(state)
    retrieval_error = state["retrieval_error"]
    if retrieval_error is not None:
        outcome: RetrievalOutcome = {
            "status": "unavailable",
            "reason_codes": ["corpus_unavailable"],
            "requested_slots": requested_slots,
            "covered_slots": [],
            "usable_record_ids": [],
            "fallback_allowed": False,
        }
        return _result(outcome, [], "检索服务不可用，无法判断语料中是否存在真实记录。")

    records = state["retrieved_records"]
    if not records:
        outcome = {
            "status": "empty",
            "reason_codes": ["no_candidate"],
            "requested_slots": requested_slots,
            "covered_slots": [],
            "usable_record_ids": [],
            "fallback_allowed": True,
        }
        return _result(outcome, [], "检索已完成，但没有返回可用的真实记录。")

    plan = state["retrieval_plan"]
    assert plan is not None
    reason_codes: list[str] = []
    formula = _formula_key(plan["filters"].get("material_formula"))
    understanding = state["understanding"]
    question = understanding["normalized_question"] if understanding else plan["query_text"]
    if not formula and needs_explicit_material_formula(state["user_message"], question):
        outcome = {
            "status": "invalid_request",
            "reason_codes": ["target_formula_missing"],
            "requested_slots": [*requested_slots, "material_formula"],
            "covered_slots": [],
            "usable_record_ids": [],
            "fallback_allowed": False,
        }
        return _result(outcome, [], "目标材料缺少可解析化学式，不能将相近材料记录当作真实证据。")

    traceable = [record for record in records if _source_is_traceable(record)]
    if not traceable:
        outcome = {
            "status": "insufficient",
            "reason_codes": ["source_metadata_invalid"],
            "requested_slots": requested_slots,
            "covered_slots": [],
            "usable_record_ids": [],
            "fallback_allowed": True,
        }
        return _result(outcome, [], "检索记录缺少可追溯的来源文本或记录标识。")
    if len(traceable) != len(records):
        reason_codes.append("source_metadata_invalid")

    candidates = traceable
    if formula:
        formula_matches = [
            record
            for record in candidates
            if _formula_key(record.get("material_formula")) == formula
        ]
        if not formula_matches:
            outcome = {
                "status": "insufficient",
                "reason_codes": [*reason_codes, "material_mismatch"],
                "requested_slots": requested_slots,
                "covered_slots": [],
                "usable_record_ids": [],
                "fallback_allowed": True,
            }
            return _result(outcome, [], "检索命中记录与目标化学式不一致。")
        candidates = formula_matches

    method = plan["filters"].get("growth_method")
    if method:
        method_matches = [
            record for record in candidates if _method_matches(record.get("growth_method"), method)
        ]
        if not method_matches:
            outcome = {
                "status": "insufficient",
                "reason_codes": [*reason_codes, "method_mismatch"],
                "requested_slots": requested_slots,
                "covered_slots": [],
                "usable_record_ids": [],
                "fallback_allowed": True,
            }
            return _result(outcome, [], "检索记录不满足指定的生长方法。")
        candidates = method_matches

    covered_slots = [
        slot for slot in requested_slots if any(_has_slot(record, slot) for record in candidates)
    ]
    usable = [
        record for record in candidates if all(_has_slot(record, slot) for slot in requested_slots)
    ]
    missing = [slot for slot in requested_slots if slot not in covered_slots]
    if not usable:
        outcome = {
            "status": "insufficient",
            "reason_codes": [*reason_codes, *(f"missing_{slot}" for slot in missing)],
            "requested_slots": requested_slots,
            "covered_slots": covered_slots,
            "usable_record_ids": [],
            "fallback_allowed": True,
        }
        return _result(outcome, [], "检索记录缺少回答当前问题所需的关键字段。")

    outcome = {
        "status": "sufficient",
        "reason_codes": reason_codes,
        "requested_slots": requested_slots,
        "covered_slots": covered_slots,
        "usable_record_ids": [record["record_id"] for record in usable],
        "fallback_allowed": False,
    }
    return _result(outcome, usable, "检索证据满足材料、方法、来源和字段覆盖要求。")


def _result(outcome: RetrievalOutcome, usable: list[RetrievedRecord], reason: str) -> dict:
    sufficient = outcome["status"] == "sufficient"
    grade: EvidenceGrade = {
        "is_sufficient": sufficient,
        "reason": reason,
        "usable_record_ids": outcome["usable_record_ids"],
        "missing_evidence": outcome["reason_codes"],
        "answer_strategy": "single_record"
        if sufficient and len(usable) == 1
        else ("compare_records" if sufficient else "insufficient"),
        "confidence": 0.9 if sufficient else 0.0,
    }
    return {
        "usable_retrieved_records": usable,
        "retrieval_outcome": outcome,
        "evidence_grade": grade,
        "trace": [
            trace(
                "assess_retrieval_sufficiency",
                "assessed",
                {
                    "status": outcome["status"],
                    "reason_codes": outcome["reason_codes"],
                    "usable_record_ids": outcome["usable_record_ids"],
                },
            )
        ],
    }
