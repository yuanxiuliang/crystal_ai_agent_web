from __future__ import annotations

import asyncio
import hashlib
import re

from ...memory.store import MemoryStore
from ..short_term_policy import (
    is_material_history_request,
    seed_material_history,
    turn_formulas,
    update_material_history,
)
from ..state import GrowthRAGState, MemoryCandidate
from ..utils import (
    assistant_message,
    bounded_active_context,
    clip_text,
    compact_short_memory,
    default_short_memory,
    trace,
)


REMEMBER_RE = re.compile(r"(?:请)?记住|\bremember\b", re.IGNORECASE)
MAX_TEMPERATURE_RE = re.compile(
    r"(?:最高|上限|不超过|不能超过|最多).{0,24}?(\d{2,4}(?:\.\d+)?)\s*(?:°\s*[Cc]|℃|摄氏度|\b[Cc]\b)"
)


async def update_memory(state: GrowthRAGState, store: MemoryStore) -> dict:
    active_context = dict(state["active_context"])
    understanding = state["understanding"]
    if understanding:
        if understanding["formulas"]:
            active_context["active_formulas"] = understanding["formulas"]
            active_context["active_materials"] = understanding["materials"]
        if understanding["growth_methods"]:
            active_context["active_growth_methods"] = understanding["growth_methods"]
        if understanding["constraints"]:
            active_context["active_constraints"] = understanding["constraints"]
        active_context["current_task"] = understanding["task_type"]
    if state["citations"]:
        active_context["last_retrieval_record_ids"] = [
            item["record_id"] for item in state["citations"]
        ]
    active_context = bounded_active_context(active_context, store.limits.active_context_max_items)

    candidates = _explicit_memory_candidates(state)
    writes = []
    for candidate in candidates:
        result = await asyncio.to_thread(
            store.upsert_memory,
            user_id=state["user_id"],
            memory_type=candidate["type"],
            memory_key=candidate["memory_key"],
            content=candidate["content"],
            source=candidate["source"],
            confidence=candidate["confidence"],
            importance=90 if candidate["type"] == "constraint" else 70,
            subject=candidate.get("subject"),
            predicate=candidate.get("predicate"),
            value_json=candidate.get("value_json"),
        )
        writes.append(
            {
                "content": candidate["content"],
                "written": result.written,
                "reason": result.reason,
            }
        )

    messages, conversation_summary = compact_short_memory(
        [*state["messages"], assistant_message(state["final_answer"] or "")],
        state["conversation_summary"],
        max_messages=store.limits.short_max_messages,
        max_summary_chars=store.limits.summary_max_chars,
    )
    short_memory = default_short_memory()
    short_memory.update(state["short_memory"])
    short_memory["conversation_summary"] = conversation_summary
    short_memory["recent_focus"] = clip_text(state["user_message"], 240)
    short_memory["material_history"] = update_material_history(
        seed_material_history(
            short_memory["material_history"],
            messages=state["messages"],
            conversation_summary=state["conversation_summary"],
            max_items=store.limits.session_material_history_max_items,
        ),
        turn_formulas(state),
        evidence_kind=state["selected_evidence_kind"],
        max_items=store.limits.session_material_history_max_items,
    )
    short_memory["last_turn_kind"] = (
        "material_history"
        if is_material_history_request(state["user_message"], state["short_memory"])
        else None
    )
    if understanding:
        slots = dict(short_memory["confirmed_slots"])
        if understanding["formulas"]:
            slots["formulas"] = understanding["formulas"][-store.limits.active_context_max_items :]
        if understanding["growth_methods"]:
            slots["growth_methods"] = understanding["growth_methods"][
                -store.limits.active_context_max_items :
            ]
        short_memory["confirmed_slots"] = slots

    if state["short_term_backend"] == "checkpointer":
        persisted_reason = "checkpointer_persisted"
        short_term_persisted = True
    else:
        persisted = await asyncio.to_thread(
            store.save_session,
            user_id=state["user_id"],
            session_id=state["session_id"],
            messages=messages,
            conversation_summary=conversation_summary,
            active_context=active_context,
            short_memory=short_memory,
        )
        persisted_reason = persisted.reason
        short_term_persisted = persisted.written
    return {
        "active_context": active_context,
        "conversation_summary": conversation_summary,
        "short_memory": short_memory,
        "memory_candidates": candidates,
        "memory_writes": writes,
        "messages": messages,
        "short_term_persisted": short_term_persisted,
        "trace": [
            trace(
                "update_memory",
                "updated",
                {
                    "short_term_updated": short_term_persisted,
                    "long_term_written": any(item["written"] for item in writes),
                    "message_count": len(messages),
                    "summary_chars": len(conversation_summary or ""),
                    "session_reason": persisted_reason,
                },
            )
        ],
    }


def _explicit_memory_candidates(state: GrowthRAGState) -> list[MemoryCandidate]:
    if not REMEMBER_RE.search(state["user_message"]):
        return []

    candidates: list[MemoryCandidate] = []
    understanding = state["understanding"]
    if understanding and understanding["formulas"]:
        for formula in understanding["formulas"][:3]:
            candidates.append(
                {
                    "type": "research_profile",
                    "memory_key": f"material:{formula}",
                    "content": f"用户明确关注材料：{formula}",
                    "source": "explicit_user_request",
                    "confidence": 0.98,
                    "write_policy": "write_now",
                    "subject": "research",
                    "predicate": "material_interest",
                    "value_json": {"formula": formula},
                }
            )

    temperature = MAX_TEMPERATURE_RE.search(state["user_message"])
    if temperature and any(token in state["user_message"] for token in ("炉", "CVT", "设备")):
        candidates.append(
            {
                "type": "constraint",
                "memory_key": "furnace.max_temperature_c",
                "content": f"用户明确实验约束：最高炉温为 {temperature.group(1)} C。",
                "source": "explicit_user_request",
                "confidence": 0.98,
                "write_policy": "write_now",
                "subject": "furnace",
                "predicate": "max_temperature_c",
                "value_json": {"value": float(temperature.group(1)), "unit": "C"},
            }
        )

    if not candidates:
        content = clip_text(_remove_remember_marker(state["user_message"]), 500)
        if content:
            fingerprint = hashlib.sha256(content.lower().encode("utf-8")).hexdigest()[:16]
            candidates.append(
                {
                    "type": "confirmed_fact",
                    "memory_key": f"explicit.{fingerprint}",
                    "content": f"用户明确要求记住：{content}",
                    "source": "explicit_user_request",
                    "confidence": 0.95,
                    "write_policy": "write_now",
                }
            )
    return candidates


def _remove_remember_marker(value: str) -> str:
    return REMEMBER_RE.sub("", value, count=1).strip(" ：:，,。")
