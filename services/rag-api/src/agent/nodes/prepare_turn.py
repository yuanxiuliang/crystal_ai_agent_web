from __future__ import annotations

from typing import Any

from ...config import settings
from ..state import GrowthRAGState
from ..utils import (
    default_active_context,
    default_runtime,
    default_short_memory,
    error,
    new_message_id,
    trace,
    user_message,
)


async def prepare_turn(
    payload: dict[str, Any], existing_state: dict[str, Any] | None = None
) -> GrowthRAGState:
    message = str(payload.get("message", "")).strip()
    message_id = payload.get("message_id") or new_message_id("user")
    runtime = default_runtime(payload.get("options"))
    history = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    short_term_backend = payload.get("_short_term_backend", "store")
    if short_term_backend not in {"store", "checkpointer"}:
        short_term_backend = "store"

    errors = []
    if not message:
        errors.append(error("prepare_turn", "empty_message", "User message is empty.", False))
    elif len(message) > settings.memory_message_max_chars:
        errors.append(
            error(
                "prepare_turn",
                "message_too_long",
                f"User message exceeds the {settings.memory_message_max_chars}-character limit.",
                False,
            )
        )
    restored_messages = []
    if short_term_backend == "checkpointer" and existing_state:
        restored_messages = existing_state.get("messages") or []
    messages = (
        [*restored_messages, user_message(message, message_id)] if message else restored_messages
    )
    if not restored_messages:
        messages = [*history[-12:], user_message(message, message_id)] if message else history[-12:]

    restored_summary = existing_state.get("conversation_summary") if existing_state else None
    restored_context = existing_state.get("active_context") if existing_state else None
    restored_short_memory = existing_state.get("short_memory") if existing_state else None

    return {
        "user_id": str(payload.get("user_id") or "demo-user"),
        "session_id": str(payload.get("session_id") or "demo-session"),
        "message_id": message_id,
        "user_message": message,
        "runtime": runtime,
        "short_term_backend": short_term_backend,
        "messages": messages,
        "conversation_summary": restored_summary if restored_messages else None,
        "active_context": restored_context if restored_messages else default_active_context(),
        "short_memory": restored_short_memory if restored_messages else default_short_memory(),
        "long_memories": [],
        "memory_candidates": [],
        "memory_writes": [],
        "short_term_persisted": False,
        "understanding": None,
        "route": None,
        "retrieval_plan": None,
        "aggregate_query": None,
        "aggregate_result": None,
        "retrieved_records": [],
        "usable_retrieved_records": [],
        "retrieval_error": None,
        "evidence_pack": None,
        "evidence_grade": None,
        "retrieval_outcome": None,
        "prediction_eligibility": None,
        "prediction_result": None,
        "prediction_error": None,
        "selected_evidence_kind": None,
        "draft_answer": None,
        "final_answer": None,
        "citations": [],
        "final_response": None,
        "trace": [trace("prepare_turn", "prepared", {"message_id": message_id})],
        "errors": errors,
    }
