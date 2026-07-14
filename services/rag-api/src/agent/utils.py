from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .state import (
    ActiveContext,
    GraphError,
    GrowthRAGState,
    Message,
    RuntimeOptions,
    ShortMemory,
    TraceEvent,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_message_id(prefix: str = "msg") -> str:
    return f"{prefix}-{uuid4().hex}"


def trace(node: str, event: str, data: dict[str, Any] | None = None) -> TraceEvent:
    return {"node": node, "event": event, "data": data or {}}


def error(node: str, code: str, message: str, recoverable: bool = True) -> GraphError:
    return {"node": node, "code": code, "message": message, "recoverable": recoverable}


def default_runtime(raw: dict[str, Any] | None = None) -> RuntimeOptions:
    raw = raw or {}
    top_k = int(raw.get("top_k", 12))
    top_k = max(1, min(30, top_k))
    mode = raw.get("retrieval_mode", "hybrid")
    if mode not in {"dense", "sparse", "hybrid"}:
        mode = "hybrid"
    return {
        "force_retrieve": bool(raw.get("force_retrieve", False)),
        "evidence_only": bool(raw.get("evidence_only", False)),
        "top_k": top_k,
        "retrieval_mode": mode,
        "model": raw.get("model"),
        "stream_trace": bool(raw.get("stream_trace", True)),
        "temperature": raw.get("temperature"),
    }


def default_active_context() -> ActiveContext:
    return {
        "active_materials": [],
        "active_formulas": [],
        "active_growth_methods": [],
        "active_constraints": [],
        "last_retrieval_record_ids": [],
        "current_task": None,
    }


def default_short_memory() -> ShortMemory:
    return {
        "conversation_summary": None,
        "recent_focus": None,
        "confirmed_slots": {},
        "open_questions": [],
        "material_history": [],
        "last_turn_kind": None,
    }


def clip_text(value: str, max_chars: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value
    return f"{value[: max(0, max_chars - 3)]}..."


def bounded_active_context(active_context: ActiveContext, max_items: int) -> ActiveContext:
    max_items = max(1, max_items)
    return {
        "active_materials": [
            clip_text(str(item), 120) for item in active_context["active_materials"][-max_items:]
        ],
        "active_formulas": [
            clip_text(str(item), 80) for item in active_context["active_formulas"][-max_items:]
        ],
        "active_growth_methods": [
            clip_text(str(item), 80)
            for item in active_context["active_growth_methods"][-max_items:]
        ],
        "active_constraints": [
            clip_text(str(item), 200) for item in active_context["active_constraints"][-max_items:]
        ],
        "last_retrieval_record_ids": [
            clip_text(str(item), 120)
            for item in active_context["last_retrieval_record_ids"][-max_items:]
        ],
        "current_task": (
            clip_text(str(active_context["current_task"]), 120)
            if active_context["current_task"]
            else None
        ),
    }


def compact_short_memory(
    messages: list[Message],
    previous_summary: str | None,
    *,
    max_messages: int,
    max_summary_chars: int,
) -> tuple[list[Message], str | None]:
    """Keep a fixed recent window and replace, never append, the compacted summary."""
    max_messages = max(2, max_messages)
    if len(messages) <= max_messages:
        return messages, previous_summary

    dropped = messages[:-max_messages]
    summary_lines = [previous_summary] if previous_summary else []
    for message in dropped:
        content = clip_text(message.get("content", ""), 180)
        if content:
            role = "用户" if message.get("role") == "user" else "助手"
            summary_lines.append(f"{role}: {content}")

    summary = "\n".join(line for line in summary_lines if line)
    if len(summary) > max_summary_chars:
        summary = f"...\n{summary[-max(0, max_summary_chars - 5) :]}"
    return messages[-max_messages:], summary or None


def user_message(content: str, message_id: str) -> Message:
    return {
        "role": "user",
        "content": content,
        "message_id": message_id,
        "created_at": utc_now(),
        "metadata": {},
    }


def assistant_message(content: str) -> Message:
    return {
        "role": "assistant",
        "content": content,
        "message_id": new_message_id("assistant"),
        "created_at": utc_now(),
        "metadata": {},
    }


def merge_state(state: GrowthRAGState, patch: dict[str, Any]) -> GrowthRAGState:
    merged = dict(state)
    for key, value in patch.items():
        merged[key] = value
    return merged  # type: ignore[return-value]
