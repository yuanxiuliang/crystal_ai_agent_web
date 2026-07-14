from __future__ import annotations

import asyncio

from ...memory.store import MemoryStore
from ..short_term_policy import seed_material_history
from ..state import GrowthRAGState
from ..utils import bounded_active_context, default_active_context, default_short_memory, trace


async def load_context(state: GrowthRAGState, store: MemoryStore) -> dict:
    if state["short_term_backend"] == "checkpointer":
        short_memory = default_short_memory()
        short_memory.update(state["short_memory"])
        # Existing checkpoint sessions predate the material ledger. Recover it once from
        # user-authored recent context and then persist it with the regular checkpoint.
        short_memory["material_history"] = seed_material_history(
            short_memory["material_history"],
            messages=state["messages"],
            conversation_summary=state["conversation_summary"],
            max_items=store.limits.session_material_history_max_items,
        )
        return {
            "messages": state["messages"][-store.limits.short_max_messages :],
            "short_memory": short_memory,
            "trace": [
                trace(
                    "load_context",
                    "restored_from_checkpointer",
                    {
                        "session_id": state["session_id"],
                        "message_count": len(state["messages"]),
                        "summary_chars": len(state["conversation_summary"] or ""),
                    },
                )
            ],
        }

    snapshot = await asyncio.to_thread(store.load_session, state["user_id"], state["session_id"])
    current_message = state["messages"][-1:] if state["messages"] else []
    if snapshot is None:
        return {
            "messages": state["messages"][-store.limits.short_max_messages :],
            "trace": [
                trace(
                    "load_context",
                    "initialized",
                    {"session_id": state["session_id"], "message_count": len(state["messages"])},
                )
            ],
        }

    active_context = default_active_context()
    active_context.update(snapshot.active_context)
    short_memory = default_short_memory()
    short_memory.update(snapshot.short_memory)
    messages = [*snapshot.messages, *current_message][-store.limits.short_max_messages :]
    short_memory["material_history"] = seed_material_history(
        short_memory["material_history"],
        messages=messages,
        conversation_summary=snapshot.conversation_summary,
        max_items=store.limits.session_material_history_max_items,
    )
    return {
        "messages": messages,
        "conversation_summary": snapshot.conversation_summary,
        "active_context": bounded_active_context(
            active_context, store.limits.active_context_max_items
        ),
        "short_memory": short_memory,
        "trace": [
            trace(
                "load_context",
                "loaded",
                {
                    "session_id": state["session_id"],
                    "message_count": len(messages),
                    "summary_chars": len(snapshot.conversation_summary or ""),
                },
            )
        ],
    }
