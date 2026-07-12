from __future__ import annotations

import asyncio

from ...memory.store import MemoryStore
from ..state import GrowthRAGState
from ..utils import trace


async def load_long_memory(state: GrowthRAGState, store: MemoryStore) -> dict:
    hints = [
        *state["active_context"]["active_formulas"],
        *state["active_context"]["active_materials"],
        *state["active_context"]["active_growth_methods"],
    ]
    query_embedding = state.get("memory_query_embedding")
    memories = await asyncio.to_thread(
        store.load_long_memories,
        user_id=state["user_id"],
        query=state["user_message"],
        context_hints=hints,
        query_embedding=query_embedding,
    )
    return {
        "long_memories": memories,
        "trace": [
            trace(
                "load_long_memory",
                "loaded",
                {
                    "count": len(memories),
                    "char_count": sum(len(item["content"]) for item in memories),
                },
            )
        ],
    }
