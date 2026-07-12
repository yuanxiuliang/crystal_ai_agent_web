from __future__ import annotations

import asyncio

from ..memory.checkpointer import close_default_checkpointer_runtime, get_default_checkpointer_runtime
from ..memory.store import get_memory_store


async def main_async() -> int:
    store = get_memory_store()
    await asyncio.to_thread(store.ensure_schema)
    checkpointer = await get_default_checkpointer_runtime().get()
    print(f"memory_database={store.kind}")
    print(f"short_term_backend={'postgres-checkpointer' if checkpointer is not None else 'bounded-store'}")
    await close_default_checkpointer_runtime()
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
