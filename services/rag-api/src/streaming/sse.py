from __future__ import annotations

import json
from typing import AsyncIterator

from .events import StreamEvent


async def encode_sse(events: AsyncIterator[StreamEvent]) -> AsyncIterator[str]:
    async for event in events:
        payload = json.dumps(event.data, ensure_ascii=False)
        yield f"event: {event.event}\ndata: {payload}\n\n"

