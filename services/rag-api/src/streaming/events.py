from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    event: str
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

