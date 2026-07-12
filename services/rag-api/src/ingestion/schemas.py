from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedReactant:
    name: str
    role: str
    ratio: float | None = None


@dataclass(frozen=True)
class NormalizedGrowthRecord:
    record_id: str
    formula: str
    doi: str
    method_raw: str
    method_normalized: str
    method_aliases: tuple[str, ...]
    method_zh: str
    reactants: tuple[NormalizedReactant, ...]
    growth: dict[str, Any]
    comment: str
    normalized_text: str = ""
    raw_record: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = {
            "record_id": self.record_id,
            "formula": self.formula,
            "doi": self.doi,
            "growth_method": self.method_raw,
            "growth_method_normalized": self.method_normalized,
            "growth_method_aliases": list(self.method_aliases),
            "growth_method_zh": self.method_zh,
            "comment": self.comment,
            **self.growth,
        }
        return data
