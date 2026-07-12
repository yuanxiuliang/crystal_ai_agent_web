from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankedHit:
    record_id: str
    score: float
    rank: int
    source: str
    entity: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedHit:
    record_id: str
    score: float
    entity: dict[str, Any]
    debug: dict[str, Any]


def reciprocal_rank_fusion(
    result_sets: dict[str, list[RankedHit]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]:
    weights = weights or {}
    fused: dict[str, FusedHit] = {}

    for source, hits in result_sets.items():
        source_weight = weights.get(source, 1.0)
        for hit in hits:
            contribution = source_weight / (rrf_k + hit.rank)
            if hit.record_id not in fused:
                fused[hit.record_id] = FusedHit(
                    record_id=hit.record_id,
                    score=0.0,
                    entity=hit.entity,
                    debug={},
                )
            fused_hit = fused[hit.record_id]
            fused_hit.score += contribution
            if not fused_hit.entity and hit.entity:
                fused_hit.entity = hit.entity
            fused_hit.debug[f"{source}_rank"] = hit.rank
            fused_hit.debug[f"{source}_score"] = hit.score
            fused_hit.debug[f"{source}_rrf"] = contribution

    return sorted(fused.values(), key=lambda hit: hit.score, reverse=True)
