from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrowthMethodInfo:
    raw: str
    normalized: str
    aliases: tuple[str, ...]
    zh_name: str


METHOD_ALIASES: dict[str, GrowthMethodInfo] = {
    "cvt": GrowthMethodInfo(
        raw="CVT",
        normalized="chemical vapor transport",
        aliases=("CVT", "chemical vapor transport", "chemical vapour transport", "vapor transport"),
        zh_name="化学气相输运",
    ),
    "flux": GrowthMethodInfo(
        raw="Flux",
        normalized="flux growth",
        aliases=("Flux", "flux growth", "solution growth"),
        zh_name="助熔剂法",
    ),
}


def normalize_method(method: str | None) -> GrowthMethodInfo:
    raw = (method or "").strip()
    key = raw.lower()
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    normalized = raw.lower() if raw else "unknown growth method"
    return GrowthMethodInfo(raw=raw or "unknown", normalized=normalized, aliases=(raw,), zh_name="")
