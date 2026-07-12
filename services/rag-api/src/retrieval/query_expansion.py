from __future__ import annotations


EXPANSION_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("cvt", "chemical vapor transport", "vapor transport", "气相输运", "化学气相输运"),
        (
            "chemical vapor transport",
            "CVT",
            "vapor transport",
            "source-zone temperature",
            "crystal-growth-zone temperature",
            "transport agent",
        ),
    ),
    (
        ("flux", "助熔剂", "助溶剂", "熔剂"),
        ("flux growth", "solution growth", "flux agent", "solvent", "additive"),
    ),
    (
        ("温度", "temperature", "加热", "降温", "冷却"),
        (
            "growth temperature",
            "source-zone temperature",
            "crystal-growth-zone temperature",
            "start temperature",
            "end temperature",
            "cooling temperature",
        ),
    ),
    (
        ("原料", "反应物", "starting material", "reactant", "precursor"),
        ("starting materials", "reactants", "precursors"),
    ),
    (
        ("添加剂", "传输剂", "输运剂", "transport agent", "additive"),
        ("additive", "transport agent", "flux agent", "solvent"),
    ),
    (
        ("时间", "多久", "duration", "hour", "day"),
        ("growth duration", "duration", "hours", "days"),
    ),
)


def expand_query_for_bm25(query: str) -> str:
    normalized = query.lower()
    terms: list[str] = [query]
    for triggers, additions in EXPANSION_RULES:
        if any(trigger.lower() in normalized for trigger in triggers):
            terms.extend(additions)
    return _join_unique(terms)


def _join_unique(values: list[str]) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = value.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return " ".join(output)
