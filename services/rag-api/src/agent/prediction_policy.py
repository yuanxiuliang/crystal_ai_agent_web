from __future__ import annotations


_FACTUAL_EVIDENCE_TERMS = (
    "文献",
    "记录",
    "doi",
    "出处",
    "报道",
    "已验证",
    "温度是多少",
    "条件是什么",
)

_CANDIDATE_GROWTH_TERMS = (
    "方案",
    "路线",
    "推荐",
    "可尝试",
    "尝试",
    "怎么做",
    "如何做",
    "怎么制备",
    "如何制备",
    "怎样制备",
    "制备",
    "怎么合成",
    "如何合成",
    "怎样合成",
    "合成",
    "如何生长",
    "怎么生长",
    "怎样生长",
    "我要长",
    "我想长",
    "想长",
    "我要做",
    "我想做",
    "想做",
    "做单晶",
    "长单晶",
    "推测",
    "预测",
    "给我一个",
    "给一个",
    "生成",
)

_GROWTH_TARGET_TERMS = (
    "生长",
    "单晶",
    "制备",
    "合成",
    "我要长",
    "我想长",
    "想长",
    "我要做",
    "我想做",
    "想做",
    "做单晶",
)


def is_candidate_growth_request(*texts: str | None) -> bool:
    """Whether the user asks for an actionable, unverified growth route."""
    combined = "\n".join(value for value in texts if value).lower()
    if any(term in combined for term in _FACTUAL_EVIDENCE_TERMS):
        return False
    return any(term in combined for term in _CANDIDATE_GROWTH_TERMS)


def is_material_growth_request(*texts: str | None) -> bool:
    """Whether an identified material is being requested for growth work.

    This is intentionally separate from candidate-fallback eligibility. A factual
    question still has to retrieve records, even though it must not use a model
    prediction when those records are insufficient.
    """
    combined = "\n".join(value for value in texts if value).lower()
    return any(term in combined for term in _GROWTH_TARGET_TERMS)


def needs_explicit_material_formula(*texts: str | None) -> bool:
    """Whether this retrieval task needs a material identity before evidence can be selected."""
    combined = "\n".join(value for value in texts if value).lower()
    return any(term in combined for term in _GROWTH_TARGET_TERMS)
