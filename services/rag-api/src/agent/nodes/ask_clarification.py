from __future__ import annotations

from ..state import GrowthRAGState
from ..utils import trace


async def ask_clarification(state: GrowthRAGState) -> dict:
    missing = state["route"]["missing_slots"] if state["route"] else []
    if "target_material" in missing:
        answer = (
            "请先告诉我目标材料或化学式。你也可以补充最高炉温、气氛、"
            "是否接受助熔剂法或化学气相输运法。"
        )
    else:
        answer = "请补充目标材料、希望比较的生长方法或实验约束，我再进行检索。"
    return {
        "final_answer": answer,
        "trace": [trace("ask_clarification", "answered", {"missing_slots": missing})],
    }

