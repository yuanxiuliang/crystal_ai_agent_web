from __future__ import annotations

from typing import Any

from ..state import GrowthRAGState
from ..utils import error, trace


def _range_text(value: Any, unit: str) -> str | None:
    if not isinstance(value, dict):
        return None
    bounds = value.get("range_c" if unit == "C" else "range_h")
    if not isinstance(bounds, list) or len(bounds) != 2:
        return None
    return f"{bounds[0]}-{bounds[1]} {unit}"


def _growth_text(route: dict[str, Any]) -> str:
    growth = route.get("growth") if isinstance(route.get("growth"), dict) else {}
    if route.get("method") == "Flux":
        fields = [
            ("起始温度", _range_text(growth.get("T_s"), "C")),
            ("结束温度", _range_text(growth.get("T_e"), "C")),
        ]
    else:
        fields = [
            ("源端温度", _range_text(growth.get("T_src"), "C")),
            ("晶体端温度", _range_text(growth.get("T_crys"), "C")),
        ]
    duration = _range_text(growth.get("dur"), "h")
    if duration:
        fields.append(("时长", duration))
    return "；".join(f"{name} {value}" for name, value in fields if value)


def _reactant_names(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "无"
    names = [str(item.get("name")) for item in value if isinstance(item, dict) and item.get("name")]
    return "、".join(names) if names else "无"


async def answer_from_prediction(state: GrowthRAGState) -> dict:
    prediction = state["prediction_result"]
    outcome = state["retrieval_outcome"]
    if prediction is None or outcome is None:
        return {
            "errors": [
                error(
                    "answer_from_prediction",
                    "missing_prediction",
                    "Prediction result or retrieval outcome is missing.",
                    False,
                )
            ]
        }

    model = prediction.get("model") if isinstance(prediction.get("model"), dict) else {}
    lines = [
        "当前检索已完成，但没有找到足以支持该问题的真实文献或实验记录。",
        (
            "以下是本地路线预测模型给出的可尝试单晶生长方案；它们尚未由当前真实"
            "文献或实验记录验证，不是文献事实，也不能替代实验验证。"
        ),
        (
            f"模型：{model.get('model_id', 'unknown')}@{model.get('model_version', 'unknown')}；"
            f"化学式：{prediction.get('formula_std') or prediction.get('formula', '')}。"
        ),
    ]
    for route in prediction.get("routes", []):
        if not isinstance(route, dict):
            continue
        lines.append(
            f"候选 {route.get('rank')}（{route.get('method')}）："
            f"原料 {_reactant_names(route.get('raw_reactants'))}；"
            f"添加剂 {_reactant_names(route.get('additives'))}；"
            f"{_growth_text(route)}。"
        )
    warnings = [str(item) for item in prediction.get("warnings", []) if item]
    if warnings:
        lines.append("注意事项：" + "；".join(warnings))
    lines.append("检索回退原因：" + "、".join(outcome["reason_codes"]))
    answer = "\n".join(lines)
    return {
        "draft_answer": answer,
        "final_answer": answer,
        "citations": [],
        "trace": [
            trace(
                "answer_from_prediction",
                "answered",
                {"prediction_run_id": prediction.get("prediction_run_id")},
            )
        ],
    }
