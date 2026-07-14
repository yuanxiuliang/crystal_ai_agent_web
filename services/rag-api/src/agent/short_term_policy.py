from __future__ import annotations

import re
from typing import Any

from .state import Message, SessionMaterial, ShortMemory


_FORMULA_RE = re.compile(r"(?<![A-Za-z])(?:[A-Z][a-z]?\d*){2,}(?![A-Za-z])")
_TARGET_ACTION_RE = re.compile(
    r"(?:我要|我想|想|帮我|请|如何|怎么)?(?:生长|制备|合成|长)(?P<target>[^，。！？；,.;!?]{0,64})"
)
_TARGET_STARTS_WITH_ASCII_RE = re.compile(r"^\s*(?:纯\s*)?[A-Za-z](?:[A-Za-z0-9 ]{0,32})?")
_REFERENCE_TERMS = (
    "它",
    "这个",
    "该材料",
    "此材料",
    "上述",
    "上面",
    "前面",
    "刚才",
    "同一材料",
    "同一个",
    "那你推测",
    "那就推测",
    "继续推测",
    "再给一个方案",
    "再推荐",
)
_HISTORY_TERMS = ("我问过", "我询问过", "我提到过", "我讨论过", "我看过")
_MATERIAL_TERMS = ("单晶样品", "单晶材料", "样品", "材料")
_HISTORY_FOLLOW_UP_TERMS = (
    "没有其他",
    "没有别的",
    "还有其他",
    "还有别的",
    "还有吗",
    "就这些",
    "全部",
    "都有哪些",
)


def has_unresolved_material_target_attempt(message: str, formulas: list[str]) -> bool:
    """Detect a new malformed formula after a growth action without guessing its identity."""
    if extract_formula_candidates(message):
        return False
    for match in _TARGET_ACTION_RE.finditer(message):
        raw_target = _TARGET_STARTS_WITH_ASCII_RE.search(match.group("target"))
        if raw_target is None:
            continue
        target_token_raw = raw_target.group(0).strip()
        if any(character.isspace() for character in target_token_raw):
            return True
        target_token = target_token_raw
        normalized_formulas = {"".join(formula.split()) for formula in formulas if formula}
        if target_token not in normalized_formulas:
            return True
    return False


def can_inherit_active_formula(message: str, formulas: list[str]) -> bool:
    """Only resolve a material from context for an explicit anaphoric follow-up."""
    if formulas or has_unresolved_material_target_attempt(message, formulas):
        return False
    normalized = "".join(message.split())
    return any(term in normalized for term in _REFERENCE_TERMS)


def is_material_history_request(message: str, short_memory: ShortMemory) -> bool:
    normalized = "".join(message.split())
    asks_about_materials = any(term in normalized for term in _MATERIAL_TERMS)
    if asks_about_materials and any(term in normalized for term in _HISTORY_TERMS):
        return True
    if short_memory.get("last_turn_kind") == "material_history":
        return any(term in normalized for term in _HISTORY_FOLLOW_UP_TERMS)
    return False


def extract_formula_candidates(value: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in _FORMULA_RE.finditer(value)))


def seed_material_history(
    existing: list[SessionMaterial] | list[dict[str, Any]] | list[str],
    *,
    messages: list[Message],
    conversation_summary: str | None,
    max_items: int,
) -> list[SessionMaterial]:
    """Migrate legacy sessions by recovering formulas from user-authored short context."""
    history = _normalize_history(existing)
    seed_texts = _summary_user_texts(conversation_summary)
    seed_texts.extend(message["content"] for message in messages if message.get("role") == "user")
    for text in seed_texts:
        history = update_material_history(
            history,
            extract_formula_candidates(text),
            evidence_kind=None,
            max_items=max_items,
        )
    return history


def update_material_history(
    existing: list[SessionMaterial] | list[dict[str, Any]] | list[str],
    formulas: list[str],
    *,
    evidence_kind: str | None,
    max_items: int,
) -> list[SessionMaterial]:
    history = _normalize_history(existing)
    by_formula = {item["formula"]: index for index, item in enumerate(history)}
    for formula in formulas:
        value = formula.strip()
        if not value:
            continue
        index = by_formula.get(value)
        if index is None:
            by_formula[value] = len(history)
            history.append({"formula": value, "evidence_kind": evidence_kind})
        elif evidence_kind:
            history[index] = {**history[index], "evidence_kind": evidence_kind}
    return history[-max(1, max_items) :]


def turn_formulas(state: dict[str, Any]) -> list[str]:
    values: list[str] = []
    understanding = state.get("understanding") or {}
    values.extend(str(item) for item in understanding.get("formulas", []) if item)

    plan = state.get("retrieval_plan") or {}
    filters = plan.get("filters") if isinstance(plan, dict) else None
    if isinstance(filters, dict) and filters.get("material_formula"):
        values.append(str(filters["material_formula"]))

    prediction = state.get("prediction_result") or {}
    if isinstance(prediction, dict):
        values.extend(
            str(prediction[key]) for key in ("formula_std", "formula") if prediction.get(key)
        )
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def material_history_answer(short_memory: ShortMemory, *, is_follow_up: bool) -> str:
    history = _normalize_history(short_memory.get("material_history", []))
    if not history:
        return "当前会话尚未记录到可识别的单晶材料或化学式。"

    if is_follow_up:
        heading = f"当前会话已保留 {len(history)} 个可识别材料，未发现其他已询问的化学式："
    else:
        heading = "本次会话中，您询问过以下单晶材料："
    lines = [heading]
    for item in history:
        label = {
            "literature_record": "真实记录回答",
            "model_prediction": "模型预测候选",
        }.get(item.get("evidence_kind"), "已提出，尚未形成有效方案")
        lines.append(f"- **{item['formula']}**（{label}）")
    return "\n".join(lines)


def _summary_user_texts(summary: str | None) -> list[str]:
    if not summary:
        return []
    return [
        line.removeprefix("用户:").strip()
        for line in summary.splitlines()
        if line.startswith("用户:")
    ]


def _normalize_history(
    value: list[SessionMaterial] | list[dict[str, Any]] | list[str],
) -> list[SessionMaterial]:
    normalized: list[SessionMaterial] = []
    known: set[str] = set()
    for raw in value:
        if isinstance(raw, str):
            formula = raw.strip()
            evidence_kind = None
        elif isinstance(raw, dict):
            formula = str(raw.get("formula") or "").strip()
            raw_kind = raw.get("evidence_kind")
            evidence_kind = (
                str(raw_kind) if raw_kind in {"literature_record", "model_prediction"} else None
            )
        else:
            continue
        if formula and formula not in known:
            known.add(formula)
            normalized.append({"formula": formula, "evidence_kind": evidence_kind})
    return normalized
