from __future__ import annotations

import re

from ..agent.state import (
    ActiveContext,
    EvidencePack,
    EvidenceRecord,
    LongMemoryItem,
    Message,
    RouteDecision,
    UserUnderstanding,
)


# Chinese text is a Unicode ``\w`` character, so word boundaries do not separate
# ``Mn3GaN怎么做``. Restrict boundaries to Latin letters instead.
FORMULA_RE = re.compile(r"(?<![A-Za-z])(?=[A-Za-z0-9]*[a-z])(?:[A-Z][a-z]?\d*){2,}(?![A-Za-z])")
TEMPERATURE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°?\s*c|℃)", re.IGNORECASE)
SOURCE_TEMPERATURE_RE = re.compile(
    r"source(?:[- ]zone)?(?: temperature)?(?: was)?\s*(\d+(?:\.\d+)?)\s*(?:°?\s*c|℃)",
    re.IGNORECASE,
)
CRYSTAL_TEMPERATURE_RE = re.compile(
    r"crystal(?:[- ]growth)?(?:[- ]zone)?(?: temperature)?(?: was)?\s*(\d+(?:\.\d+)?)\s*(?:°?\s*c|℃)",
    re.IGNORECASE,
)
DURATION_RE = re.compile(
    r"(?:growth )?duration(?: was)?\s*(\d+(?:\.\d+)?)\s*h",
    re.IGNORECASE,
)


def _fact_values(record: EvidenceRecord, prefix: str) -> list[str]:
    return [fact[len(prefix) :].strip() for fact in record["key_facts"] if fact.startswith(prefix)]


def _numeric_values(pattern: re.Pattern[str], values: list[str]) -> list[float]:
    result: list[float] = []
    for value in values:
        result.extend(float(match.group(1)) for match in pattern.finditer(value))
    return result


def _format_number(value: float) -> str:
    return f"{value:g}"


def _format_range(values: list[float], unit: str) -> str | None:
    if not values:
        return None
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return f"{_format_number(minimum)} {unit}"
    return f"{_format_number(minimum)}-{_format_number(maximum)} {unit}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _method_summary(records: list[EvidenceRecord]) -> str:
    methods = _unique([record["growth_method"] or "未提供" for record in records])
    if len(methods) == 1:
        return f"所有可用记录均采用 **{methods[0]}**。"
    return "记录中出现的方法包括 **" + "**、**".join(methods) + "**。"


def _temperature_summary(records: list[EvidenceRecord]) -> list[str]:
    programs = [
        value
        for record in records
        for value in _fact_values(record, "temperature program:")
    ]
    source_temperatures = _numeric_values(SOURCE_TEMPERATURE_RE, programs)
    crystal_temperatures = _numeric_values(CRYSTAL_TEMPERATURE_RE, programs)
    durations = _numeric_values(DURATION_RE, programs)
    all_temperatures = _numeric_values(TEMPERATURE_RE, programs)

    lines: list[str] = []
    source_range = _format_range(source_temperatures, "C")
    crystal_range = _format_range(crystal_temperatures, "C")
    duration_range = _format_range(durations, "h")
    if source_range or crystal_range:
        fields = []
        if source_range:
            fields.append(f"源区 **{source_range}**")
        if crystal_range:
            fields.append(f"晶体生长区 **{crystal_range}**")
        lines.append("双温区条件覆盖：" + "；".join(fields) + "。")
    elif all_temperatures:
        lines.append(f"记录给出的温度覆盖 **{_format_range(all_temperatures, 'C')}**。")
    if duration_range:
        lines.append(f"已报告的生长时长覆盖 **{duration_range}**。")
    if not lines:
        lines.append("当前可用记录未提供可汇总的温度程序或生长时长。")
    return lines


def _representative_conditions(records: list[EvidenceRecord], limit: int = 4) -> list[str]:
    conditions: list[str] = []
    for record in records:
        programs = _fact_values(record, "temperature program:")
        if not programs:
            continue
        reference = record["doi"] or record["record_id"]
        method = record["growth_method"] or "未提供方法"
        conditions.append(f"- **{method}**：{programs[0]}（{reference}）")
        if len(conditions) >= limit:
            break
    return conditions


def _evidence_boundaries(evidence_pack: EvidencePack) -> list[str]:
    records = evidence_pack["records"]
    missing: list[str] = []
    if not any(_fact_values(record, "atmosphere:") for record in records):
        missing.append("气氛或真空条件")
    if not any(_fact_values(record, "precursors:") for record in records):
        missing.append("原料配比、传输剂或助熔剂")
    if evidence_pack["missing_fields"]:
        missing.extend(evidence_pack["missing_fields"])
    labels = {
        "temperature_program": "完整温度程序",
        "atmosphere": "气氛或真空条件",
        "precursors": "原料配比、传输剂或助熔剂",
    }
    return _unique([labels.get(item, item) for item in missing])


class MockLLMClient:
    async def analyze_and_route(
        self,
        user_message: str,
        messages: list[Message],
        long_memories: list[LongMemoryItem],
        force_retrieve: bool,
        conversation_summary: str | None = None,
        active_context: ActiveContext | None = None,
    ) -> tuple[UserUnderstanding, RouteDecision]:
        understanding = await self.understand(
            user_message, messages, long_memories, conversation_summary, active_context
        )
        route = await self.route(understanding, force_retrieve)
        return understanding, route

    async def understand(
        self,
        user_message: str,
        messages: list[Message],
        long_memories: list[LongMemoryItem],
        conversation_summary: str | None = None,
        active_context: ActiveContext | None = None,
    ) -> UserUnderstanding:
        formulas = FORMULA_RE.findall(user_message)
        lowered = user_message.lower()
        methods: list[str] = []
        if "flux" in lowered or "助熔剂" in user_message:
            methods.append("Flux")
        if "cvt" in lowered or "化学气相输运" in user_message:
            methods.append("CVT")

        temperature_mentions = re.findall(r"\d+(?:\.\d+)?\s*(?:°c|c|k|℃)", lowered)
        atmosphere_mentions = [
            item
            for item in ["argon", "ar", "nitrogen", "n2", "vacuum", "air"]
            if re.search(rf"\b{re.escape(item)}\b", lowered)
        ]
        atmosphere_mentions.extend(
            item for item in ["氩", "氮", "真空", "空气"] if item in user_message
        )

        concept_question = any(
            token in user_message for token in ["什么是", "解释", "原理", "概念"]
        )
        asks_retrieval = not concept_question and any(
            token in user_message
            for token in [
                "生长",
                "温度",
                "气氛",
                "助熔剂",
                "原料",
                "单晶",
                "方法",
                "对比",
                "推荐",
                "怎么做",
                "制备",
                "合成",
                "我要长",
                "我想长",
                "想长",
                "我要做",
                "我想做",
                "想做",
                "做单晶",
                "推测",
                "预测",
            ]
        )
        task_type = "retrieve" if asks_retrieval else "explain"
        missing_slots: list[str] = []
        if asks_retrieval and not formulas:
            missing_slots.append("target_material")

        return {
            "normalized_question": user_message,
            "task_type": task_type,
            "materials": formulas,
            "formulas": formulas,
            "growth_methods": methods,
            "temperature_mentions": temperature_mentions,
            "atmosphere_mentions": atmosphere_mentions,
            "precursor_mentions": [],
            "constraints": [],
            "missing_slots": missing_slots,
            "confidence": 0.75 if missing_slots else 0.9,
        }

    async def route(self, understanding: UserUnderstanding, force_retrieve: bool) -> RouteDecision:
        if force_retrieve:
            intent = "retrieve"
        elif understanding["missing_slots"]:
            intent = "clarify"
        elif understanding["task_type"] in {"retrieve", "compare", "recommend"}:
            intent = "retrieve"
        else:
            intent = "direct_answer"

        should_retrieve = intent == "retrieve"
        answer_mode = "evidence_grounded" if should_retrieve else "direct"
        if intent == "clarify":
            answer_mode = "ask_clarification"

        return {
            "intent": intent,
            "should_retrieve": should_retrieve,
            "reason": "Mock router selected retrieval based on material-growth keywords."
            if should_retrieve
            else "Mock router selected non-retrieval path.",
            "answer_mode": answer_mode,
            "required_slots": ["target_material"] if intent == "clarify" else [],
            "missing_slots": understanding["missing_slots"],
            "confidence": understanding["confidence"],
        }

    async def answer_direct(
        self,
        user_message: str,
        understanding: UserUnderstanding,
        messages: list[Message],
        long_memories: list[LongMemoryItem],
        conversation_summary: str | None = None,
        active_context: ActiveContext | None = None,
    ) -> str:
        return (
            "这是一个通用问题，当前不需要检索单晶生长数据条。"
            "如果你给出具体材料或化学式，我可以切换到基于数据条的检索增强回答。"
        )

    async def answer_with_evidence(
        self,
        understanding: UserUnderstanding,
        evidence_pack: EvidencePack,
        long_memories: list[LongMemoryItem],
    ) -> str:
        records = evidence_pack["records"]
        formulas = _unique(
            [record["material_formula"] or "" for record in records]
            or understanding["formulas"]
        )
        material = "、".join(formulas) if formulas else "目标材料"
        lines = [
            "## 真实记录综合结论",
            f"针对 **{material}**，当前知识库中有 **{len(records)} 条**可追溯的真实生长记录。",
            _method_summary(records),
            "\n### 记录支持的条件范围",
            *_temperature_summary(records),
        ]

        representatives = _representative_conditions(records)
        if representatives:
            lines.extend(["\n### 代表性已报道条件", *representatives])
            remaining = len(records) - len(representatives)
            if remaining > 0:
                lines.append(f"其余 {remaining} 条记录可在证据面板中查看。")

        if evidence_pack["conflicts"]:
            lines.extend(
                ["\n### 条件差异", *[f"- {item}" for item in evidence_pack["conflicts"]]]
            )

        boundaries = _evidence_boundaries(evidence_pack)
        lines.append("\n### 证据边界")
        if boundaries:
            lines.append("当前记录未能统一给出：" + "、".join(boundaries) + "。")
        else:
            lines.append("当前记录未显示统一缺失的结构化关键字段。")
        lines.append("以上内容是对真实记录的归纳，不表示其中任一条件已被证明为最优方案。")

        references = _unique([record["doi"] or record["record_id"] for record in records])
        if references:
            shown = references[:6]
            suffix = "；其余来源见证据面板。" if len(references) > len(shown) else "。"
            lines.append("\n### 证据来源\n" + "、".join(shown) + suffix)
        return "\n".join(lines)

    async def answer_with_limits(
        self,
        understanding: UserUnderstanding,
        evidence_pack: EvidencePack | None,
        reason: str,
    ) -> str:
        answer = f"当前证据不足，不能给出确定性的单晶生长方案。原因：{reason}"
        if evidence_pack and evidence_pack["records"]:
            answer += "\n已找到部分相关记录，但缺少回答该问题所需的关键字段。"
        answer += "\n请补充目标材料、可接受的生长方法、最高炉温或气氛限制后再检索。"
        return answer