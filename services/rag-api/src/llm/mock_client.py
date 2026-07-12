from __future__ import annotations

import re

from ..agent.state import ActiveContext, EvidencePack, LongMemoryItem, Message, RouteDecision, UserUnderstanding


FORMULA_RE = re.compile(r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)+\b")


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

        concept_question = any(token in user_message for token in ["什么是", "解释", "原理", "概念"])
        asks_retrieval = not concept_question and any(
            token in user_message
            for token in ["生长", "温度", "气氛", "助熔剂", "原料", "单晶", "方法", "对比", "推荐"]
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
        lines = ["根据当前检索到的单晶生长数据条，可以得到以下结论："]
        for idx, record in enumerate(evidence_pack["records"], start=1):
            facts = "；".join(record["key_facts"]) if record["key_facts"] else "记录缺少结构化关键字段"
            lines.append(f"{idx}. {record['record_id']}：{facts}")
        if evidence_pack["conflicts"]:
            lines.append("需要注意，检索结果中存在条件差异：" + "；".join(evidence_pack["conflicts"]))
        lines.append("以上结论只基于当前数据条；未在证据中出现的实验参数不作确定性补充。")
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
