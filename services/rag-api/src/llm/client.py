from __future__ import annotations

from typing import Protocol

from ..agent.state import (
    ActiveContext,
    EvidencePack,
    LongMemoryItem,
    Message,
    RouteDecision,
    UserUnderstanding,
)


class LLMClient(Protocol):
    async def analyze_and_route(
        self,
        user_message: str,
        messages: list[Message],
        long_memories: list[LongMemoryItem],
        force_retrieve: bool,
        conversation_summary: str | None = None,
        active_context: ActiveContext | None = None,
    ) -> tuple[UserUnderstanding, RouteDecision]:
        ...

    async def understand(
        self,
        user_message: str,
        messages: list[Message],
        long_memories: list[LongMemoryItem],
        conversation_summary: str | None = None,
        active_context: ActiveContext | None = None,
    ) -> UserUnderstanding:
        ...

    async def route(self, understanding: UserUnderstanding, force_retrieve: bool) -> RouteDecision:
        ...

    async def answer_direct(
        self,
        user_message: str,
        understanding: UserUnderstanding,
        messages: list[Message],
        long_memories: list[LongMemoryItem],
        conversation_summary: str | None = None,
        active_context: ActiveContext | None = None,
    ) -> str:
        ...

    async def answer_with_evidence(
        self,
        understanding: UserUnderstanding,
        evidence_pack: EvidencePack,
        long_memories: list[LongMemoryItem],
    ) -> str:
        ...

    async def answer_with_limits(
        self,
        understanding: UserUnderstanding,
        evidence_pack: EvidencePack | None,
        reason: str,
    ) -> str:
        ...
