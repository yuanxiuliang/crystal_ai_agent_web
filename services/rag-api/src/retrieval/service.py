from __future__ import annotations

from typing import Protocol

from ..agent.state import RetrievalPlan, RetrievedRecord


class RetrievalService(Protocol):
    async def retrieve(self, plan: RetrievalPlan) -> list[RetrievedRecord]:
        ...

