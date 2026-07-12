from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from src.agent.graph import GrowthRAGGraph
from src.config import Settings
from src.llm.mock_client import MockLLMClient
from src.memory.checkpointer import CheckpointerRuntime
from src.memory.store import MemoryLimits, MemoryStore
from src.retrieval.mock_retrieval import MockRetrievalService


POSTGRES_URL = os.getenv("RAG_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="RAG_TEST_POSTGRES_URL is not configured")


def _limits(*, checkpoint_thread_rollover_turns: int = 100) -> MemoryLimits:
    return MemoryLimits(
        short_max_messages=4,
        summary_max_chars=220,
        message_max_chars=400,
        active_context_max_items=4,
        long_max_items_per_user=10,
        long_prompt_max_items=4,
        long_prompt_max_chars=600,
        session_ttl_days=30,
        event_ttl_days=90,
        cleanup_interval_seconds=3600,
        postgres_connect_timeout_seconds=3,
        checkpoint_thread_rollover_turns=checkpoint_thread_rollover_turns,
    )


async def _run_turn(graph: GrowthRAGGraph, user_id: str, session_id: str, message: str) -> dict:
    final: dict = {}
    async for event in graph.stream(
        {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "options": {"top_k": 1, "stream_trace": False},
        }
    ):
        if event.event == "final":
            final = event.data
    return final


class MemoryAwareMockLLM(MockLLMClient):
    async def answer_direct(self, user_message, understanding, messages, long_memories, *args):
        if not long_memories:
            return "长期记忆：无"
        return "长期记忆：" + " | ".join(item["content"] for item in long_memories)


def test_postgres_long_memory_and_checkpointer() -> None:
    assert POSTGRES_URL
    user_id = f"pg-user-{uuid4().hex}"
    session_id = f"pg-session-{uuid4().hex}"
    store = MemoryStore(POSTGRES_URL, _limits())

    created = store.upsert_memory(
        user_id=user_id,
        memory_type="constraint",
        memory_key="furnace.max_temperature_c",
        content="用户明确实验约束：最高炉温为 900 C。",
        source="explicit_user_request",
        confidence=1.0,
        subject="furnace",
        predicate="max_temperature_c",
        value_json={"value": 900, "unit": "C"},
    )
    assert created.written is True
    assert created.memory_id

    stored = store.store_memory_embedding(created.memory_id, [1.0] + [0.0] * 383)
    assert stored.written is True
    matched = store.load_long_memories(
        user_id=user_id,
        query="最高炉温",
        query_embedding=[1.0] + [0.0] * 383,
    )
    assert len(matched) == 1
    assert matched[0]["type"] == "constraint"

    config = Settings(memory_database_url=POSTGRES_URL, memory_checkpoint_backend="postgres")
    runtime = CheckpointerRuntime(config)

    async def scenario() -> None:
        first_graph = GrowthRAGGraph(
            llm=MockLLMClient(),
            retrieval=MockRetrievalService(),
            memory_store=store,
            checkpointer_runtime=runtime,
        )
        first = await _run_turn(first_graph, user_id, session_id, "请记住，我关注 ZnIn2S4。")
        assert first["memory"]["short_term_updated"] is True
        assert first["memory"]["long_term_written"] is True

        second_graph = GrowthRAGGraph(
            llm=MockLLMClient(),
            retrieval=MockRetrievalService(),
            memory_store=store,
            checkpointer_runtime=runtime,
        )
        second = await _run_turn(second_graph, user_id, session_id, "请解释这个概念。")
        assert second["memory"]["short_term_updated"] is True

        checkpointer = await runtime.get()
        checkpoint_tuple = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": GrowthRAGGraph._thread_id({"user_id": user_id, "session_id": session_id})}}
        )
        assert checkpoint_tuple is not None
        channel_values = checkpoint_tuple.checkpoint["channel_values"]
        assert len(channel_values["messages"]) == 4
        assert channel_values["retrieved_records"] == []
        assert channel_values["evidence_pack"] is None
        await runtime.close()

    asyncio.run(scenario())


def test_postgres_checkpointer_rollover_preserves_bounded_short_memory() -> None:
    assert POSTGRES_URL
    user_id = f"pg-rollover-user-{uuid4().hex}"
    session_id = f"pg-rollover-session-{uuid4().hex}"
    store = MemoryStore(POSTGRES_URL, _limits(checkpoint_thread_rollover_turns=2))
    config = Settings(memory_database_url=POSTGRES_URL, memory_checkpoint_backend="postgres")
    runtime = CheckpointerRuntime(config)

    async def scenario() -> None:
        graph = GrowthRAGGraph(
            llm=MockLLMClient(),
            retrieval=MockRetrievalService(),
            memory_store=store,
            checkpointer_runtime=runtime,
        )
        await _run_turn(graph, user_id, session_id, "第一轮，请解释一个通用概念。")
        await _run_turn(graph, user_id, session_id, "第二轮，请解释另一个通用概念。")

        before = store.get_or_create_checkpoint_session(
            user_id=user_id,
            session_id=session_id,
            initial_thread_id=GrowthRAGGraph._thread_id(
                {"user_id": user_id, "session_id": session_id}
            ),
        )
        assert before.turn_count == 2

        await _run_turn(graph, user_id, session_id, "第三轮，请解释第三个通用概念。")

        after = store.get_or_create_checkpoint_session(
            user_id=user_id,
            session_id=session_id,
            initial_thread_id=GrowthRAGGraph._thread_id(
                {"user_id": user_id, "session_id": session_id}
            ),
        )
        assert after.graph_thread_id != before.graph_thread_id
        assert after.turn_count == 1

        checkpointer = await runtime.get()
        old_checkpoint = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": before.graph_thread_id}}
        )
        assert old_checkpoint is None
        current_checkpoint = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": after.graph_thread_id}}
        )
        assert current_checkpoint is not None
        values = current_checkpoint.checkpoint["channel_values"]
        assert len(values["messages"]) == 4
        assert values["conversation_summary"]
        assert values["retrieved_records"] == []
        await runtime.close()

    asyncio.run(scenario())


def test_postgres_long_memory_is_visible_only_to_its_user() -> None:
    assert POSTGRES_URL
    alice_id = f"pg-alice-{uuid4().hex}"
    bob_id = f"pg-bob-{uuid4().hex}"
    shared_session_id = f"shared-session-{uuid4().hex}"
    store = MemoryStore(POSTGRES_URL, _limits())
    config = Settings(memory_database_url=POSTGRES_URL, memory_checkpoint_backend="postgres")
    runtime = CheckpointerRuntime(config)

    async def scenario() -> None:
        graph = GrowthRAGGraph(
            llm=MemoryAwareMockLLM(),
            retrieval=MockRetrievalService(),
            memory_store=store,
            checkpointer_runtime=runtime,
        )

        remembered = await _run_turn(graph, alice_id, shared_session_id, "请记住，我关注 ZnIn2S4。")
        assert remembered["memory"]["long_term_written"] is True

        alice_follow_up = await _run_turn(graph, alice_id, shared_session_id, "我当前关注什么材料？")
        assert "ZnIn2S4" in alice_follow_up["answer"]

        bob_follow_up = await _run_turn(graph, bob_id, shared_session_id, "我当前关注什么材料？")
        assert bob_follow_up["answer"] == "长期记忆：无"

        alice_memories = store.load_long_memories(user_id=alice_id, query="关注材料")
        bob_memories = store.load_long_memories(user_id=bob_id, query="关注材料")
        assert len(alice_memories) == 1
        assert "ZnIn2S4" in alice_memories[0]["content"]
        assert bob_memories == []

        checkpointer = await runtime.get()
        alice_thread = GrowthRAGGraph._thread_id(
            {"user_id": alice_id, "session_id": shared_session_id}
        )
        bob_thread = GrowthRAGGraph._thread_id(
            {"user_id": bob_id, "session_id": shared_session_id}
        )
        assert alice_thread != bob_thread
        assert await checkpointer.aget_tuple({"configurable": {"thread_id": alice_thread}})
        assert await checkpointer.aget_tuple({"configurable": {"thread_id": bob_thread}})
        await runtime.close()

    asyncio.run(scenario())
