from __future__ import annotations

import asyncio

from src.agent.graph import GrowthRAGGraph
from src.llm.mock_client import MockLLMClient
from src.memory.store import MemoryLimits, MemoryStore
from src.memory.worker import MemoryWorker
from src.retrieval.mock_retrieval import MockRetrievalService


def _store(tmp_path) -> MemoryStore:
    limits = MemoryLimits(
        short_max_messages=4,
        summary_max_chars=220,
        message_max_chars=400,
        active_context_max_items=4,
        long_max_items_per_user=2,
        long_prompt_max_items=2,
        long_prompt_max_chars=400,
        session_ttl_days=30,
        event_ttl_days=90,
        cleanup_interval_seconds=3600,
        postgres_connect_timeout_seconds=3,
    )
    return MemoryStore(f"sqlite:///{tmp_path / 'memory.sqlite3'}", limits)


async def _run_turn(graph: GrowthRAGGraph, *, user_id: str, session_id: str, message: str) -> dict:
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


def test_memory_is_persistent_and_bounded(tmp_path) -> None:
    store = _store(tmp_path)
    reloaded_store = MemoryStore(store.database_url, store.limits)
    graph = GrowthRAGGraph(
        llm=MockLLMClient(), retrieval=MockRetrievalService(), memory_store=reloaded_store
    )

    first = asyncio.run(
        _run_turn(
            graph,
            user_id="researcher-a",
            session_id="session-a",
            message="请记住，我关注 ZnIn2S4。",
        )
    )
    assert first["memory"]["short_term_updated"] is True
    assert first["memory"]["long_term_written"] is True

    graph = GrowthRAGGraph(
        llm=MockLLMClient(), retrieval=MockRetrievalService(), memory_store=store
    )
    for index in range(5):
        final = asyncio.run(
            _run_turn(
                graph,
                user_id="researcher-a",
                session_id="session-a",
                message=f"请解释第 {index} 个通用概念。",
            )
        )
        assert final["memory"]["short_term_updated"] is True

    snapshot = store.load_session("researcher-a", "session-a")
    assert snapshot is not None
    assert len(snapshot.messages) == 4
    assert len(snapshot.conversation_summary or "") <= 220

    memories = store.load_long_memories(user_id="researcher-a", query="ZnIn2S4")
    assert len(memories) == 1
    assert memories[0]["type"] == "research_profile"
    assert "ZnIn2S4" in memories[0]["content"]


def test_short_memory_material_history_survives_message_compaction(tmp_path) -> None:
    store = _store(tmp_path)
    graph = GrowthRAGGraph(
        llm=MockLLMClient(), retrieval=MockRetrievalService(), memory_store=store
    )
    user_id = "researcher-a"
    session_id = "material-history"
    formulas = ["Mn3GaN", "EuCr2As2", "Mn3ZnN"]

    for formula in formulas:
        asyncio.run(
            _run_turn(
                graph,
                user_id=user_id,
                session_id=session_id,
                message=f"我要长{formula}单晶",
            )
        )

    recalled = asyncio.run(
        _run_turn(
            graph,
            user_id=user_id,
            session_id=session_id,
            message="我问过哪些单晶样品？",
        )
    )
    follow_up = asyncio.run(
        _run_turn(
            graph,
            user_id=user_id,
            session_id=session_id,
            message="没有其他的吗？",
        )
    )
    snapshot = store.load_session(user_id, session_id)

    assert snapshot is not None
    assert [item["formula"] for item in snapshot.short_memory["material_history"]] == formulas
    assert all(formula in recalled["answer"] for formula in formulas)
    assert all(formula in follow_up["answer"] for formula in formulas)


def test_reset_short_term_session_discards_snapshot_and_replaces_checkpoint_mapping(tmp_path) -> None:
    store = _store(tmp_path)
    user_id = "researcher-a"
    session_id = "edited-session"
    old_thread_id = "growth-rag-before-edit"

    store.save_session(
        user_id=user_id,
        session_id=session_id,
        messages=[{"role": "user", "content": "TaAs怎么做？"}],
        conversation_summary="用户询问 TaAs。",
        active_context={"formula": "TaAs"},
        short_memory={"material_history": [{"formula": "TaAs"}]},
    )
    checkpoint = store.get_or_create_checkpoint_session(
        user_id=user_id,
        session_id=session_id,
        initial_thread_id=old_thread_id,
    )
    assert checkpoint.graph_thread_id == old_thread_id
    assert store.complete_checkpoint_turn(
        user_id=user_id,
        session_id=session_id,
        graph_thread_id=old_thread_id,
    )

    assert store.reset_short_term_session(
        user_id=user_id,
        session_id=session_id,
        replacement_thread_id="growth-rag-after-edit",
    ) == old_thread_id
    assert store.load_session(user_id, session_id) is None

    replacement = store.get_or_create_checkpoint_session(
        user_id=user_id,
        session_id=session_id,
        initial_thread_id="ignored-after-reset",
    )
    assert replacement.graph_thread_id == "growth-rag-after-edit"
    assert replacement.turn_count == 0


def test_long_memory_upserts_and_enforces_quota(tmp_path) -> None:
    store = _store(tmp_path)
    created = store.upsert_memory(
        user_id="researcher-a",
        memory_type="constraint",
        memory_key="furnace.max_temperature_c",
        content="最高炉温 900 C",
        source="explicit_user_request",
        confidence=0.98,
    )
    updated = store.upsert_memory(
        user_id="researcher-a",
        memory_type="constraint",
        memory_key="furnace.max_temperature_c",
        content="最高炉温 1100 C",
        source="explicit_user_request",
        confidence=0.98,
    )
    second = store.upsert_memory(
        user_id="researcher-a",
        memory_type="preference",
        memory_key="answer.language",
        content="优先中文回答",
        source="explicit_user_request",
        confidence=0.95,
    )
    rejected = store.upsert_memory(
        user_id="researcher-a",
        memory_type="confirmed_fact",
        memory_key="extra.fact",
        content="超出配额的事实",
        source="explicit_user_request",
        confidence=0.95,
    )

    assert created.reason == "created"
    assert updated.reason == "updated"
    assert second.reason == "created"
    assert rejected.written is False
    assert rejected.reason == "active_memory_quota_reached"

    memories = store.load_long_memories(user_id="researcher-a", query="炉温")
    assert len(memories) == 2
    assert any("1100 C" in item["content"] for item in memories)

    jobs = store.claim_memory_jobs()
    assert jobs
    embedding = [1.0] + [0.0] * 383
    embedded = store.store_memory_embedding(jobs[0]["memory_id"], embedding)
    assert embedded.written is True
    store.complete_memory_job(jobs[0]["id"])


def test_memory_worker_embeds_confirmed_long_memory(tmp_path) -> None:
    class FakeEmbeddingClient:
        def embed_query(self, text: str) -> list[float]:
            assert "中文" in text
            return [1.0] + [0.0] * 383

    store = _store(tmp_path)
    result = store.upsert_memory(
        user_id="researcher-a",
        memory_type="preference",
        memory_key="answer.language",
        content="用户偏好中文回答。",
        source="explicit_user_request",
        confidence=1.0,
    )
    assert result.written is True
    worker = MemoryWorker(store=store, embedding_client=FakeEmbeddingClient(), poll_seconds=1)
    assert asyncio.run(worker.run_once()) == 1
    memories = store.load_long_memories(
        user_id="researcher-a",
        query="输出语言",
        query_embedding=[1.0] + [0.0] * 383,
    )
    assert memories[0]["type"] == "preference"
