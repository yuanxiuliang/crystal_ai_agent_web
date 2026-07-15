from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from src.agent.graph import GrowthRAGGraph
from src.config import settings
from src.llm.mock_client import MockLLMClient
from src.memory.store import MemoryLimits, MemoryStore
from src.prediction.contracts import PredictionModelInfo, PredictionResult, PredictionRoute
from src.prediction.service import PredictionService
from src.retrieval.service import RetrievalService


class StaticRetrieval(RetrievalService):
    def __init__(self, records: list[dict] | None = None, error: Exception | None = None) -> None:
        self.records = records or []
        self.error = error
        self.calls = 0

    async def retrieve(self, plan):
        self.calls += 1
        if self.error:
            raise self.error
        return self.records


class RecordingPredictionService:
    def __init__(self) -> None:
        self.calls = []

    async def predict(self, request):
        self.calls.append(request)
        return PredictionResult(
            prediction_run_id="prediction-fallback-test",
            source=request.source,
            formula=request.formula,
            formula_std=request.formula,
            formula_tokens=["Mn", "3", "Ga", "N"],
            target_elements=["Mn", "Ga", "N"],
            unknown_formula_tokens=[],
            routes=[
                PredictionRoute(
                    rank=1,
                    relative_rank_weight=1.0,
                    method="Flux",
                    raw_reactants=[
                        {"name": "Mn", "type": "raw", "r": None, "elements": ["Mn"]},
                        {"name": "Ga", "type": "raw", "r": None, "elements": ["Ga"]},
                    ],
                    additives=[{"name": "N2", "type": "adtv", "r": None, "elements": ["N"]}],
                    growth={
                        "T_s": {"token": "TEMP_BIN_80", "range_c": [750, 760]},
                        "T_e": {"token": "TEMP_BIN_55", "range_c": [500, 510]},
                        "dur": {"token": "DUR_BIN_17", "range_h": [85, 90]},
                    },
                    element_coverage_ok=True,
                )
            ],
            model=PredictionModelInfo(
                model_id="growth-route-transformer",
                model_version="v2.0.0",
                artifact_digest="a" * 64,
                supported_methods=["Flux", "CVT"],
                parameter_count=6_614_099,
            ),
            warnings=["candidate only"],
            runtime_ms=1,
        )


class MalformedTargetLLM(MockLLMClient):
    async def analyze_and_route(
        self,
        user_message,
        messages,
        long_memories,
        force_retrieve,
        conversation_summary=None,
        active_context=None,
    ):
        if user_message == "我想长M n":
            return (
                {
                    "normalized_question": user_message,
                    "task_type": "retrieve",
                    "materials": ["M n"],
                    "formulas": ["Mn"],
                    "growth_methods": [],
                    "temperature_mentions": [],
                    "atmosphere_mentions": [],
                    "precursor_mentions": [],
                    "constraints": [],
                    "missing_slots": [],
                    "confidence": 0.8,
                },
                {
                    "intent": "retrieve",
                    "should_retrieve": True,
                    "reason": "test malformed material route",
                    "answer_mode": "evidence_grounded",
                    "required_slots": ["target_material"],
                    "missing_slots": [],
                    "confidence": 0.8,
                },
            )
        return await super().analyze_and_route(
            user_message,
            messages,
            long_memories,
            force_retrieve,
            conversation_summary,
            active_context,
        )


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(
        f"sqlite:///{tmp_path / 'memory.sqlite3'}",
        MemoryLimits(
            short_max_messages=4,
            summary_max_chars=220,
            message_max_chars=400,
            active_context_max_items=4,
            long_max_items_per_user=4,
            long_prompt_max_items=4,
            long_prompt_max_chars=600,
            session_ttl_days=30,
            event_ttl_days=90,
            cleanup_interval_seconds=3600,
            postgres_connect_timeout_seconds=3,
        ),
    )


async def _run(
    graph: GrowthRAGGraph,
    message: str,
    *,
    options: dict | None = None,
) -> tuple[dict, list[tuple[str, dict]]]:
    final: dict = {}
    events: list[tuple[str, dict]] = []
    async for event in graph.stream(
        {
            "user_id": "fallback-alice",
            "session_id": "fallback-session",
            "message": message,
            "options": {"top_k": 3, "stream_trace": True, **(options or {})},
        }
    ):
        events.append((event.event, event.data))
        if event.event == "final":
            final = event.data
    return final, events


def _sufficient_record() -> dict:
    return {
        "record_id": "record-mn3gan-1",
        "score": 0.9,
        "dense_score": 0.9,
        "sparse_score": 0.9,
        "material_formula": "Mn3GaN",
        "material_name": None,
        "growth_method": "flux growth",
        "temperature_program": "750 C to 500 C",
        "atmosphere": "argon",
        "precursors": ["Mn", "Ga", "N2"],
        "doi": "10.1000/example",
        "source_text": "Mn3GaN was grown by flux growth from 750 C to 500 C under argon.",
        "source_file": "test",
        "matched_fields": ["material_formula", "temperature_program"],
    }


def test_sufficient_retrieval_never_invokes_prediction(tmp_path) -> None:
    retrieval = StaticRetrieval([_sufficient_record()])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "请为 Mn3GaN 推荐可尝试的单晶生长方案。"))

    assert retrieval.calls == 1
    assert prediction.calls == []
    assert final["evidence_kind"] == "literature_record"
    assert final["citations"]
    assert final["evidence_records"] == [
        {
            "record_id": "record-mn3gan-1",
            "score": 0.9,
            "title": None,
            "material_formula": "Mn3GaN",
            "growth_method": "flux growth",
            "temperature_program": "750 C to 500 C",
            "atmosphere": "argon",
            "precursors": ["Mn", "Ga", "N2"],
            "key_facts": [
                "growth method: flux growth",
                "temperature program: 750 C to 500 C",
                "atmosphere: argon",
                "precursors: Mn, Ga, N2",
            ],
            "source_text": "Mn3GaN was grown by flux growth from 750 C to 500 C under argon.",
            "doi": "10.1000/example",
        }
    ]
    assert final["prediction"] is None
    assert "真实记录综合结论" in final["answer"]
    assert "记录支持的条件范围" in final["answer"]
    assert "证据边界" in final["answer"]
    assert any(
        name == "retrieval_outcome" and data["status"] == "sufficient" for name, data in events
    )
    assert not any(name == "prediction_started" for name, _ in events)


def test_empty_route_request_uses_prediction_fallback_only(tmp_path) -> None:
    retrieval = StaticRetrieval([])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "请为 Mn3GaN 推荐可尝试的单晶生长方案。"))

    assert len(prediction.calls) == 1
    assert prediction.calls[0].source == "retrieval_fallback"
    assert final["evidence_kind"] == "model_prediction"
    assert final["citations"] == []
    assert final["prediction"]["source"] == "retrieval_fallback"
    assert "不是文献事实" in final["answer"]
    assert any(name == "prediction_result" for name, _ in events)
    assert any(name == "prediction_warning" for name, _ in events)


def test_empty_how_to_make_request_uses_prediction_fallback(tmp_path) -> None:
    retrieval = StaticRetrieval([])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "Mn3GaN怎么做？"))

    assert len(prediction.calls) == 1
    assert prediction.calls[0].formula == "Mn3GaN"
    assert final["evidence_kind"] == "model_prediction"
    assert final["citations"] == []
    assert "可尝试" in final["answer"]
    assert any(name == "prediction_eligible" and data["eligible"] is True for name, data in events)


def test_empty_i_want_to_grow_request_uses_prediction_fallback(tmp_path) -> None:
    retrieval = StaticRetrieval([])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "我要长Mn3GaN单晶"))

    assert len(prediction.calls) == 1
    assert prediction.calls[0].formula == "Mn3GaN"
    assert final["evidence_kind"] == "model_prediction"
    assert any(name == "prediction_started" for name, _ in events)


def test_make_formula_request_retrieves_before_prediction_fallback(tmp_path) -> None:
    retrieval = StaticRetrieval([])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "我要做Mn3ZnN"))

    assert retrieval.calls == 1
    assert final["route"]["intent"] == "retrieve"
    assert final["retrieval"]["filters"]["material_formula"] == "Mn3ZnN"
    assert prediction.calls[0].formula == "Mn3ZnN"
    assert final["evidence_kind"] == "model_prediction"
    assert any(
        name == "node_started" and data["node"] == "plan_retrieval" for name, data in events
    )
    assert any(
        name == "node_started" and data["node"] == "retrieve_records" for name, data in events
    )


def test_follow_up_speculation_uses_active_formula_after_retrieval(tmp_path) -> None:
    retrieval = StaticRetrieval([])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    first, _ = asyncio.run(_run(graph, "Mn3GaN 的 CVT 温度有哪些文献记录？"))
    snapshot = graph.memory_store.load_session("fallback-alice", "fallback-session")
    second, events = asyncio.run(_run(graph, "那你推测一个"))

    assert first["prediction"] is None
    assert snapshot is not None
    assert snapshot.active_context["active_formulas"] == ["Mn3GaN"]
    assert len(prediction.calls) == 1
    assert second["evidence_kind"] == "model_prediction"
    assert any(name == "prediction_eligible" and data["eligible"] is True for name, data in events)


def test_unparseable_growth_target_does_not_use_related_material_records(tmp_path) -> None:
    retrieval = StaticRetrieval([_sufficient_record()])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, _ = asyncio.run(_run(graph, "你好，我要长MN"))

    assert prediction.calls == []
    assert final["evidence_kind"] is None
    assert final["route"]["intent"] == "clarify"
    assert final["retrieval"] is None
    assert final["citations"] == []


def test_malformed_new_target_never_inherits_previous_active_formula(tmp_path) -> None:
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MalformedTargetLLM(),
        retrieval=StaticRetrieval([]),
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    first, _ = asyncio.run(_run(graph, "我要长Mn3GaN单晶"))
    second, _ = asyncio.run(_run(graph, "我想长M n"))
    snapshot = graph.memory_store.load_session("fallback-alice", "fallback-session")

    assert first["evidence_kind"] == "model_prediction"
    assert len(prediction.calls) == 1
    assert second["route"]["intent"] == "clarify"
    assert second["prediction"] is None
    assert second["retrieval"] is None
    assert snapshot is not None
    assert snapshot.active_context["active_formulas"] == []


def test_factual_literature_question_does_not_fallback_to_prediction(tmp_path) -> None:
    retrieval = StaticRetrieval([])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "Mn3GaN 的 CVT 温度有哪些文献记录？"))

    assert prediction.calls == []
    assert final["evidence_kind"] is None
    assert final["prediction"] is None
    assert any(name == "prediction_eligible" and data["eligible"] is False for name, data in events)


def test_evidence_only_mode_never_falls_back_to_prediction(tmp_path) -> None:
    retrieval = StaticRetrieval([])
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, _ = asyncio.run(
        _run(
            graph,
            "请为 Mn3GaN 推荐可尝试的单晶生长方案。",
            options={"force_retrieve": True, "evidence_only": True},
        )
    )

    assert prediction.calls == []
    assert final["evidence_kind"] is None


def test_retrieval_outage_never_invokes_prediction(tmp_path) -> None:
    retrieval = StaticRetrieval(error=RuntimeError("milvus offline"))
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "请为 Mn3GaN 推荐可尝试的单晶生长方案。"))

    assert prediction.calls == []
    assert final["retrieval"]["outcome"]["status"] == "unavailable"
    assert any(
        name == "retrieval_outcome" and data["status"] == "unavailable" for name, data in events
    )


@pytest.mark.integration
def test_real_prediction_service_runs_only_after_empty_retrieval(tmp_path) -> None:
    config = replace(
        settings,
        prediction_database_url=f"sqlite:///{tmp_path / 'prediction.sqlite3'}",
        prediction_return_sequences=3,
    )
    prediction = PredictionService(config)
    graph = GrowthRAGGraph(
        llm=MockLLMClient(),
        retrieval=StaticRetrieval([]),
        prediction=prediction,
        memory_store=_store(tmp_path),
    )

    final, _ = asyncio.run(_run(graph, "请为 Mn3GaN 推荐可尝试的单晶生长方案。"))

    assert final["evidence_kind"] == "model_prediction"
    assert final["prediction"]["source"] == "retrieval_fallback"
    assert final["citations"] == []