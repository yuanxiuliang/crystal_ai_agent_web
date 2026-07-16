from __future__ import annotations

import asyncio
from pathlib import Path

from src.agent.graph import GrowthRAGGraph
from src.memory.store import MemoryLimits, MemoryStore
from src.retrieval.catalog_query import detect_aggregate_query, formula_elements
from src.retrieval.fact_catalog import FactCatalog
from src.retrieval.service import RetrievalService


FIXTURE_PATH = Path(__file__).resolve().parents[3] / "e2e" / "fixtures" / "growth_records.raw.jsonl"


class FailIfCalledRetrieval(RetrievalService):
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, plan):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise AssertionError("Aggregate queries must not call the material Milvus retrieval service.")


class RecordingPredictionService:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def predict(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        raise AssertionError("Aggregate real-record queries must never invoke prediction.")


class FailIfCalledLLM:
    async def analyze_and_route(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Structured aggregate queries must be recognized before the LLM.")


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


def _catalog(tmp_path) -> FactCatalog:
    catalog = FactCatalog(f"sqlite:///{tmp_path / 'catalog.sqlite3'}")
    assert catalog.sync_from_jsonl(FIXTURE_PATH).status == "rebuilt"
    assert catalog.sync_from_jsonl(FIXTURE_PATH).status == "ready"
    return catalog


def test_catalog_query_detection_is_structured_and_formula_safe() -> None:
    assert formula_elements("EuNi1.95As2") == ["Eu", "Ni", "As"]
    assert formula_elements("I₂") == ["I"]
    assert formula_elements("not-a-formula") == []

    eu_query = detect_aggregate_query("Eu基化合物一般采用哪些单晶生长方法？")
    assert eu_query == {
        "kind": "element_method_distribution",
        "label": "含 Eu 的目标化学式",
        "element": "Eu",
        "growth_method": None,
        "reactants": [],
    }
    assert detect_aggregate_query("碘传输剂使用哪些化合物的单晶生长呢？")["reactants"] == [
        {"name": "I2", "roles": ["additive", "raw_and_additive"]}
    ]
    raw_query = detect_aggregate_query("原料Ta和As可以生长哪些单晶？")
    assert [item["name"] for item in raw_query["reactants"]] == ["Ta", "As"]


def test_catalog_uses_exact_element_method_reactant_and_role_filters(tmp_path) -> None:
    catalog = _catalog(tmp_path)

    eu = catalog.aggregate(detect_aggregate_query("Eu基化合物一般采用哪些单晶生长方法？"))
    assert eu["total_records"] == 2
    assert eu["total_formulas"] == 2
    assert {(item["label"], item["record_count"]) for item in eu["groups"]} == {
        ("chemical vapor transport", 1),
        ("flux growth", 1),
    }

    flux = catalog.aggregate(detect_aggregate_query("Flux方法一般适用于哪些化合物？"))
    assert flux["total_records"] == 2
    assert {item["label"] for item in flux["groups"]} == {"EuCr2As2", "Mn3Ge"}

    iodine = catalog.aggregate(detect_aggregate_query("碘传输剂使用哪些化合物的单晶生长呢？"))
    assert iodine["total_records"] == 4
    assert {item["label"] for item in iodine["groups"]} == {"EuTe", "MnTe2", "TaAs", "ZnIn2S4"}
    assert all(record["growth_method"] == "chemical vapor transport" for record in iodine["representatives"])

    raw = catalog.aggregate(detect_aggregate_query("原料Ta和As可以生长哪些单晶？"))
    assert raw["total_records"] == 1
    assert raw["groups"][0]["label"] == "TaAs"


async def _run(graph: GrowthRAGGraph, message: str) -> tuple[dict, list[tuple[str, dict]]]:
    final: dict = {}
    events: list[tuple[str, dict]] = []
    async for event in graph.stream(
        {
            "user_id": "aggregate-alice",
            "session_id": "aggregate-session",
            "message": message,
            "options": {"top_k": 3, "stream_trace": True},
        }
    ):
        events.append((event.event, event.data))
        if event.event == "final":
            final = event.data
    return final, events


def test_graph_routes_aggregate_questions_to_real_record_catalog_without_prediction(tmp_path) -> None:
    retrieval = FailIfCalledRetrieval()
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=FailIfCalledLLM(),  # type: ignore[arg-type]
        retrieval=retrieval,
        prediction=prediction,  # type: ignore[arg-type]
        fact_catalog=_catalog(tmp_path),
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "Eu基化合物一般采用哪些单晶生长方法？"))

    assert retrieval.calls == 0
    assert prediction.calls == []
    assert final["evidence_kind"] == "literature_record"
    assert final["retrieval"]["mode"] == "aggregate_fact"
    assert final["aggregate"]["total_records"] == 2
    assert "真实记录统计" in final["answer"]
    assert "方法分布" in final["answer"]
    assert final["citations"]
    assert any(
        name == "node_started" and data["node"] == "plan_aggregate_retrieval"
        for name, data in events
    )
    assert any(
        name == "node_started" and data["node"] == "retrieve_aggregate_records"
        for name, data in events
    )
    assert not any(name == "prediction_started" for name, _ in events)


def test_empty_aggregate_query_never_invokes_prediction(tmp_path) -> None:
    catalog = _catalog(tmp_path)
    prediction = RecordingPredictionService()
    graph = GrowthRAGGraph(
        llm=FailIfCalledLLM(),  # type: ignore[arg-type]
        retrieval=FailIfCalledRetrieval(),
        prediction=prediction,  # type: ignore[arg-type]
        fact_catalog=catalog,
        memory_store=_store(tmp_path),
    )

    final, events = asyncio.run(_run(graph, "原料Nb和O可以生长哪些单晶？"))

    assert final["evidence_kind"] is None
    assert final["retrieval"]["mode"] == "aggregate_fact"
    assert final["retrieval"]["outcome"]["status"] == "empty"
    assert prediction.calls == []
    assert not any(name == "prediction_eligible" for name, _ in events)
