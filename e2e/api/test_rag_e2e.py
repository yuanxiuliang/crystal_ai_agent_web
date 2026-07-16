from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from uuid import uuid4

import httpx

from src.config import settings
from src.memory.store import MemoryLimits, MemoryStore


API_BASE_URL = os.getenv("RAG_E2E_API_BASE_URL", "http://127.0.0.1:8003")
PASSWORD = "e2e-password-12345"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def _email(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}@e2e.invalid"


def _login(client: httpx.Client, email: str, password: str = PASSWORD) -> dict[str, Any]:
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    response.raise_for_status()
    return response.json()


def _new_session(client: httpx.Client) -> str:
    response = client.post("/api/rag/sessions")
    response.raise_for_status()
    return str(response.json()["id"])


def _stream_chat(
    client: httpx.Client,
    session_id: str,
    message: str,
    *,
    replace_message_id: str | None = None,
) -> list[dict[str, Any]]:
    payload = {
        "session_id": session_id,
        "message": message,
        "options": {
            "force_retrieve": False,
            "top_k": 3,
            "retrieval_mode": "hybrid",
            "stream_trace": False,
        },
    }
    if replace_message_id:
        payload["replace_message_id"] = replace_message_id
    events: list[dict[str, Any]] = []
    with client.stream("POST", "/api/rag/chat/stream", json=payload) as response:
        response.raise_for_status()
        event_name: str | None = None
        data: dict[str, Any] | None = None
        for line in response.iter_lines():
            if not line:
                if event_name and data is not None:
                    events.append({"event": event_name, "data": data})
                event_name = None
                data = None
                continue
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
    assert events, "The chat endpoint returned no SSE events."
    assert events[-1]["event"] == "run_finished", events
    return events


def _event_data(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event["data"] for event in events if event["event"] == name]


def _final(events: list[dict[str, Any]]) -> dict[str, Any]:
    values = _event_data(events, "final")
    assert len(values) == 1, events
    return values[0]


def _node_names(events: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("node")) for item in _event_data(events, "node_started")]


def _assert_no_untrusted_dois(answer: str, allowed: set[str]) -> None:
    found = {value.rstrip(".,;:)") for value in DOI_PATTERN.findall(answer)}
    assert found <= allowed, f"LLM returned DOI values outside evidence: {sorted(found - allowed)}"


def _memory_store() -> MemoryStore:
    return MemoryStore(settings.memory_database_url, MemoryLimits.from_settings(settings))


def test_real_llm_rag_contracts() -> None:
    deadline = time.monotonic() + 15
    while True:
        try:
            health = httpx.get(f"{API_BASE_URL}/api/rag/health", timeout=5).json()
            if health.get("status") == "ok":
                break
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline, "RAG API did not become healthy."
        time.sleep(0.5)

    assert health["memory_database"] == "postgres"
    assert health["short_term_backend"] == "postgres-checkpointer"
    assert health["prediction"] == "enabled"
    assert health["llm_backend"] == "openai-compatible"

    with httpx.Client(base_url=API_BASE_URL, timeout=120) as alice, httpx.Client(
        base_url=API_BASE_URL, timeout=120
    ) as bob:
        alice_email = _email("alice")
        first_login = _login(alice, alice_email)
        assert first_login["created"] is True
        alice_id = str(first_login["user"]["id"])

        alice.post("/api/auth/logout").raise_for_status()
        rejected_login = alice.post(
            "/api/auth/login", json={"email": alice_email, "password": "wrong-password-123"}
        )
        assert rejected_login.status_code == 401
        second_login = _login(alice, alice_email)
        assert second_login["created"] is False

        taas_session = _new_session(alice)
        literature_events = _stream_chat(alice, taas_session, "TaAs 单晶怎么做？")
        literature_final = _final(literature_events)
        literature_nodes = _node_names(literature_events)
        literature_outcome = _event_data(literature_events, "retrieval_outcome")[-1]

        assert literature_final["evidence_kind"] == "literature_record"
        assert literature_final["prediction"] is None
        assert literature_outcome["status"] == "sufficient"
        assert "run_prediction" not in literature_nodes
        assert "answer_with_evidence" in literature_nodes
        assert literature_final["citations"]
        assert {item["doi"] for item in literature_final["citations"]} == {"10.5555/e2e.taas.001"}
        assert len(literature_final["evidence_records"]) == 1
        literature_record = literature_final["evidence_records"][0]
        assert literature_record["record_id"] == "e2e::taas::001"
        assert literature_record["material_formula"] == "TaAs"
        assert literature_record["growth_method"] == "chemical vapor transport"
        assert literature_record["temperature_program"] == (
            "source_temperature_c=1050; crystal_temperature_c=950; duration_hours=336"
        )
        assert literature_record["doi"] == "10.5555/e2e.taas.001"
        assert "TaAs" in literature_final["answer"]
        assert any(
            token in literature_final["answer"]
            for token in ("1050", "950", "336", "化学气相", "chemical vapor transport")
        )
        _assert_no_untrusted_dois(literature_final["answer"], {"10.5555/e2e.taas.001"})

        aggregate_session = _new_session(alice)
        aggregate_events = _stream_chat(
            alice, aggregate_session, "Eu基化合物一般采用哪些单晶生长方法？"
        )
        aggregate_final = _final(aggregate_events)
        aggregate_nodes = _node_names(aggregate_events)
        aggregate_outcome = _event_data(aggregate_events, "retrieval_outcome")[-1]
        assert aggregate_final["evidence_kind"] == "literature_record"
        assert aggregate_final["prediction"] is None
        assert aggregate_final["retrieval"]["mode"] == "aggregate_fact"
        assert aggregate_outcome["status"] == "sufficient"
        assert "plan_aggregate_retrieval" in aggregate_nodes
        assert "retrieve_aggregate_records" in aggregate_nodes
        assert "run_prediction" not in aggregate_nodes
        assert aggregate_final["aggregate"]["total_records"] == 2
        assert aggregate_final["aggregate"]["total_formulas"] == 2
        assert "真实记录统计" in aggregate_final["answer"]
        assert "方法分布" in aggregate_final["answer"]
        assert {item["doi"] for item in aggregate_final["citations"]} == {
            "10.5555/e2e.eucr2as2.001",
            "10.5555/e2e.eute.001",
        }

        follow_up_events = _stream_chat(alice, taas_session, "它的生长温度是多少？")
        follow_up_final = _final(follow_up_events)
        follow_up_outcome = _event_data(follow_up_events, "retrieval_outcome")[-1]
        assert follow_up_final["evidence_kind"] == "literature_record"
        assert follow_up_outcome["status"] == "sufficient"
        assert "TaAs" in follow_up_final["answer"]
        assert any(token in follow_up_final["answer"] for token in ("1050", "950"))

        original_history = alice.get(f"/api/rag/sessions/{taas_session}/messages")
        original_history.raise_for_status()
        original_question_id = str(original_history.json()[0]["id"])
        edited_events = _stream_chat(
            alice,
            taas_session,
            "EuCr2As2 单晶怎么做？",
            replace_message_id=original_question_id,
        )
        edited_final = _final(edited_events)
        assert edited_final["evidence_kind"] == "literature_record"
        assert {item["doi"] for item in edited_final["citations"]} == {
            "10.5555/e2e.eucr2as2.001"
        }
        edited_history = alice.get(f"/api/rag/sessions/{taas_session}/messages")
        edited_history.raise_for_status()
        visible_history = edited_history.json()
        assert [item["role"] for item in visible_history] == ["user", "assistant"]
        assert visible_history[0]["content"] == "EuCr2As2 单晶怎么做？"
        assert "TaAs 单晶怎么做" not in "\n".join(item["content"] for item in visible_history)
        assert "它的生长温度" not in "\n".join(item["content"] for item in visible_history)

        prediction_session = _new_session(alice)
        prediction_events = _stream_chat(alice, prediction_session, "我要做 Mn3ZnN")
        prediction_final = _final(prediction_events)
        prediction_nodes = _node_names(prediction_events)
        prediction_outcome = _event_data(prediction_events, "retrieval_outcome")[-1]

        assert prediction_final["evidence_kind"] == "model_prediction"
        assert prediction_outcome["status"] == "insufficient"
        assert "material_mismatch" in prediction_outcome["reason_codes"]
        assert "plan_retrieval" in prediction_nodes
        assert "retrieve_records" in prediction_nodes
        assert "run_prediction" in prediction_nodes
        assert _event_data(prediction_events, "prediction_started")
        assert prediction_final["citations"] == []
        prediction = prediction_final["prediction"]
        assert prediction is not None
        assert prediction["formula_std"] == "Mn3ZnN"
        assert 1 <= len(prediction["routes"]) <= 3
        assert "未由当前真实文献或实验记录验证" in prediction_final["answer"]

        limits_session = _new_session(alice)
        limits_events = _stream_chat(alice, limits_session, "请给出 Mn3GaN 单晶生长的文献证据和 DOI")
        limits_final = _final(limits_events)
        limits_nodes = _node_names(limits_events)
        assert limits_final["evidence_kind"] is None
        assert limits_final["prediction"] is None
        assert limits_final["citations"] == []
        assert "answer_with_limits" in limits_nodes
        assert "run_prediction" not in limits_nodes
        _assert_no_untrusted_dois(limits_final["answer"], set())

        memory_session = _new_session(alice)
        memory_events = _stream_chat(alice, memory_session, "请记住我最高炉温为 1200 C")
        memory_final = _final(memory_events)
        assert memory_final["memory"]["long_term_written"] is True

        bob_login = _login(bob, _email("bob"))
        bob_id = str(bob_login["user"]["id"])
        assert bob_id != alice_id
        assert bob.get(f"/api/rag/sessions/{taas_session}/messages").status_code == 404
        blocked_edit = bob.post(
            "/api/rag/chat/stream",
            json={
                "session_id": taas_session,
                "message": "不应修改其他用户的问题。",
                "replace_message_id": original_question_id,
            },
        )
        assert blocked_edit.status_code == 404

        store = _memory_store()
        alice_memories = store.load_long_memories(
            user_id=alice_id, query="最高炉温", context_hints=[]
        )
        bob_memories = store.load_long_memories(user_id=bob_id, query="最高炉温", context_hints=[])
        assert any("1200" in item["content"] for item in alice_memories)
        assert not any("1200" in item["content"] for item in bob_memories)
