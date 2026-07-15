from __future__ import annotations

import asyncio
import hashlib
from typing import Any, AsyncIterator
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..llm.client import LLMClient
from ..llm.factory import get_default_llm_client
from ..memory.checkpointer import CheckpointerRuntime, get_default_checkpointer_runtime
from ..memory.store import MemoryStore, get_memory_store
from ..prediction.factory import get_default_prediction_service
from ..prediction.service import PredictionService
from ..retrieval.embedding import EmbeddingClient, get_default_embedding_client
from ..retrieval.factory import get_default_retrieval_service
from ..retrieval.service import RetrievalService
from ..streaming.events import StreamEvent
from .nodes import (
    analyze_and_route,
    answer_direct,
    answer_from_prediction,
    answer_with_evidence,
    answer_with_limits,
    assess_prediction_eligibility,
    assess_retrieval_sufficiency,
    ask_clarification,
    build_evidence_pack,
    compact_persistent_state,
    finalize_response,
    load_context,
    load_long_memory,
    plan_retrieval,
    prepare_turn,
    retrieve_records,
    run_prediction,
    update_memory,
)
from .state import GrowthRAGState
from .utils import merge_state


NODE_LABELS = {
    "prepare_turn": "接收并校验输入",
    "load_context": "读取当前会话上下文",
    "load_long_memory": "读取长期记忆",
    "analyze_and_route": "分析问题并判断是否检索",
    "ask_clarification": "生成澄清问题",
    "answer_direct": "生成直接回答",
    "plan_retrieval": "生成检索计划",
    "retrieve_records": "检索单晶生长数据",
    "assess_retrieval_sufficiency": "判定真实记录是否充分",
    "build_evidence_pack": "整理检索证据",
    "assess_prediction_eligibility": "判断是否允许模型回退",
    "run_prediction": "生成候选生长路线",
    "answer_with_evidence": "基于真实记录生成回答",
    "answer_from_prediction": "基于模型候选生成回答",
    "answer_with_limits": "生成受限回答",
    "update_memory": "更新会话记忆",
    "finalize_response": "整理最终响应",
    "compact_persistent_state": "压缩持久化会话状态",
}

ANSWER_NODES = {
    "ask_clarification",
    "answer_direct",
    "answer_with_evidence",
    "answer_from_prediction",
    "answer_with_limits",
}


class GrowthRAGGraph:
    """LangGraph-backed GrowthRAG workflow with the existing StreamEvent contract."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        retrieval: RetrievalService | None = None,
        prediction: PredictionService | None = None,
        memory_store: MemoryStore | None = None,
        checkpointer_runtime: CheckpointerRuntime | None = None,
    ) -> None:
        self.llm = llm or get_default_llm_client()
        # The ONNX embedding session is the largest resident object in this service. Defer its
        # construction until a graph execution actually takes the retrieval branch.
        self.retrieval = retrieval
        self.prediction = prediction
        self.memory_store = memory_store or get_memory_store()
        self.checkpointer_runtime = checkpointer_runtime or get_default_checkpointer_runtime()
        self.memory_embedding_client: EmbeddingClient | None = None
        self._fallback_graph = self._build_graph()

    def _build_graph(self, checkpointer: Any | None = None):
        builder = StateGraph(GrowthRAGState)

        builder.add_node("prepare_turn", self._prepare_turn_node)
        builder.add_node("load_context", self._load_context_node)
        builder.add_node("load_long_memory", self._load_long_memory_node)
        builder.add_node("analyze_and_route", self._analyze_and_route_node)
        builder.add_node("ask_clarification", ask_clarification)
        builder.add_node("answer_direct", self._answer_direct_node)
        builder.add_node("plan_retrieval", plan_retrieval)
        builder.add_node("retrieve_records", self._retrieve_records_node)
        builder.add_node("assess_retrieval_sufficiency", assess_retrieval_sufficiency)
        builder.add_node("build_evidence_pack", build_evidence_pack)
        builder.add_node("assess_prediction_eligibility", assess_prediction_eligibility)
        builder.add_node("run_prediction", self._run_prediction_node)
        builder.add_node("answer_with_evidence", self._answer_with_evidence_node)
        builder.add_node("answer_from_prediction", answer_from_prediction)
        builder.add_node("answer_with_limits", self._answer_with_limits_node)
        builder.add_node("update_memory", self._update_memory_node)
        builder.add_node("finalize_response", finalize_response)
        builder.add_node("compact_persistent_state", compact_persistent_state)

        builder.add_edge(START, "prepare_turn")
        builder.add_conditional_edges(
            "prepare_turn",
            self._after_prepare,
            {"load_context": "load_context", "finalize_response": "finalize_response"},
        )
        builder.add_edge("load_context", "load_long_memory")
        builder.add_edge("load_long_memory", "analyze_and_route")
        builder.add_conditional_edges(
            "analyze_and_route",
            self._after_route,
            {
                "ask_clarification": "ask_clarification",
                "answer_direct": "answer_direct",
                "plan_retrieval": "plan_retrieval",
            },
        )
        builder.add_edge("ask_clarification", "update_memory")
        builder.add_edge("answer_direct", "update_memory")
        builder.add_edge("plan_retrieval", "retrieve_records")
        builder.add_edge("retrieve_records", "assess_retrieval_sufficiency")
        builder.add_conditional_edges(
            "assess_retrieval_sufficiency",
            self._after_retrieval_assessment,
            {
                "build_evidence_pack": "build_evidence_pack",
                "assess_prediction_eligibility": "assess_prediction_eligibility",
                "answer_with_limits": "answer_with_limits",
            },
        )
        builder.add_edge("build_evidence_pack", "answer_with_evidence")
        builder.add_conditional_edges(
            "assess_prediction_eligibility",
            self._after_prediction_eligibility,
            {"run_prediction": "run_prediction", "answer_with_limits": "answer_with_limits"},
        )
        builder.add_conditional_edges(
            "run_prediction",
            self._after_prediction,
            {
                "answer_from_prediction": "answer_from_prediction",
                "answer_with_limits": "answer_with_limits",
            },
        )
        builder.add_edge("answer_with_evidence", "update_memory")
        builder.add_edge("answer_from_prediction", "update_memory")
        builder.add_edge("answer_with_limits", "update_memory")
        builder.add_edge("update_memory", "finalize_response")
        builder.add_edge("finalize_response", "compact_persistent_state")
        builder.add_edge("compact_persistent_state", END)

        return builder.compile(checkpointer=checkpointer)

    async def _prepare_turn_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await prepare_turn(state["input_payload"], state)

    async def _load_context_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await load_context(state, self.memory_store)

    async def _load_long_memory_node(self, state: GrowthRAGState) -> dict[str, Any]:
        query_embedding = None
        if self._semantic_memory_enabled():
            query_embedding = await self._embed_memory_query(state["user_message"])
        return await load_long_memory(
            {**state, "memory_query_embedding": query_embedding}, self.memory_store
        )

    async def _analyze_and_route_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await analyze_and_route(state, self.llm)

    async def _answer_direct_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await answer_direct(state, self.llm)

    async def _retrieve_records_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await retrieve_records(state, self._get_retrieval_service())

    async def _run_prediction_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await run_prediction(state, self._get_prediction_service())

    async def _answer_with_evidence_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await answer_with_evidence(state, self.llm)

    async def _answer_with_limits_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await answer_with_limits(state, self.llm)

    async def _update_memory_node(self, state: GrowthRAGState) -> dict[str, Any]:
        return await update_memory(state, self.memory_store)

    async def reset_session_context(self, *, user_id: str, session_id: str) -> None:
        """Discard the old short-term branch before a user edits an earlier question."""
        replacement_thread_id = self._replacement_thread_id()
        old_thread_id = await asyncio.to_thread(
            self.memory_store.reset_short_term_session,
            user_id=user_id,
            session_id=session_id,
            replacement_thread_id=replacement_thread_id,
        )
        if old_thread_id is None:
            return
        checkpointer = await self.checkpointer_runtime.get()
        if checkpointer is not None and old_thread_id != replacement_thread_id:
            await checkpointer.adelete_thread(old_thread_id)

    def _get_retrieval_service(self) -> RetrievalService:
        if self.retrieval is None:
            self.retrieval = get_default_retrieval_service()
        return self.retrieval

    def _get_prediction_service(self) -> PredictionService:
        if self.prediction is None:
            self.prediction = get_default_prediction_service()
        return self.prediction

    def _semantic_memory_enabled(self) -> bool:
        value = settings.memory_semantic_search_enabled.strip().lower()
        if value == "auto":
            return self.memory_store.kind == "postgres"
        return value in {"1", "true", "yes", "on"}

    async def _embed_memory_query(self, text: str) -> list[float]:
        if self.memory_embedding_client is None:
            self.memory_embedding_client = get_default_embedding_client()
        return await asyncio.to_thread(self.memory_embedding_client.embed_query, text)

    @staticmethod
    def _thread_id(payload: dict[str, Any]) -> str:
        user_id = str(payload.get("user_id") or "demo-user")
        session_id = str(payload.get("session_id") or "demo-session")
        # A LangGraph checkpointer scopes only by thread_id. Include user identity in this
        # opaque key so two users cannot share short-term state by reusing a session label.
        digest = hashlib.sha256(f"{user_id}\x00{session_id}".encode("utf-8")).hexdigest()
        return f"growth-rag-{digest}"

    @staticmethod
    def _replacement_thread_id() -> str:
        return f"growth-rag-{uuid4().hex}"

    @staticmethod
    def _checkpoint_seed_state(values: dict[str, Any]) -> dict[str, Any]:
        """Copy only the bounded state that is allowed to survive a checkpoint rollover."""
        return {
            "user_id": values.get("user_id", "demo-user"),
            "session_id": values.get("session_id", "demo-session"),
            "messages": values.get("messages", []),
            "conversation_summary": values.get("conversation_summary"),
            "active_context": values.get("active_context", {}),
            "short_memory": values.get("short_memory", {}),
        }

    async def _checkpoint_config(
        self,
        graph: Any,
        checkpointer: Any,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Resolve an active thread and roll it over only after state migration succeeds."""
        user_id = str(payload.get("user_id") or "demo-user")
        session_id = str(payload.get("session_id") or "demo-session")
        initial_thread_id = self._thread_id(payload)
        expired_thread_ids = await asyncio.to_thread(
            self.memory_store.take_expired_checkpoint_threads
        )
        for expired_thread_id in expired_thread_ids:
            await checkpointer.adelete_thread(expired_thread_id)
        session = await asyncio.to_thread(
            self.memory_store.get_or_create_checkpoint_session,
            user_id=user_id,
            session_id=session_id,
            initial_thread_id=initial_thread_id,
        )
        rollover_turns = self.memory_store.limits.checkpoint_thread_rollover_turns
        if session.turn_count < rollover_turns:
            return {"configurable": {"thread_id": session.graph_thread_id}}, session.graph_thread_id

        old_config = {"configurable": {"thread_id": session.graph_thread_id}}
        checkpoint_state = await graph.aget_state(old_config)
        next_thread_id = self._replacement_thread_id()
        next_config = {"configurable": {"thread_id": next_thread_id}}
        try:
            if checkpoint_state.values:
                await graph.aupdate_state(
                    next_config,
                    self._checkpoint_seed_state(dict(checkpoint_state.values)),
                    as_node="compact_persistent_state",
                )
            replaced = await asyncio.to_thread(
                self.memory_store.replace_checkpoint_session_thread,
                user_id=user_id,
                session_id=session_id,
                expected_thread_id=session.graph_thread_id,
                next_thread_id=next_thread_id,
            )
        except Exception:
            await checkpointer.adelete_thread(next_thread_id)
            raise

        if replaced:
            await checkpointer.adelete_thread(session.graph_thread_id)
            return next_config, next_thread_id

        # Another API process moved the same session first. The unreferenced seed can be removed.
        await checkpointer.adelete_thread(next_thread_id)
        active = await asyncio.to_thread(
            self.memory_store.get_or_create_checkpoint_session,
            user_id=user_id,
            session_id=session_id,
            initial_thread_id=initial_thread_id,
        )
        return {"configurable": {"thread_id": active.graph_thread_id}}, active.graph_thread_id

    @staticmethod
    def _after_prepare(state: GrowthRAGState) -> str:
        if any(not item["recoverable"] for item in state["errors"]):
            return "finalize_response"
        return "load_context"

    @staticmethod
    def _after_route(state: GrowthRAGState) -> str:
        intent = state["route"]["intent"] if state["route"] else "unsupported"
        if intent == "clarify":
            return "ask_clarification"
        if intent in {"direct_answer", "smalltalk", "unsupported"}:
            return "answer_direct"
        return "plan_retrieval"

    @staticmethod
    def _after_retrieval_assessment(state: GrowthRAGState) -> str:
        outcome = state["retrieval_outcome"]
        if outcome is None:
            return "answer_with_limits"
        if outcome["status"] == "sufficient":
            return "build_evidence_pack"
        if outcome["status"] in {"empty", "insufficient"}:
            return "assess_prediction_eligibility"
        return "answer_with_limits"

    @staticmethod
    def _after_prediction_eligibility(state: GrowthRAGState) -> str:
        eligibility = state["prediction_eligibility"]
        return "run_prediction" if eligibility and eligibility["eligible"] else "answer_with_limits"

    @staticmethod
    def _after_prediction(state: GrowthRAGState) -> str:
        return "answer_from_prediction" if state["prediction_result"] else "answer_with_limits"

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        yield StreamEvent("run_started", {"session_id": payload.get("session_id", "demo-session")})

        # The compiled graph owns execution and conditional routing. This local view only adapts
        # LangGraph updates to the stable event protocol used by the API and CLI.
        checkpointer = await self.checkpointer_runtime.get()
        payload = {
            **payload,
            "_short_term_backend": "checkpointer" if checkpointer is not None else "store",
        }
        graph = (
            self._build_graph(checkpointer) if checkpointer is not None else self._fallback_graph
        )
        state: dict[str, Any] = {"input_payload": payload}
        graph_config = None
        checkpoint_thread_id = None
        if checkpointer is not None:
            graph_config, checkpoint_thread_id = await self._checkpoint_config(
                graph, checkpointer, payload
            )
        updates = graph.astream(state, config=graph_config, stream_mode="updates")
        next_node = "prepare_turn"
        fatal_prepare_error = False
        final_emitted = False

        while True:
            yield StreamEvent("node_started", self._node_event(next_node))
            if next_node == "run_prediction":
                yield StreamEvent("prediction_started", {"source": "retrieval_fallback"})
            try:
                update = await updates.__anext__()
            except StopAsyncIteration:
                break

            if not update:
                continue
            graph_node, patch = next(iter(update.items()))
            if graph_node != next_node:
                raise RuntimeError(
                    f"Unexpected LangGraph node: expected {next_node}, got {graph_node}"
                )

            state = merge_state(state, patch)
            current_state = state
            yield StreamEvent("node_finished", self._finished_node_event(graph_node, current_state))

            if graph_node == "prepare_turn":
                fatal_prepare_error = any(not item["recoverable"] for item in state["errors"])
            elif graph_node == "analyze_and_route":
                yield StreamEvent("route_decision", state["route"] or {})
            elif graph_node == "plan_retrieval" and state["retrieval_plan"]:
                yield StreamEvent("retrieval_plan", state["retrieval_plan"])
            elif graph_node == "retrieve_records":
                for record in state["retrieved_records"]:
                    yield StreamEvent("retrieval_result", record)
            elif graph_node == "assess_retrieval_sufficiency" and state["retrieval_outcome"]:
                yield StreamEvent("retrieval_outcome", state["retrieval_outcome"])
                yield StreamEvent("evidence_grade", state["evidence_grade"] or {})
            elif graph_node == "assess_prediction_eligibility" and state["prediction_eligibility"]:
                yield StreamEvent("prediction_eligible", state["prediction_eligibility"])
            elif graph_node == "run_prediction" and state["prediction_result"]:
                yield StreamEvent("prediction_result", state["prediction_result"])
                for warning in state["prediction_result"].get("warnings", []):
                    yield StreamEvent("prediction_warning", {"message": str(warning)})

            if graph_node in ANSWER_NODES:
                if graph_node == "answer_with_evidence":
                    for citation in state["citations"]:
                        yield StreamEvent("citation", citation)
                if state["final_answer"]:
                    for token in self._chunk_answer(state["final_answer"]):
                        yield StreamEvent("token", {"text": token})

            if graph_node == "update_memory":
                yield StreamEvent(
                    "memory_update",
                    {
                        "short_term_updated": state["short_term_persisted"],
                        "long_term_written": any(
                            item["written"] for item in state["memory_writes"]
                        ),
                    },
                )

            if graph_node == "finalize_response":
                if fatal_prepare_error:
                    yield StreamEvent("error", {"errors": state["errors"]})
                yield StreamEvent("final", state["final_response"] or {})
                final_emitted = True
            elif graph_node == "compact_persistent_state":
                if checkpoint_thread_id is not None:
                    await asyncio.to_thread(
                        self.memory_store.complete_checkpoint_turn,
                        user_id=state["user_id"],
                        session_id=state["session_id"],
                        graph_thread_id=checkpoint_thread_id,
                    )
                if final_emitted:
                    yield StreamEvent("run_finished", {"session_id": state["session_id"]})
                break

            next_node = self._next_node(graph_node, state)

    def _next_node(self, current: str, state: dict[str, Any]) -> str:
        if current == "prepare_turn":
            return "finalize_response" if self._has_fatal_error(state) else "load_context"
        if current == "load_context":
            return "load_long_memory"
        if current == "load_long_memory":
            return "analyze_and_route"
        if current == "analyze_and_route":
            return self._after_route(state)  # type: ignore[arg-type]
        if current in {"ask_clarification", "answer_direct"}:
            return "update_memory"
        if current == "plan_retrieval":
            return "retrieve_records"
        if current == "retrieve_records":
            return "assess_retrieval_sufficiency"
        if current == "assess_retrieval_sufficiency":
            return self._after_retrieval_assessment(state)  # type: ignore[arg-type]
        if current == "build_evidence_pack":
            return "answer_with_evidence"
        if current == "assess_prediction_eligibility":
            return self._after_prediction_eligibility(state)  # type: ignore[arg-type]
        if current == "run_prediction":
            return self._after_prediction(state)  # type: ignore[arg-type]
        if current in {"answer_with_evidence", "answer_from_prediction", "answer_with_limits"}:
            return "update_memory"
        if current == "update_memory":
            return "finalize_response"
        if current == "finalize_response":
            return "compact_persistent_state"
        raise RuntimeError(f"No next node configured after {current}")

    @staticmethod
    def _has_fatal_error(state: dict[str, Any]) -> bool:
        return any(not item["recoverable"] for item in state.get("errors", []))

    def _chunk_answer(self, answer: str, size: int = 28) -> list[str]:
        return [answer[index : index + size] for index in range(0, len(answer), size)]

    def _node_event(self, name: str) -> dict[str, str]:
        return {"node": name, "label": NODE_LABELS.get(name, name)}

    def _finished_node_event(self, name: str, state: dict[str, Any]) -> dict[str, Any]:
        event = self._node_event(name)
        if name == "analyze_and_route":
            return {**event, "understanding": state["understanding"], "route": state["route"]}
        return event
