from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


RetrievalMode = Literal["dense", "sparse", "hybrid"]
Intent = Literal["direct_answer", "retrieve", "clarify", "smalltalk", "unsupported"]
AnswerMode = Literal["direct", "evidence_grounded", "ask_clarification", "refuse_or_redirect"]


class RuntimeOptions(TypedDict):
    force_retrieve: bool
    top_k: int
    retrieval_mode: RetrievalMode
    model: str | None
    stream_trace: bool
    temperature: float | None


class Message(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    message_id: str
    created_at: str
    metadata: dict[str, Any]


class ActiveContext(TypedDict):
    active_materials: list[str]
    active_formulas: list[str]
    active_growth_methods: list[str]
    active_constraints: list[str]
    last_retrieval_record_ids: list[str]
    current_task: str | None


class ShortMemory(TypedDict):
    conversation_summary: str | None
    recent_focus: str | None
    confirmed_slots: dict[str, Any]
    open_questions: list[str]


MemoryType = Literal["preference", "constraint", "research_profile", "project_digest", "confirmed_fact"]
MemorySource = Literal["user_confirmed", "explicit_user_request", "inferred"]


class LongMemoryItem(TypedDict):
    memory_id: str
    type: MemoryType
    content: str
    source: MemorySource
    confidence: float
    created_at: str
    updated_at: str


class UserUnderstanding(TypedDict):
    normalized_question: str
    task_type: Literal["explain", "retrieve", "compare", "recommend", "summarize", "clarify", "unknown"]
    materials: list[str]
    formulas: list[str]
    growth_methods: list[str]
    temperature_mentions: list[str]
    atmosphere_mentions: list[str]
    precursor_mentions: list[str]
    constraints: list[str]
    missing_slots: list[str]
    confidence: float


class RouteDecision(TypedDict):
    intent: Intent
    should_retrieve: bool
    reason: str
    answer_mode: AnswerMode
    required_slots: list[str]
    missing_slots: list[str]
    confidence: float


class RetrievalFilters(TypedDict):
    material_formula: str | None
    material_name: str | None
    growth_method: str | None
    temperature_min: float | None
    temperature_max: float | None
    atmosphere: str | None
    doi: str | None


class RetrievalPlan(TypedDict):
    query_text: str
    dense_query: str
    sparse_query: str
    filters: RetrievalFilters
    top_k: int
    retrieval_mode: RetrievalMode
    relax_filters_if_empty: bool
    must_have: list[str]
    nice_to_have: list[str]


class RetrievedRecord(TypedDict):
    record_id: str
    score: float
    dense_score: float | None
    sparse_score: float | None
    material_formula: str | None
    material_name: str | None
    growth_method: str | None
    temperature_program: str | None
    atmosphere: str | None
    precursors: list[str]
    doi: str | None
    source_text: str
    source_file: str | None
    matched_fields: list[str]


class EvidenceRecord(TypedDict):
    record_id: str
    score: float
    title: str | None
    material_formula: str | None
    growth_method: str | None
    key_facts: list[str]
    source_text: str
    doi: str | None


class EvidencePack(TypedDict):
    records: list[EvidenceRecord]
    summary: str
    conflicts: list[str]
    missing_fields: list[str]


class EvidenceGrade(TypedDict):
    is_sufficient: bool
    reason: str
    usable_record_ids: list[str]
    missing_evidence: list[str]
    answer_strategy: Literal["single_record", "compare_records", "recommend_with_limits", "insufficient"]
    confidence: float


class Citation(TypedDict):
    record_id: str
    doi: str | None
    source_text: str
    score: float
    fields_used: list[str]


class MemoryCandidate(TypedDict):
    type: MemoryType
    memory_key: str
    content: str
    source: MemorySource
    confidence: float
    write_policy: Literal["write_now", "defer_until_repeated_or_confirmed", "do_not_write"]
    subject: NotRequired[str]
    predicate: NotRequired[str]
    value_json: NotRequired[dict[str, Any]]


class MemoryWriteResult(TypedDict):
    content: str
    written: bool
    reason: str


class TraceEvent(TypedDict):
    node: str
    event: str
    data: dict[str, Any]


class GraphError(TypedDict):
    node: str
    code: str
    message: str
    recoverable: bool


class FinalResponse(TypedDict):
    message_id: str
    session_id: str
    answer: str
    citations: list[Citation]
    route: RouteDecision | None
    retrieval: dict[str, Any] | None
    memory: dict[str, Any]
    errors: list[GraphError]


class GrowthRAGState(TypedDict):
    input_payload: NotRequired[dict[str, Any]]

    user_id: str
    session_id: str
    message_id: str
    user_message: str
    runtime: RuntimeOptions
    short_term_backend: Literal["store", "checkpointer"]
    memory_query_embedding: NotRequired[list[float] | None]

    # The entire bounded window is replaced after compaction. It must not accumulate through
    # LangGraph reducers, or a long-lived thread will eventually grow without bound.
    messages: list[Message]
    conversation_summary: str | None
    active_context: ActiveContext

    short_memory: ShortMemory
    long_memories: list[LongMemoryItem]
    memory_candidates: list[MemoryCandidate]
    memory_writes: list[MemoryWriteResult]
    short_term_persisted: bool

    understanding: UserUnderstanding | None
    route: RouteDecision | None

    retrieval_plan: RetrievalPlan | None
    retrieved_records: list[RetrievedRecord]
    evidence_pack: EvidencePack | None
    evidence_grade: EvidenceGrade | None

    answer_plan: NotRequired[dict[str, Any] | None]
    draft_answer: str | None
    final_answer: str | None
    citations: list[Citation]

    final_response: FinalResponse | None
    trace: list[TraceEvent]
    errors: list[GraphError]
