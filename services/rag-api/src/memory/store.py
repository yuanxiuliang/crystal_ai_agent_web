from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from ..config import Settings, settings


MEMORY_TYPES = {"preference", "constraint", "research_profile", "project_digest", "confirmed_fact"}
MEMORY_SOURCES = {"user_confirmed", "explicit_user_request", "inferred"}
LEGACY_MEMORY_TYPES = {
    "research_interest": "research_profile",
    "lab_constraint": "constraint",
    "fact": "confirmed_fact",
}
LEGACY_MEMORY_SOURCES = {"explicit_remember_request": "explicit_user_request"}


@dataclass(frozen=True)
class MemoryLimits:
    short_max_messages: int
    summary_max_chars: int
    message_max_chars: int
    active_context_max_items: int
    long_max_items_per_user: int
    long_prompt_max_items: int
    long_prompt_max_chars: int
    session_ttl_days: int
    event_ttl_days: int
    cleanup_interval_seconds: int
    postgres_connect_timeout_seconds: int
    checkpoint_thread_rollover_turns: int = 100
    semantic_candidate_limit: int = 24
    session_material_history_max_items: int = 40

    @classmethod
    def from_settings(cls, value: Settings) -> "MemoryLimits":
        return cls(
            short_max_messages=max(2, min(20, value.memory_short_max_messages)),
            summary_max_chars=max(200, min(4000, value.memory_summary_max_chars)),
            message_max_chars=max(200, min(12000, value.memory_message_max_chars)),
            active_context_max_items=max(1, min(20, value.memory_active_context_max_items)),
            session_material_history_max_items=max(
                4, min(100, value.memory_session_material_history_max_items)
            ),
            long_max_items_per_user=max(10, min(1000, value.memory_long_max_items_per_user)),
            long_prompt_max_items=max(1, min(12, value.memory_long_prompt_max_items)),
            long_prompt_max_chars=max(300, min(4000, value.memory_long_prompt_max_chars)),
            session_ttl_days=max(1, min(365, value.memory_session_ttl_days)),
            event_ttl_days=max(1, min(365, value.memory_event_ttl_days)),
            cleanup_interval_seconds=max(60, value.memory_cleanup_interval_seconds),
            postgres_connect_timeout_seconds=max(
                1, min(15, value.memory_postgres_connect_timeout_seconds)
            ),
            checkpoint_thread_rollover_turns=max(1, min(1000, value.memory_thread_rollover_turns)),
            semantic_candidate_limit=max(1, min(100, value.memory_semantic_candidate_limit)),
        )


@dataclass(frozen=True)
class SessionSnapshot:
    messages: list[dict[str, Any]]
    conversation_summary: str | None
    active_context: dict[str, Any]
    short_memory: dict[str, Any]


@dataclass(frozen=True)
class CheckpointSession:
    """The active LangGraph thread for one user's visible session."""

    graph_thread_id: str
    turn_count: int


@dataclass(frozen=True)
class PersistResult:
    written: bool
    reason: str
    memory_id: str | None = None


class MemoryStore:
    """Bounded persistence for a resource-constrained RAG process.

    SQLite is the default because it adds no resident service on the 1 GiB host. The same
    SQL schema also works with an external PostgreSQL instance when psycopg is installed.
    """

    def __init__(self, database_url: str, limits: MemoryLimits) -> None:
        self.database_url = database_url
        self.limits = limits
        self.kind, self.location = self._parse_database_url(database_url)
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = 0.0

    @staticmethod
    def _parse_database_url(value: str) -> tuple[str, str]:
        if value.startswith("sqlite:///"):
            return "sqlite", value.removeprefix("sqlite:///")
        if value.startswith("postgresql://") or value.startswith("postgres://"):
            return "postgres", value
        raise ValueError(
            f"MEMORY_DATABASE_URL must use sqlite:///... or postgresql://...; got {value!r}."
        )

    def _connect(self):
        if self.kind == "sqlite":
            if self.location != ":memory:":
                Path(self.location).expanduser().parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.location, timeout=5, check_same_thread=False)
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA cache_size = -2000")
            return connection

        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL memory storage requires psycopg. Install the rag-api dependencies first."
            ) from exc
        return psycopg.connect(
            self.location, connect_timeout=self.limits.postgres_connect_timeout_seconds
        )

    def _sql(self, query: str) -> str:
        return query if self.kind == "sqlite" else query.replace("?", "%s")

    def _execute(self, cursor: Any, query: str, params: Iterable[Any] = ()) -> None:
        cursor.execute(self._sql(query), tuple(params))

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection = self._connect()
            try:
                if self.kind == "sqlite":
                    connection.execute("PRAGMA journal_mode = WAL")
                    connection.execute("PRAGMA synchronous = NORMAL")
                    connection.execute("PRAGMA wal_autocheckpoint = 100")
                cursor = connection.cursor()
                for statement in _SCHEMA:
                    self._execute(cursor, statement)
                self._ensure_schema_extensions(connection, cursor)
                connection.commit()
                self._schema_ready = True
            finally:
                connection.close()

    def _ensure_schema_extensions(self, connection: Any, cursor: Any) -> None:
        if self.kind == "postgres":
            for statement in _POSTGRES_SCHEMA:
                self._execute(cursor, statement)
            return

        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(memory_items)").fetchall()
        }
        for name, definition in _SQLITE_MEMORY_COLUMNS.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE memory_items ADD COLUMN {name} {definition}")

    def load_session(self, user_id: str, session_id: str) -> SessionSnapshot | None:
        self.ensure_schema()
        self._maybe_cleanup()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                SELECT messages_json, conversation_summary, active_context_json, short_memory_json
                FROM session_memory
                WHERE user_id = ? AND session_id = ? AND expires_at > ?
                """,
                (user_id, session_id, _utc_now()),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return SessionSnapshot(
                messages=_json_list(row[0]),
                conversation_summary=_optional_text(row[1]),
                active_context=_json_dict(row[2]),
                short_memory=_json_dict(row[3]),
            )
        finally:
            connection.close()

    def save_session(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        conversation_summary: str | None,
        active_context: dict[str, Any],
        short_memory: dict[str, Any],
    ) -> PersistResult:
        self.ensure_schema()
        messages = _bounded_messages(
            messages, self.limits.short_max_messages, self.limits.message_max_chars
        )
        conversation_summary = (
            _clip_text(conversation_summary, self.limits.summary_max_chars)
            if conversation_summary
            else None
        )
        active_context = _bounded_context(active_context, self.limits.active_context_max_items)
        short_memory = _bounded_short_memory(short_memory, self.limits)
        now = _utc_now()
        expires_at = _utc_after(days=self.limits.session_ttl_days)
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                INSERT INTO session_memory (
                    user_id, session_id, messages_json, conversation_summary,
                    active_context_json, short_memory_json, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    messages_json = excluded.messages_json,
                    conversation_summary = excluded.conversation_summary,
                    active_context_json = excluded.active_context_json,
                    short_memory_json = excluded.short_memory_json,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    user_id,
                    session_id,
                    _json(messages),
                    conversation_summary,
                    _json(active_context),
                    _json(short_memory),
                    now,
                    expires_at,
                ),
            )
            connection.commit()
            return PersistResult(True, "session_saved")
        except Exception as exc:
            connection.rollback()
            return PersistResult(False, f"session_save_failed:{type(exc).__name__}")
        finally:
            connection.close()

    def get_or_create_checkpoint_session(
        self,
        *,
        user_id: str,
        session_id: str,
        initial_thread_id: str,
    ) -> CheckpointSession:
        """Return the active short-memory thread without mixing user identities.

        This table is only a mapping and counter. The actual short-term state remains in
        LangGraph's PostgreSQL checkpoint tables.
        """
        self.ensure_schema()
        now = _utc_now()
        expires_at = _utc_after(days=self.limits.session_ttl_days)
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                SELECT graph_thread_id, turn_count, expires_at
                FROM checkpoint_sessions
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            )
            row = cursor.fetchone()
            if row and str(row[2]) > now:
                self._execute(
                    cursor,
                    """
                    UPDATE checkpoint_sessions SET updated_at = ?, expires_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (now, expires_at, user_id, session_id),
                )
                connection.commit()
                return CheckpointSession(graph_thread_id=str(row[0]), turn_count=int(row[1]))

            if row:
                self._execute(
                    cursor,
                    """
                    DELETE FROM checkpoint_sessions
                    WHERE user_id = ? AND session_id = ? AND expires_at <= ?
                    """,
                    (user_id, session_id, now),
                )
            self._execute(
                cursor,
                """
                INSERT INTO checkpoint_sessions (
                    user_id, session_id, graph_thread_id, turn_count, updated_at, expires_at
                ) VALUES (?, ?, ?, 0, ?, ?)
                ON CONFLICT(user_id, session_id) DO NOTHING
                """,
                (user_id, session_id, initial_thread_id, now, expires_at),
            )
            self._execute(
                cursor,
                """
                SELECT graph_thread_id, turn_count
                FROM checkpoint_sessions
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            )
            active = cursor.fetchone()
            connection.commit()
            if not active:
                raise RuntimeError("checkpoint_session_create_failed")
            return CheckpointSession(graph_thread_id=str(active[0]), turn_count=int(active[1]))
        finally:
            connection.close()

    def complete_checkpoint_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        graph_thread_id: str,
    ) -> bool:
        """Count a completed graph run only while its thread is still active."""
        self.ensure_schema()
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                UPDATE checkpoint_sessions
                SET turn_count = turn_count + 1, updated_at = ?, expires_at = ?
                WHERE user_id = ? AND session_id = ? AND graph_thread_id = ?
                """,
                (
                    now,
                    _utc_after(days=self.limits.session_ttl_days),
                    user_id,
                    session_id,
                    graph_thread_id,
                ),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def replace_checkpoint_session_thread(
        self,
        *,
        user_id: str,
        session_id: str,
        expected_thread_id: str,
        next_thread_id: str,
    ) -> bool:
        """Atomically move a session to a seeded replacement LangGraph thread."""
        self.ensure_schema()
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                UPDATE checkpoint_sessions
                SET graph_thread_id = ?, turn_count = 0, updated_at = ?, expires_at = ?
                WHERE user_id = ? AND session_id = ? AND graph_thread_id = ?
                """,
                (
                    next_thread_id,
                    now,
                    _utc_after(days=self.limits.session_ttl_days),
                    user_id,
                    session_id,
                    expected_thread_id,
                ),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def take_expired_checkpoint_threads(self, limit: int = 100) -> list[str]:
        """Remove expired mappings and return their LangGraph threads for safe deletion."""
        self.ensure_schema()
        limit = max(1, min(1000, limit))
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                SELECT user_id, session_id, graph_thread_id
                FROM checkpoint_sessions
                WHERE expires_at <= ?
                ORDER BY expires_at
                LIMIT ?
                """,
                (now, limit),
            )
            rows = cursor.fetchall()
            expired_thread_ids: list[str] = []
            for row in rows:
                self._execute(
                    cursor,
                    """
                    DELETE FROM checkpoint_sessions
                    WHERE user_id = ? AND session_id = ? AND graph_thread_id = ? AND expires_at <= ?
                    """,
                    (str(row[0]), str(row[1]), str(row[2]), now),
                )
                if cursor.rowcount == 1:
                    expired_thread_ids.append(str(row[2]))
            connection.commit()
            return expired_thread_ids
        finally:
            connection.close()

    def load_long_memories(
        self,
        *,
        user_id: str,
        query: str,
        context_hints: Iterable[str] = (),
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        self._maybe_cleanup()
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                SELECT id, memory_type, content, source, confidence, created_at, updated_at,
                       memory_key, importance, last_used_at, embedding_json
                FROM memory_items
                WHERE user_id = ? AND status = 'active' AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, now, self.limits.long_max_items_per_user),
            )
            rows = cursor.fetchall()
            semantic_scores = self._semantic_scores(
                connection, cursor, user_id, now, query_embedding, rows
            )
            selected = self._select_relevant(rows, query, context_hints, semantic_scores)
            if selected:
                self._execute_many(
                    cursor,
                    "UPDATE memory_items SET last_used_at = ? WHERE id = ?",
                    [(now, item["memory_id"]) for item in selected],
                )
                connection.commit()
            return selected
        finally:
            connection.close()

    def upsert_memory(
        self,
        *,
        user_id: str,
        memory_type: str,
        memory_key: str,
        content: str,
        source: str,
        confidence: float,
        importance: int = 50,
        expires_at: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        value_json: dict[str, Any] | None = None,
    ) -> PersistResult:
        memory_type = LEGACY_MEMORY_TYPES.get(memory_type, memory_type)
        source = LEGACY_MEMORY_SOURCES.get(source, source)
        if memory_type not in MEMORY_TYPES:
            return PersistResult(False, "invalid_memory_type")
        if source not in MEMORY_SOURCES:
            return PersistResult(False, "invalid_memory_source")
        memory_key = _clean_key(memory_key)
        content = _clip_text(content, 500)
        if not memory_key or not content:
            return PersistResult(False, "empty_memory")

        self.ensure_schema()
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._delete_expired(cursor, now)
            self._execute(
                cursor,
                """
                SELECT id FROM memory_items
                WHERE user_id = ? AND memory_type = ? AND memory_key = ?
                """,
                (user_id, memory_type, memory_key),
            )
            existing = cursor.fetchone()
            if existing:
                memory_id = str(existing[0])
                self._execute(
                    cursor,
                    """
                    UPDATE memory_items SET content = ?, source = ?, confidence = ?, importance = ?,
                        status = 'active', updated_at = ?, expires_at = ?, subject = ?, predicate = ?,
                        value_json = ?, embedding = NULL, embedding_json = NULL,
                        embedding_updated_at = NULL
                    WHERE id = ?
                    """,
                    (
                        content,
                        source,
                        max(0.0, min(1.0, confidence)),
                        max(0, min(100, importance)),
                        now,
                        expires_at,
                        _clip_text(subject or "", 120) or None,
                        _clip_text(predicate or "", 120) or None,
                        _json(value_json) if value_json is not None else None,
                        memory_id,
                    ),
                )
                self._record_event(cursor, memory_id, "updated", now)
                self._enqueue_embedding(cursor, memory_id, now)
                connection.commit()
                return PersistResult(True, "updated", memory_id)

            self._execute(
                cursor,
                "SELECT COUNT(*) FROM memory_items WHERE user_id = ? AND status = 'active'",
                (user_id,),
            )
            active_count = int(cursor.fetchone()[0])
            if active_count >= self.limits.long_max_items_per_user:
                connection.rollback()
                return PersistResult(False, "active_memory_quota_reached")

            memory_id = f"mem-{uuid4().hex}"
            self._execute(
                cursor,
                """
                INSERT INTO memory_items (
                    id, user_id, memory_type, memory_key, content, source, confidence, importance,
                    status, created_at, updated_at, last_used_at, expires_at, subject, predicate, value_json,
                    embedding, embedding_json, embedding_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    memory_id,
                    user_id,
                    memory_type,
                    memory_key,
                    content,
                    source,
                    max(0.0, min(1.0, confidence)),
                    max(0, min(100, importance)),
                    now,
                    now,
                    expires_at,
                    _clip_text(subject or "", 120) or None,
                    _clip_text(predicate or "", 120) or None,
                    _json(value_json) if value_json is not None else None,
                ),
            )
            self._record_event(cursor, memory_id, "created", now)
            self._enqueue_embedding(cursor, memory_id, now)
            connection.commit()
            return PersistResult(True, "created", memory_id)
        except Exception as exc:
            connection.rollback()
            return PersistResult(False, f"memory_write_failed:{type(exc).__name__}")
        finally:
            connection.close()

    def _select_relevant(
        self,
        rows: Iterable[tuple[Any, ...]],
        query: str,
        context_hints: Iterable[str],
        semantic_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        terms = _terms(query, context_hints)
        ranked: list[tuple[int, tuple[Any, ...]]] = []
        type_weight = {
            "constraint": 80,
            "preference": 70,
            "research_profile": 50,
            "project_digest": 45,
            "confirmed_fact": 35,
        }
        for row in rows:
            memory_type = str(row[1])
            content = str(row[2])
            memory_key = str(row[7])
            haystack = f"{memory_key} {content}".lower()
            overlap = sum(1 for term in terms if term in haystack)
            semantic_score = (semantic_scores or {}).get(str(row[0]), 0.0)
            score = (
                int(row[8])
                + type_weight.get(memory_type, 0)
                + overlap * 20
                + int(semantic_score * 40)
            )
            ranked.append((score, row))
        ranked.sort(key=lambda item: (item[0], str(item[1][6])), reverse=True)

        selected: list[dict[str, Any]] = []
        used_chars = 0
        for _, row in ranked:
            content = _clip_text(str(row[2]), 500)
            if len(selected) >= self.limits.long_prompt_max_items:
                break
            if selected and used_chars + len(content) > self.limits.long_prompt_max_chars:
                continue
            selected.append(
                {
                    "memory_id": str(row[0]),
                    "type": str(row[1]),
                    "content": content,
                    "source": str(row[3]),
                    "confidence": float(row[4]),
                    "created_at": str(row[5]),
                    "updated_at": str(row[6]),
                }
            )
            used_chars += len(content)
        return selected

    def _semantic_scores(
        self,
        connection: Any,
        cursor: Any,
        user_id: str,
        now: str,
        query_embedding: list[float] | None,
        rows: Iterable[tuple[Any, ...]],
    ) -> dict[str, float]:
        if not query_embedding:
            return {}
        if len(query_embedding) != 384:
            return {}
        if self.kind == "postgres":
            vector = _vector_literal(query_embedding)
            self._execute(
                cursor,
                """
                SELECT id, 1 - (embedding <=> ?::vector) AS similarity
                FROM memory_items
                WHERE user_id = ? AND status = 'active' AND embedding IS NOT NULL
                    AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY embedding <=> ?::vector
                LIMIT ?
                """,
                (vector, user_id, now, vector, self.limits.semantic_candidate_limit),
            )
            return {str(row[0]): max(0.0, float(row[1])) for row in cursor.fetchall()}

        scores: dict[str, float] = {}
        for row in rows:
            embedding = _json_float_list(row[10] if len(row) > 10 else None)
            if len(embedding) == len(query_embedding):
                scores[str(row[0])] = max(
                    0.0, sum(left * right for left, right in zip(embedding, query_embedding))
                )
        return scores

    def _execute_many(self, cursor: Any, query: str, params: list[tuple[Any, ...]]) -> None:
        cursor.executemany(self._sql(query), params)

    def _enqueue_embedding(self, cursor: Any, memory_id: str, now: str) -> None:
        self._execute(
            cursor,
            "DELETE FROM memory_jobs WHERE job_type = 'embed_memory' AND memory_id = ? AND status = 'pending'",
            (memory_id,),
        )
        self._execute(
            cursor,
            """
            INSERT INTO memory_jobs (
                id, job_type, memory_id, payload_json, status, attempts, available_at,
                locked_at, created_at, updated_at, last_error
            ) VALUES (?, 'embed_memory', ?, '{}', 'pending', 0, ?, NULL, ?, ?, NULL)
            """,
            (f"job-{uuid4().hex}", memory_id, now, now, now),
        )

    def claim_memory_jobs(self, limit: int = 4) -> list[dict[str, Any]]:
        self.ensure_schema()
        limit = max(1, min(32, limit))
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            if self.kind == "postgres":
                self._execute(
                    cursor,
                    """
                    WITH claimed AS (
                        SELECT id FROM memory_jobs
                        WHERE status = 'pending' AND available_at <= ?
                        ORDER BY created_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT ?
                    )
                    UPDATE memory_jobs AS jobs
                    SET status = 'running', locked_at = ?, attempts = attempts + 1, updated_at = ?
                    FROM claimed
                    WHERE jobs.id = claimed.id
                    RETURNING jobs.id, jobs.job_type, jobs.memory_id, jobs.payload_json, jobs.attempts
                    """,
                    (now, limit, now, now),
                )
                rows = cursor.fetchall()
            else:
                self._execute(
                    cursor,
                    """
                    SELECT id, job_type, memory_id, payload_json, attempts
                    FROM memory_jobs
                    WHERE status = 'pending' AND available_at <= ?
                    ORDER BY created_at
                    LIMIT ?
                    """,
                    (now, limit),
                )
                rows = cursor.fetchall()
                for row in rows:
                    self._execute(
                        cursor,
                        """
                        UPDATE memory_jobs
                        SET status = 'running', locked_at = ?, attempts = attempts + 1, updated_at = ?
                        WHERE id = ? AND status = 'pending'
                        """,
                        (now, now, row[0]),
                    )
            connection.commit()
            return [
                {
                    "id": str(row[0]),
                    "job_type": str(row[1]),
                    "memory_id": str(row[2]) if row[2] else None,
                    "payload": _json_dict(row[3]),
                    "attempts": int(row[4]),
                }
                for row in rows
            ]
        finally:
            connection.close()

    def get_memory_content(self, memory_id: str) -> str | None:
        self.ensure_schema()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                "SELECT content FROM memory_items WHERE id = ? AND status = 'active'",
                (memory_id,),
            )
            row = cursor.fetchone()
            return str(row[0]) if row else None
        finally:
            connection.close()

    def store_memory_embedding(self, memory_id: str, embedding: list[float]) -> PersistResult:
        if len(embedding) != 384:
            return PersistResult(False, "invalid_embedding_dimension")
        self.ensure_schema()
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            serialized = _json(embedding)
            if self.kind == "postgres":
                self._execute(
                    cursor,
                    """
                    UPDATE memory_items
                    SET embedding = ?::vector, embedding_json = ?, embedding_updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    (_vector_literal(embedding), serialized, now, memory_id),
                )
            else:
                self._execute(
                    cursor,
                    """
                    UPDATE memory_items SET embedding_json = ?, embedding_updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    (serialized, now, memory_id),
                )
            if cursor.rowcount != 1:
                connection.rollback()
                return PersistResult(False, "memory_not_found", memory_id)
            self._record_event(cursor, memory_id, "embedded", now)
            connection.commit()
            return PersistResult(True, "embedded", memory_id)
        except Exception as exc:
            connection.rollback()
            return PersistResult(False, f"embedding_write_failed:{type(exc).__name__}", memory_id)
        finally:
            connection.close()

    def complete_memory_job(self, job_id: str) -> None:
        self._finish_memory_job(job_id, status="completed", last_error=None)

    def fail_memory_job(self, job_id: str, reason: str) -> None:
        self._finish_memory_job(job_id, status="failed", last_error=_clip_text(reason, 500))

    def _finish_memory_job(self, job_id: str, *, status: str, last_error: str | None) -> None:
        self.ensure_schema()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                "UPDATE memory_jobs SET status = ?, updated_at = ?, last_error = ? WHERE id = ?",
                (status, _utc_now(), last_error, job_id),
            )
            connection.commit()
        finally:
            connection.close()

    def _record_event(self, cursor: Any, memory_id: str, event_type: str, occurred_at: str) -> None:
        self._execute(
            cursor,
            "INSERT INTO memory_events (id, memory_id, event_type, occurred_at) VALUES (?, ?, ?, ?)",
            (f"evt-{uuid4().hex}", memory_id, event_type, occurred_at),
        )

    def _maybe_cleanup(self) -> None:
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_cleanup < self.limits.cleanup_interval_seconds:
            return
        with self._cleanup_lock:
            if now_monotonic - self._last_cleanup < self.limits.cleanup_interval_seconds:
                return
            connection = self._connect()
            try:
                cursor = connection.cursor()
                now = _utc_now()
                self._delete_expired(cursor, now)
                self._execute(
                    cursor,
                    "DELETE FROM memory_events WHERE occurred_at <= ?",
                    (_utc_after(days=-self.limits.event_ttl_days),),
                )
                self._execute(
                    cursor,
                    """
                    DELETE FROM memory_jobs
                    WHERE status IN ('completed', 'failed') AND updated_at <= ?
                    """,
                    (_utc_after(days=-self.limits.event_ttl_days),),
                )
                connection.commit()
                self._last_cleanup = now_monotonic
            finally:
                connection.close()

    def _delete_expired(self, cursor: Any, now: str) -> None:
        self._execute(cursor, "DELETE FROM session_memory WHERE expires_at <= ?", (now,))
        self._execute(
            cursor,
            "DELETE FROM memory_items WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _utc_after(*, days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_list(value: Any) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [item for item in decoded if isinstance(item, dict)] if isinstance(decoded, list) else []


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_float_list(value: Any) -> list[float]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(decoded, list):
        return []
    try:
        return [float(item) for item in decoded]
    except (TypeError, ValueError):
        return []


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in embedding) + "]"


def _optional_text(value: Any) -> str | None:
    return str(value) if value else None


def _clip_text(value: str, max_chars: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= max_chars else f"{value[: max_chars - 3]}..."


def _clean_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", value.strip().lower())[:120].strip("_")


def _bounded_messages(
    messages: list[dict[str, Any]], max_messages: int, max_content_chars: int
) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for item in messages[-max_messages:]:
        role = str(item.get("role", "user"))
        if role not in {"user", "assistant", "system", "tool"}:
            role = "user"
        content = _clip_text(str(item.get("content", "")), max_content_chars)
        if not content:
            continue
        bounded.append(
            {
                "role": role,
                "content": content,
                "message_id": _clip_text(str(item.get("message_id", "")), 120),
                "created_at": _clip_text(str(item.get("created_at", "")), 80),
                "metadata": {},
            }
        )
    return bounded


def _bounded_context(value: dict[str, Any], max_items: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    list_fields = {
        "active_materials": 120,
        "active_formulas": 80,
        "active_growth_methods": 80,
        "active_constraints": 200,
        "last_retrieval_record_ids": 120,
    }
    for field, max_chars in list_fields.items():
        raw = value.get(field, [])
        if not isinstance(raw, list):
            raw = []
        result[field] = [_clip_text(str(item), max_chars) for item in raw[-max_items:]]
    current_task = value.get("current_task")
    result["current_task"] = _clip_text(str(current_task), 120) if current_task else None
    return result


def _bounded_short_memory(value: dict[str, Any], limits: MemoryLimits) -> dict[str, Any]:
    confirmed_slots = value.get("confirmed_slots", {})
    if not isinstance(confirmed_slots, dict):
        confirmed_slots = {}
    bounded_slots: dict[str, Any] = {}
    for key, slot_value in list(confirmed_slots.items())[: limits.active_context_max_items]:
        if isinstance(slot_value, list):
            bounded_slots[_clip_text(str(key), 80)] = [
                _clip_text(str(item), 120)
                for item in slot_value[-limits.active_context_max_items :]
            ]
        else:
            bounded_slots[_clip_text(str(key), 80)] = _clip_text(str(slot_value), 240)
    open_questions = value.get("open_questions", [])
    if not isinstance(open_questions, list):
        open_questions = []
    material_history = value.get("material_history", [])
    if not isinstance(material_history, list):
        material_history = []
    bounded_material_history: list[dict[str, str | None]] = []
    known_formulas: set[str] = set()
    for item in material_history:
        if not isinstance(item, dict):
            continue
        formula = _clip_text(str(item.get("formula", "")), 80)
        if not formula or formula in known_formulas:
            continue
        known_formulas.add(formula)
        evidence_kind = item.get("evidence_kind")
        bounded_material_history.append(
            {
                "formula": formula,
                "evidence_kind": (
                    str(evidence_kind)
                    if evidence_kind in {"literature_record", "model_prediction"}
                    else None
                ),
            }
        )
    summary = value.get("conversation_summary")
    return {
        "conversation_summary": _clip_text(str(summary), limits.summary_max_chars)
        if summary
        else None,
        "recent_focus": _clip_text(str(value.get("recent_focus", "")), 240) or None,
        "confirmed_slots": bounded_slots,
        "open_questions": [
            _clip_text(str(item), 240)
            for item in open_questions[-limits.active_context_max_items :]
        ],
        "material_history": bounded_material_history[-limits.session_material_history_max_items :],
        "last_turn_kind": (
            "material_history" if value.get("last_turn_kind") == "material_history" else None
        ),
    }


def _terms(query: str, hints: Iterable[str]) -> set[str]:
    values = [query, *[str(item) for item in hints]]
    terms: set[str] = set()
    for value in values:
        lowered = value.lower()
        terms.update(match.group(0) for match in re.finditer(r"[a-z0-9]{2,}", lowered))
        terms.update(
            match.group(0)
            for match in re.finditer(r"[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)+", value)
        )
    return terms


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS session_memory (
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        messages_json TEXT NOT NULL,
        conversation_summary TEXT,
        active_context_json TEXT NOT NULL,
        short_memory_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        PRIMARY KEY (user_id, session_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_session_memory_expiry ON session_memory (expires_at)",
    """
    CREATE TABLE IF NOT EXISTS checkpoint_sessions (
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        graph_thread_id TEXT NOT NULL,
        turn_count INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        PRIMARY KEY (user_id, session_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_checkpoint_sessions_expiry ON checkpoint_sessions (expires_at)",
    """
    CREATE TABLE IF NOT EXISTS memory_items (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        memory_key TEXT NOT NULL,
        content TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence REAL NOT NULL,
        importance INTEGER NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_used_at TEXT,
        expires_at TEXT,
        UNIQUE (user_id, memory_type, memory_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_items_active
    ON memory_items (user_id, status, importance, updated_at)
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_items_expiry ON memory_items (expires_at)",
    """
    CREATE TABLE IF NOT EXISTS memory_jobs (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        memory_id TEXT,
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        available_at TEXT NOT NULL,
        locked_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_error TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_jobs_ready
    ON memory_jobs (status, available_at, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_events (
        id TEXT PRIMARY KEY,
        memory_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        occurred_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_events_expiry ON memory_events (occurred_at)",
)


_SQLITE_MEMORY_COLUMNS = {
    "subject": "TEXT",
    "predicate": "TEXT",
    "value_json": "TEXT",
    "embedding": "TEXT",
    "embedding_json": "TEXT",
    "embedding_updated_at": "TEXT",
}


_POSTGRES_SCHEMA = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    "ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS subject TEXT",
    "ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS predicate TEXT",
    "ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS value_json JSONB",
    "ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS embedding vector(384)",
    "ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS embedding_json TEXT",
    "ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS embedding_updated_at TEXT",
    """
    CREATE INDEX IF NOT EXISTS idx_memory_items_embedding_hnsw
    ON memory_items USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL
    """,
)


_default_store: MemoryStore | None = None
_default_store_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = MemoryStore(
                    settings.memory_database_url, MemoryLimits.from_settings(settings)
                )
    return _default_store
