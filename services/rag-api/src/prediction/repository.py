from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .contracts import PredictionExecutionRequest, PredictionModelInfo


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return decoded if isinstance(decoded, list) else []


class PredictionRunRepository:
    """Prediction model and run persistence scoped only by the trusted user identity."""

    def __init__(self, database_url: str, *, postgres_connect_timeout_seconds: int = 3) -> None:
        self.database_url = database_url
        self.postgres_connect_timeout_seconds = max(1, min(15, postgres_connect_timeout_seconds))
        self.kind, self.location = self._parse_database_url(database_url)
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @staticmethod
    def _parse_database_url(value: str) -> tuple[str, str]:
        if value.startswith("sqlite:///"):
            return "sqlite", value.removeprefix("sqlite:///")
        if value.startswith(("postgresql://", "postgres://")):
            return "postgres", value
        raise ValueError(
            f"PREDICTION_DATABASE_URL must use sqlite:///... or postgresql://...; got {value!r}."
        )

    def _connect(self):
        if self.kind == "sqlite":
            if self.location != ":memory:":
                Path(self.location).expanduser().parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.location, timeout=5, check_same_thread=False)
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL prediction persistence requires the rag-api [postgres] dependencies."
            ) from exc
        return psycopg.connect(self.location, connect_timeout=self.postgres_connect_timeout_seconds)

    def _sql(self, query: str) -> str:
        return query if self.kind == "sqlite" else query.replace("?", "%s")

    def _execute(self, cursor: Any, query: str, params: Iterable[Any] = ()) -> None:
        cursor.execute(self._sql(query), tuple(params))

    def _db_json(self, value: Any) -> Any:
        if self.kind == "sqlite":
            return _json(value)
        try:
            from psycopg.types.json import Jsonb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("PostgreSQL prediction persistence requires psycopg.") from exc
        return Jsonb(value)

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
                    json_type = "TEXT"
                else:
                    json_type = "JSONB"
                cursor = connection.cursor()
                self._execute(
                    cursor,
                    f"""
                    CREATE TABLE IF NOT EXISTS prediction_models (
                        model_id TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        artifact_digest TEXT NOT NULL,
                        supported_methods_json {json_type} NOT NULL,
                        manifest_json {json_type} NOT NULL,
                        parameter_count INTEGER NOT NULL,
                        registered_at TEXT NOT NULL,
                        PRIMARY KEY (model_id, model_version, artifact_digest)
                    )
                    """,
                )
                self._execute(
                    cursor,
                    f"""
                    CREATE TABLE IF NOT EXISTS prediction_runs (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        message_id TEXT,
                        source TEXT NOT NULL,
                        formula TEXT NOT NULL,
                        formula_std TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        artifact_digest TEXT NOT NULL,
                        request_json {json_type} NOT NULL,
                        result_json {json_type},
                        warnings_json {json_type} NOT NULL,
                        status TEXT NOT NULL,
                        runtime_ms INTEGER,
                        error_code TEXT,
                        created_at TEXT NOT NULL,
                        completed_at TEXT
                    )
                    """,
                )
                self._execute(
                    cursor,
                    """
                    CREATE INDEX IF NOT EXISTS idx_prediction_runs_user_created
                    ON prediction_runs (user_id, created_at DESC)
                    """,
                )
                self._execute(
                    cursor,
                    """
                    CREATE INDEX IF NOT EXISTS idx_prediction_runs_status_created
                    ON prediction_runs (status, created_at)
                    """,
                )
                connection.commit()
                self._schema_ready = True
            finally:
                connection.close()

    def register_model(self, model: PredictionModelInfo, manifest: dict[str, Any]) -> None:
        self.ensure_schema()
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                INSERT INTO prediction_models (
                    model_id, model_version, artifact_digest, supported_methods_json,
                    manifest_json, parameter_count, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, model_version, artifact_digest) DO UPDATE SET
                    supported_methods_json = excluded.supported_methods_json,
                    manifest_json = excluded.manifest_json,
                    parameter_count = excluded.parameter_count
                """,
                (
                    model.model_id,
                    model.model_version,
                    model.artifact_digest,
                    self._db_json(model.supported_methods),
                    self._db_json(manifest),
                    model.parameter_count,
                    now,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def create_run(
        self,
        request: PredictionExecutionRequest,
        *,
        formula_std: str,
        model: PredictionModelInfo,
    ) -> str:
        self.ensure_schema()
        run_id = f"prediction-{uuid4().hex}"
        now = _utc_now()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                INSERT INTO prediction_runs (
                    id, user_id, session_id, message_id, source, formula, formula_std,
                    model_id, model_version, artifact_digest, request_json, result_json,
                    warnings_json, status, runtime_ms, error_code, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 'running', NULL, NULL, ?, NULL)
                """,
                (
                    run_id,
                    request.user_id,
                    request.session_id,
                    request.message_id,
                    request.source,
                    request.formula,
                    formula_std,
                    model.model_id,
                    model.model_version,
                    model.artifact_digest,
                    self._db_json(
                        {
                            "formula": request.formula,
                            "session_id": request.session_id,
                            "message_id": request.message_id,
                            "source": request.source,
                        }
                    ),
                    self._db_json([]),
                    now,
                ),
            )
            connection.commit()
            return run_id
        finally:
            connection.close()

    def complete_run(
        self,
        run_id: str,
        *,
        result: dict[str, Any],
        warnings: list[str],
        runtime_ms: int,
    ) -> None:
        self._finish_run(
            run_id,
            status="completed",
            result=result,
            warnings=warnings,
            runtime_ms=runtime_ms,
            error_code=None,
        )

    def fail_run(self, run_id: str, *, error_code: str, warnings: list[str]) -> None:
        self._finish_run(
            run_id,
            status="failed",
            result=None,
            warnings=warnings,
            runtime_ms=None,
            error_code=error_code[:120],
        )

    def _finish_run(
        self,
        run_id: str,
        *,
        status: str,
        result: dict[str, Any] | None,
        warnings: list[str],
        runtime_ms: int | None,
        error_code: str | None,
    ) -> None:
        self.ensure_schema()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                UPDATE prediction_runs
                SET status = ?, result_json = ?, warnings_json = ?, runtime_ms = ?,
                    error_code = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    self._db_json(result) if result is not None else None,
                    self._db_json(warnings),
                    runtime_ms,
                    error_code,
                    _utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise KeyError(f"Prediction run not found: {run_id}")
            connection.commit()
        finally:
            connection.close()

    def list_runs(self, *, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        self.ensure_schema()
        bounded_limit = max(1, min(100, limit))
        connection = self._connect()
        try:
            cursor = connection.cursor()
            self._execute(
                cursor,
                """
                SELECT id, user_id, session_id, message_id, source, formula, formula_std,
                       model_id, model_version, artifact_digest, result_json, warnings_json,
                       status, runtime_ms, error_code, created_at, completed_at
                FROM prediction_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, bounded_limit),
            )
            return [
                {
                    "prediction_run_id": str(row[0]),
                    "user_id": str(row[1]),
                    "session_id": str(row[2]) if row[2] else None,
                    "message_id": str(row[3]) if row[3] else None,
                    "source": str(row[4]),
                    "formula": str(row[5]),
                    "formula_std": str(row[6]),
                    "model": {
                        "model_id": str(row[7]),
                        "model_version": str(row[8]),
                        "artifact_digest": str(row[9]),
                    },
                    "result": _json_dict(row[10]) if row[10] else None,
                    "warnings": [str(item) for item in _json_list(row[11])],
                    "status": str(row[12]),
                    "runtime_ms": int(row[13]) if row[13] is not None else None,
                    "error_code": str(row[14]) if row[14] else None,
                    "created_at": str(row[15]),
                    "completed_at": str(row[16]) if row[16] else None,
                }
                for row in cursor.fetchall()
            ]
        finally:
            connection.close()
