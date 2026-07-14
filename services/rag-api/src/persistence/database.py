from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


class Database:
    """Minimal SQLite/PostgreSQL adapter for application-owned tables."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.kind, self.location = self._parse_url(database_url)

    @staticmethod
    def _parse_url(value: str) -> tuple[str, str]:
        if value.startswith("sqlite:///"):
            return "sqlite", value.removeprefix("sqlite:///")
        if value.startswith(("postgresql://", "postgres://")):
            return "postgres", value
        raise ValueError(f"Database URL must be SQLite or PostgreSQL, got {value!r}.")

    def connect(self) -> Any:
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
                "PostgreSQL persistence requires the rag-api [postgres] extra."
            ) from exc
        return psycopg.connect(self.location, connect_timeout=5)

    def sql(self, query: str) -> str:
        return query if self.kind == "sqlite" else query.replace("?", "%s")

    def execute(self, cursor: Any, query: str, params: Iterable[Any] = ()) -> None:
        cursor.execute(self.sql(query), tuple(params))
