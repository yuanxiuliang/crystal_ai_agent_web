from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..config import settings
from ..persistence.database import Database


class ConversationStore:
    """Owns user-visible conversation metadata and durable display history."""

    def __init__(self, database_url: str = settings.memory_database_url) -> None:
        self.database = Database(database_url)
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection = self.database.connect()
            try:
                cursor = connection.cursor()
                for statement in _SCHEMA:
                    self.database.execute(cursor, statement)
                connection.commit()
                self._schema_ready = True
            finally:
                connection.close()

    def create_session(self, *, user_id: str) -> dict[str, str]:
        self.ensure_schema()
        now = _utc_now()
        session = {"id": uuid4().hex, "title": "新对话", "created_at": now, "updated_at": now}
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session["id"], user_id, session["title"], now, now),
            )
            connection.commit()
            return session
        finally:
            connection.close()

    def list_sessions(self, *, user_id: str, limit: int = 50) -> list[dict[str, str]]:
        self.ensure_schema()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, max(1, min(100, limit))),
            )
            return [
                {
                    "id": str(row[0]),
                    "title": str(row[1]),
                    "created_at": str(row[2]),
                    "updated_at": str(row[3]),
                }
                for row in cursor.fetchall()
            ]
        finally:
            connection.close()

    def get_session(self, *, user_id: str, session_id: str) -> dict[str, str] | None:
        self.ensure_schema()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ? AND id = ?
                """,
                (user_id, session_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": str(row[0]),
                "title": str(row[1]),
                "created_at": str(row[2]),
                "updated_at": str(row[3]),
            }
        finally:
            connection.close()

    def rename_session(self, *, user_id: str, session_id: str, title: str) -> dict[str, str] | None:
        clean_title = _clean_title(title)
        self.ensure_schema()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                UPDATE chat_sessions SET title = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
                """,
                (clean_title, _utc_now(), user_id, session_id),
            )
            if cursor.rowcount == 0:
                connection.rollback()
                return None
            connection.commit()
        finally:
            connection.close()
        return self.get_session(user_id=user_id, session_id=session_id)

    def delete_session(self, *, user_id: str, session_id: str) -> bool:
        self.ensure_schema()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                "DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            self.database.execute(
                cursor,
                "DELETE FROM chat_sessions WHERE user_id = ? AND id = ?",
                (user_id, session_id),
            )
            deleted = cursor.rowcount > 0
            connection.commit()
            return deleted
        finally:
            connection.close()

    def append_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Chat messages must have user or assistant roles.")
        if not content.strip():
            raise ValueError("Chat message content is required.")
        self.ensure_schema()
        now = _utc_now()
        message = {
            "id": uuid4().hex,
            "role": role,
            "content": content.strip(),
            "created_at": now,
            "response": response,
        }
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                "SELECT title FROM chat_sessions WHERE user_id = ? AND id = ?",
                (user_id, session_id),
            )
            session = cursor.fetchone()
            if session is None:
                raise KeyError("Chat session was not found.")
            self.database.execute(
                cursor,
                """
                INSERT INTO chat_messages (id, user_id, session_id, role, content, response_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message["id"],
                    user_id,
                    session_id,
                    role,
                    message["content"],
                    json.dumps(response, ensure_ascii=False) if response is not None else None,
                    now,
                ),
            )
            title = str(session[0])
            next_title = (
                _title_from_message(message["content"])
                if role == "user" and title == "新对话"
                else title
            )
            self.database.execute(
                cursor,
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE user_id = ? AND id = ?",
                (next_title, now, user_id, session_id),
            )
            connection.commit()
            return message
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def replace_user_message_and_truncate(
        self,
        *,
        user_id: str,
        session_id: str,
        message_id: str,
        content: str,
    ) -> list[dict[str, Any]] | None:
        """Replace one user turn and discard the later display-history branch."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Chat message content is required.")
        self.ensure_schema()
        now = _utc_now()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                SELECT content
                FROM chat_messages
                WHERE id = ? AND user_id = ? AND session_id = ? AND role = 'user'
                """,
                (message_id, user_id, session_id),
            )
            target = cursor.fetchone()
            if target is None:
                connection.rollback()
                return None
            original_content = str(target[0])
            self.database.execute(
                cursor,
                """
                SELECT id
                FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (user_id, session_id),
            )
            ordered_message_ids = [str(row[0]) for row in cursor.fetchall()]
            try:
                target_index = ordered_message_ids.index(message_id)
            except ValueError:
                connection.rollback()
                return None
            self.database.execute(
                cursor,
                """
                UPDATE chat_messages
                SET content = ?, response_json = NULL
                WHERE id = ? AND user_id = ? AND session_id = ?
                """,
                (clean_content, message_id, user_id, session_id),
            )
            for later_message_id in ordered_message_ids[target_index + 1 :]:
                self.database.execute(
                    cursor,
                    """
                    DELETE FROM chat_messages
                    WHERE id = ? AND user_id = ? AND session_id = ?
                    """,
                    (later_message_id, user_id, session_id),
                )
            self.database.execute(
                cursor,
                "SELECT title FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            session = cursor.fetchone()
            if session is None:
                connection.rollback()
                return None
            title = str(session[0])
            next_title = _title_from_message(clean_content) if title == _title_from_message(original_content) else title
            self.database.execute(
                cursor,
                """
                UPDATE chat_sessions SET title = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (next_title, now, session_id, user_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.list_messages(user_id=user_id, session_id=session_id)

    def list_messages(
        self, *, user_id: str, session_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                SELECT id, role, content, response_json, created_at
                FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (user_id, session_id, max(1, min(500, limit))),
            )
            messages: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                response = None
                if row[3]:
                    try:
                        response = json.loads(str(row[3]))
                    except json.JSONDecodeError:
                        response = None
                messages.append(
                    {
                        "id": str(row[0]),
                        "role": str(row[1]),
                        "content": str(row[2]),
                        "response": response,
                        "created_at": str(row[4]),
                    }
                )
            return messages
        finally:
            connection.close()


def _clean_title(value: str) -> str:
    title = " ".join(value.split())
    if not title:
        raise ValueError("会话标题不能为空。")
    return title[:72]


def _title_from_message(value: str) -> str:
    return _clean_title(value)[:36]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated ON chat_sessions (user_id, updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        response_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages (user_id, session_id, created_at)",
)


_default_store: ConversationStore | None = None
_default_store_lock = threading.Lock()


def get_default_conversation_store() -> ConversationStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = ConversationStore()
    return _default_store
