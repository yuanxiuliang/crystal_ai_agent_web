from __future__ import annotations

import hashlib
import re
import secrets
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from ..config import Settings, settings
from ..persistence.database import Database


_EMAIL_RE = re.compile(r"^[^@\s]{1,120}@[^@\s]{1,120}\.[^@\s]{2,63}$")


class AccountInputError(ValueError):
    """The user supplied an invalid email address or password."""


class InvalidCredentials(ValueError):
    """The supplied password does not match an existing account."""


@dataclass(frozen=True)
class Account:
    id: str
    email: str
    created_at: str


@dataclass(frozen=True)
class LoginResult:
    account: Account
    created: bool


class AccountStore:
    """Account and opaque login-session persistence without email ownership verification."""

    def __init__(self, database_url: str, config: Settings = settings) -> None:
        self.database = Database(database_url)
        self.config = config
        self.password_hasher = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1)
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

    def authenticate_or_register(self, *, email: str, password: str) -> LoginResult:
        self.ensure_schema()
        normalized_email = normalize_email(email)
        validate_password(password)
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
                (normalized_email,),
            )
            row = cursor.fetchone()
            if row is None:
                account = Account(id=uuid4().hex, email=normalized_email, created_at=_utc_now())
                self.database.execute(
                    cursor,
                    """
                    INSERT INTO users (id, email, password_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        account.id,
                        account.email,
                        self.password_hasher.hash(password),
                        account.created_at,
                        account.created_at,
                    ),
                )
                connection.commit()
                return LoginResult(account=account, created=True)

            password_hash = str(row[2])
            try:
                password_valid = self.password_hasher.verify(password_hash, password)
            except (InvalidHashError, VerifyMismatchError) as exc:
                raise InvalidCredentials("Email or password is incorrect.") from exc
            if not password_valid:
                raise InvalidCredentials("Email or password is incorrect.")

            account = Account(id=str(row[0]), email=str(row[1]), created_at=str(row[3]))
            if self.password_hasher.check_needs_rehash(password_hash):
                self.database.execute(
                    cursor,
                    "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                    (self.password_hasher.hash(password), _utc_now(), account.id),
                )
            connection.commit()
            return LoginResult(account=account, created=False)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_login_session(self, *, account_id: str) -> str:
        self.ensure_schema()
        token = secrets.token_urlsafe(32)
        now = _utc_now()
        expires_at = _utc_after(days=max(1, min(90, self.config.auth_session_days)))
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                INSERT INTO auth_sessions (id, user_id, token_hash, created_at, last_seen_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (uuid4().hex, account_id, _digest_token(token), now, now, expires_at),
            )
            connection.commit()
            return token
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_account_for_token(self, token: str) -> Account | None:
        if not token:
            return None
        self.ensure_schema()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                SELECT users.id, users.email, users.created_at, auth_sessions.id
                FROM auth_sessions
                JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.token_hash = ?
                  AND auth_sessions.revoked_at IS NULL
                  AND auth_sessions.expires_at > ?
                """,
                (_digest_token(token), _utc_now()),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            self.database.execute(
                cursor,
                "UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?",
                (_utc_now(), str(row[3])),
            )
            connection.commit()
            return Account(id=str(row[0]), email=str(row[1]), created_at=str(row[2]))
        finally:
            connection.close()

    def revoke_login_session(self, token: str) -> None:
        if not token:
            return
        self.ensure_schema()
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                """
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (_utc_now(), _digest_token(token)),
            )
            connection.commit()
        finally:
            connection.close()


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if not _EMAIL_RE.fullmatch(email):
        raise AccountInputError("请输入有效的邮箱地址。")
    return email


def validate_password(value: str) -> None:
    if len(value) < 10:
        raise AccountInputError("密码至少需要 10 个字符。")
    if len(value) > 128:
        raise AccountInputError("密码不能超过 128 个字符。")


def _digest_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _utc_after(*, days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        token_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        revoked_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_expiry ON auth_sessions (user_id, expires_at)",
)


_default_store: AccountStore | None = None
_default_store_lock = threading.Lock()


def get_default_account_store() -> AccountStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AccountStore(settings.memory_database_url)
    return _default_store
