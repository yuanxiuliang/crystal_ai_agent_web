from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request, Response, status

from ..accounts.dependencies import clear_login_cookie, require_current_account, set_login_cookie
from ..accounts.store import AccountInputError, InvalidCredentials, get_default_account_store
from ..config import settings
from ..schemas.auth import CurrentUserResponse, LoginRequest, LoginResponse


router = APIRouter()


class _LoginRateLimiter:
    """Bound password guesses in the single-process RAG API deployment."""

    def __init__(self, *, maximum_attempts: int = 5, window_seconds: int = 900) -> None:
        self.maximum_attempts = maximum_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            attempts = self._attempts[key]
            while attempts and attempts[0] <= now - self.window_seconds:
                attempts.popleft()
            return len(attempts) < self.maximum_attempts

    def failed(self, key: str) -> None:
        with self._lock:
            self._attempts[key].append(time.monotonic())

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)


_login_limiter = _LoginRateLimiter()


def _limiter_key(request: Request, email: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{email.strip().lower()}"


@router.post("/login", response_model=LoginResponse)
async def login(request: Request, response: Response, body: LoginRequest) -> LoginResponse:
    key = _limiter_key(request, body.email)
    if not _login_limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="登录尝试次数过多，请稍后再试。",
        )

    store = get_default_account_store()
    try:
        result = await asyncio.to_thread(
            store.authenticate_or_register, email=body.email, password=body.password
        )
    except AccountInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except InvalidCredentials as exc:
        _login_limiter.failed(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="邮箱或密码错误。"
        ) from exc

    token = await asyncio.to_thread(store.create_login_session, account_id=result.account.id)
    _login_limiter.succeeded(key)
    set_login_cookie(response, token)
    return LoginResponse(
        user=CurrentUserResponse(id=result.account.id, email=result.account.email),
        created=result.created,
    )


@router.get("/me", response_model=CurrentUserResponse)
async def current_user(request: Request) -> CurrentUserResponse:
    account = require_current_account(request)
    return CurrentUserResponse(id=account.id, email=account.email)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request) -> Response:
    token = request.cookies.get(settings.auth_cookie_name, "")
    if token:
        await asyncio.to_thread(get_default_account_store().revoke_login_session, token)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_login_cookie(response)
    return response
