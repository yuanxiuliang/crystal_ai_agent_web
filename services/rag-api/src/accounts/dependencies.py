from __future__ import annotations

from fastapi import HTTPException, Request, Response, status

from ..config import settings
from .store import Account, get_default_account_store


def require_current_account(request: Request) -> Account:
    token = request.cookies.get(settings.auth_cookie_name, "")
    account = get_default_account_store().get_account_for_token(token)
    if account is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录。")
    return account


def set_login_cookie(response: Response, token: str) -> None:
    max_age = max(1, min(90, settings.auth_session_days)) * 24 * 60 * 60
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=max_age,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_login_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.auth_cookie_name,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )
