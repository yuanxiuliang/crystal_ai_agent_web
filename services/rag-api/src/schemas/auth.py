from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class CurrentUserResponse(BaseModel):
    id: str
    email: str


class LoginResponse(BaseModel):
    user: CurrentUserResponse
    created: bool
