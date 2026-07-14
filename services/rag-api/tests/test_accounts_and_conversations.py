from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.accounts import dependencies as account_dependencies
from src.accounts.store import AccountStore, InvalidCredentials
from src.api import auth as auth_api
from src.api import conversations as conversations_api
from src.conversations.store import ConversationStore
from src.main import app


def test_first_login_registers_then_existing_login_requires_matching_password(tmp_path) -> None:
    store = AccountStore(f"sqlite:///{tmp_path / 'accounts.sqlite3'}")

    first = store.authenticate_or_register(email="Researcher@Example.com", password="correct-pass")
    second = store.authenticate_or_register(email="researcher@example.com", password="correct-pass")

    assert first.created is True
    assert second.created is False
    assert first.account.id == second.account.id
    assert second.account.email == "researcher@example.com"
    with pytest.raises(InvalidCredentials):
        store.authenticate_or_register(email="researcher@example.com", password="wrong-password")

    token = store.create_login_session(account_id=first.account.id)
    assert store.get_account_for_token(token) == first.account
    store.revoke_login_session(token)
    assert store.get_account_for_token(token) is None


def test_conversation_history_and_metadata_are_scoped_to_the_owner(tmp_path) -> None:
    store = ConversationStore(f"sqlite:///{tmp_path / 'conversations.sqlite3'}")
    session = store.create_session(user_id="alice")
    store.append_message(
        user_id="alice",
        session_id=session["id"],
        role="user",
        content="Mn3GaN怎么做？",
    )
    store.append_message(
        user_id="alice",
        session_id=session["id"],
        role="assistant",
        content="可尝试方案。",
        response={"evidence_kind": "model_prediction", "citations": []},
    )

    assert store.list_sessions(user_id="alice")[0]["title"] == "Mn3GaN怎么做？"
    assert store.list_sessions(user_id="bob") == []
    assert store.get_session(user_id="bob", session_id=session["id"]) is None
    assert store.list_messages(user_id="bob", session_id=session["id"]) == []

    messages = store.list_messages(user_id="alice", session_id=session["id"])
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["response"]["evidence_kind"] == "model_prediction"


def test_auth_and_session_api_use_cookie_identity(monkeypatch, tmp_path) -> None:
    account_store = AccountStore(f"sqlite:///{tmp_path / 'accounts.sqlite3'}")
    conversation_store = ConversationStore(f"sqlite:///{tmp_path / 'conversations.sqlite3'}")
    monkeypatch.setattr(auth_api, "get_default_account_store", lambda: account_store)
    monkeypatch.setattr(account_dependencies, "get_default_account_store", lambda: account_store)
    monkeypatch.setattr(
        conversations_api, "get_default_conversation_store", lambda: conversation_store
    )

    with TestClient(app) as alice_client:
        created = alice_client.post(
            "/api/auth/login",
            json={"email": "alice@example.com", "password": "alice-password"},
        )
        assert created.status_code == 200
        assert created.json()["created"] is True
        assert alice_client.get("/api/auth/me").json()["email"] == "alice@example.com"

        session = alice_client.post("/api/rag/sessions")
        assert session.status_code == 201
        session_id = session.json()["id"]
        assert alice_client.get("/api/rag/sessions").json()[0]["id"] == session_id

        logout = alice_client.post("/api/auth/logout")
        assert logout.status_code == 204
        assert "agentweb_session=" in logout.headers["set-cookie"]
        assert alice_client.get("/api/auth/me").status_code == 401

    with TestClient(app) as bob_client:
        login = bob_client.post(
            "/api/auth/login",
            json={"email": "bob@example.com", "password": "bob-password"},
        )
        assert login.status_code == 200
        assert bob_client.get(f"/api/rag/sessions/{session_id}/messages").status_code == 404
