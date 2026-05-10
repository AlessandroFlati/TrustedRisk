"""Phase 13.12 G1 -- WebSocket A2A streaming + multi-turn chat tests."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from a2a_agent.ws_chat import (
    get_session_state, handle_user_message, reset_session,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# In-process handler smoke (no WebSocket)
# ─────────────────────────────────────────────────────────────────────

def test_handle_user_message_emits_plan_step_summaries_and_final():
    reset_session("smoke-1")
    out = _run(handle_user_message(
        "smoke-1", "Is this patient safe to discharge home?",
    ))
    types = [m["type"] for m in out]
    assert types[0] == "plan"
    assert types[-1] == "final"
    # Each step gets its own step_summary message
    n_steps = out[0]["n_steps"]
    n_step_summaries = sum(1 for t in types if t == "step_summary")
    assert n_step_summaries == n_steps


def test_session_carries_turn_history_across_messages():
    reset_session("multi-1")
    _run(handle_user_message("multi-1", "Discharge planning."))
    _run(handle_user_message("multi-1", "Translate to Spanish."))
    state = get_session_state("multi-1")
    assert state is not None
    assert len(state["turns"]) == 2
    # Tools accumulate across turns
    assert len(state["tools_invoked"]) >= 2


def test_reset_session_clears_state():
    reset_session("reset-1")
    _run(handle_user_message("reset-1", "Hello"))
    assert get_session_state("reset-1") is not None
    reset_session("reset-1")
    assert get_session_state("reset-1") is None


# ─────────────────────────────────────────────────────────────────────
# WebSocket end-to-end
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def specialist_app():
    from apps.specialist_quality.server import app
    return app


def test_ws_chat_round_trip(specialist_app):
    client = TestClient(specialist_app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "user_message",
            "session_id": "ws-1",
            "text": "HEDIS Stars rating forecast for this MA contract.",
        }))
        # Expect: plan + N×step_summary + final
        first = ws.receive_json()
        assert first["type"] == "plan"
        assert first["session_id"] == "ws-1"
        n_steps = first["n_steps"]
        for _ in range(n_steps):
            msg = ws.receive_json()
            assert msg["type"] == "step_summary"
        final = ws.receive_json()
        assert final["type"] == "final"
        assert "rationale" in final


def test_ws_chat_handles_reset(specialist_app):
    client = TestClient(specialist_app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "user_message", "session_id": "ws-reset",
            "text": "Discharge planning.",
        }))
        # Drain the stream
        msg = None
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "final":
                break
        assert msg["type"] == "final"

        ws.send_text(json.dumps({
            "type": "reset", "session_id": "ws-reset",
        }))
        ack = ws.receive_json()
        assert ack["type"] == "reset_ok"
        assert ack["session_id"] == "ws-reset"


def test_ws_chat_rejects_malformed_json(specialist_app):
    client = TestClient(specialist_app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text("not json at all")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["reason"] == "malformed_json"


def test_ws_chat_rejects_missing_session_id(specialist_app):
    client = TestClient(specialist_app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "user_message", "text": "hi",
        }))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["reason"] == "missing_session_id"


def test_ws_chat_rejects_unknown_type(specialist_app):
    client = TestClient(specialist_app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "weird", "session_id": "ws-x",
        }))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["reason"].startswith("unknown_type")


def test_ws_chat_rejects_empty_text(specialist_app):
    client = TestClient(specialist_app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "user_message", "session_id": "ws-empty", "text": "",
        }))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["reason"] == "empty_text"


# ─────────────────────────────────────────────────────────────────────
# Session memory bound
# ─────────────────────────────────────────────────────────────────────

def test_turn_history_truncates_at_soft_cap():
    reset_session("trunc-1")
    for i in range(25):
        _run(handle_user_message(
            "trunc-1", f"Discharge planning iteration {i}",
        ))
    state = get_session_state("trunc-1")
    assert state is not None
    # Soft cap is 20
    assert len(state["turns"]) <= 20
