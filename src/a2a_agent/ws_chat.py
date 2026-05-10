"""Phase 13.12 G1 -- WebSocket A2A streaming + multi-turn chat.

Adds a `/ws/chat` endpoint that maintains per-session state across
turns, routes user messages through the planner brain, and streams
back step-by-step responses.

Wire format (JSON messages):

    Client -> Server:
        {"type": "user_message", "session_id": "...", "text": "..."}
        {"type": "reset", "session_id": "..."}

    Server -> Client:
        {"type": "plan", "session_id": "...", "steps": [...]}
        {"type": "step_summary", "session_id": "...",
         "tool": "...", "summary": "..."}
        {"type": "final", "session_id": "...", "rationale": "..."}
        {"type": "error", "session_id": "...", "reason": "..."}

The planner is the same `a2a_agent.planner.plan_tool_use` used by
the deterministic-floor router elsewhere; per-session memory is held
in an in-process `_SESSIONS` dict keyed by session_id.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect


# ─────────────────────────────────────────────────────────────────────
# Per-session memory
# ─────────────────────────────────────────────────────────────────────


_SESSIONS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_SESSION_MAX = 256
_SESSION_TURN_HISTORY_MAX = 20


def _get_session(session_id: str) -> dict[str, Any]:
    if session_id not in _SESSIONS:
        if len(_SESSIONS) >= _SESSION_MAX:
            _SESSIONS.popitem(last=False)
        _SESSIONS[session_id] = {
            "created_at": time.time(),
            "turns": [],
            "tools_invoked": [],
        }
    _SESSIONS.move_to_end(session_id)
    return _SESSIONS[session_id]


def reset_session(session_id: str) -> None:
    """Test-helper + client-callable: drop a session from memory."""
    _SESSIONS.pop(session_id, None)


def get_session_state(session_id: str) -> dict[str, Any] | None:
    """Test-helper: snapshot the current session memory."""
    if session_id in _SESSIONS:
        return dict(_SESSIONS[session_id])
    return None


# ─────────────────────────────────────────────────────────────────────
# Per-message handler
# ─────────────────────────────────────────────────────────────────────


async def handle_user_message(
    session_id: str, text: str,
) -> list[dict[str, Any]]:
    """Process one user turn. Returns the list of response messages
    that the WebSocket should send (in order).

    Pure-async, no WebSocket dependency -- so unit tests can drive the
    handler directly without a TestClient.WebSocketTestSession."""
    from a2a_agent.planner import plan_tool_use

    session = _get_session(session_id)
    plan = await plan_tool_use(text)

    # Truncate turn history at the soft cap
    if len(session["turns"]) >= _SESSION_TURN_HISTORY_MAX:
        session["turns"] = session["turns"][-_SESSION_TURN_HISTORY_MAX + 1:]
    session["turns"].append({
        "user_text": text,
        "n_steps": plan.n_steps,
        "ts": time.time(),
    })

    out: list[dict[str, Any]] = []
    out.append({
        "type": "plan",
        "session_id": session_id,
        "n_steps": plan.n_steps,
        "safety_floor_engaged": plan.safety_floor_engaged,
        "steps": [
            {
                "step_id": s.step_id, "specialist": s.specialist,
                "bundle": s.bundle, "tool": s.tool,
                "mandatory": s.mandatory,
                "rationale": s.rationale[:200],
            }
            for s in plan.steps
        ],
    })

    for s in plan.steps:
        session["tools_invoked"].append(s.tool)
        out.append({
            "type": "step_summary",
            "session_id": session_id,
            "step_id": s.step_id,
            "tool": s.tool,
            "specialist": s.specialist,
            "summary": s.rationale[:200],
        })

    out.append({
        "type": "final",
        "session_id": session_id,
        "rationale": plan.rationale,
        "confidence": plan.confidence,
        "n_tools_invoked_this_session": len(session["tools_invoked"]),
        "n_turns_in_session": len(session["turns"]),
    })
    return out


# ─────────────────────────────────────────────────────────────────────
# Starlette WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────


async def ws_chat_endpoint(websocket: WebSocket) -> None:
    """`/ws/chat` -- JSON-over-WebSocket multi-turn chat. Each user
    message produces a stream of response messages culminating in a
    `final` envelope."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "reason": "malformed_json",
                })
                continue

            mtype = msg.get("type")
            session_id = str(msg.get("session_id") or "")
            if not session_id:
                await websocket.send_json({
                    "type": "error",
                    "reason": "missing_session_id",
                })
                continue

            if mtype == "reset":
                reset_session(session_id)
                await websocket.send_json({
                    "type": "reset_ok", "session_id": session_id,
                })
                continue

            if mtype != "user_message":
                await websocket.send_json({
                    "type": "error", "session_id": session_id,
                    "reason": f"unknown_type:{mtype}",
                })
                continue

            text = str(msg.get("text") or "")
            if not text:
                await websocket.send_json({
                    "type": "error", "session_id": session_id,
                    "reason": "empty_text",
                })
                continue

            try:
                responses = await handle_user_message(session_id, text)
            except Exception as exc:    # noqa: BLE001
                await websocket.send_json({
                    "type": "error", "session_id": session_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue

            for r in responses:
                await websocket.send_json(r)
    except WebSocketDisconnect:
        return
