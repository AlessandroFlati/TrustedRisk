"""INT-2 -- SMART on FHIR launch context handshake.

Implements the EHR-launch flow per the SMART App Launch IG (v2):
  1. POST /smart/launch  -> records (iss, launch, client_id, scope, state)
                             and returns the URL the EHR would redirect the
                             user to (the EHR's authorize endpoint).
  2. POST /smart/callback -> exchanges the authorization code at the EHR's
                              token endpoint, stores the resolved SMART
                              context (server_url, access_token, scope,
                              patient_id, expires_at), returns a session id.
  3. GET  /smart/session/{session_id} -> returns the resolved context the
                                            agent uses to populate SHARP
                                            headers (X-FHIR-Server-URL,
                                            X-FHIR-Access-Token, X-Patient-ID).

This module deliberately avoids any browser-side concerns (PKCE flow,
JS App Launch HTML page) -- it focuses on the server-side handshake the
agent needs to complete in order to talk to a real EHR.

Token exchange goes through `_token_exchange()` which is overridable in
tests (no real HTTP calls).
"""

from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


# ─────────────────────── In-process state ───────────────────────

# {session_id: {iss, client_id, scope, state, code_verifier, ...}}
_PENDING: dict[str, dict[str, Any]] = {}

# {session_id: {server_url, access_token, scope, patient_id, expires_at}}
_RESOLVED: dict[str, dict[str, Any]] = {}


def _reset_state() -> None:
    """Test helper -- clear all in-process state."""
    _PENDING.clear()
    _RESOLVED.clear()


# ─────────────────────── Discovery (.well-known/smart-configuration) ───────────────────────

async def _fetch_smart_config(iss: str) -> dict[str, Any]:
    """Fetch the .well-known/smart-configuration document from the EHR.

    Per SMART v2 the document carries `authorization_endpoint` and
    `token_endpoint`. Fall-back to the FHIR CapabilityStatement is also
    valid but not implemented here.

    Tests monkeypatch this function.
    """
    url = iss.rstrip("/") + "/.well-known/smart-configuration"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()


# ─────────────────────── Token exchange ───────────────────────

async def _token_exchange(token_endpoint: str, *,
                              code: str, redirect_uri: str,
                              client_id: str,
                              code_verifier: str | None = None,
                              ) -> dict[str, Any]:
    """POST authorization code to the EHR's token endpoint.

    Returns the parsed token response (must include access_token).
    Tests override via monkeypatch.
    """
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
    }
    if code_verifier is not None:
        body["code_verifier"] = code_verifier
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            token_endpoint, data=body,
            headers={"Accept": "application/json",
                       "Content-Type": "application/x-www-form-urlencoded"})
        r.raise_for_status()
        return r.json()


# ─────────────────────── Routes ───────────────────────

async def smart_launch(request: Request) -> JSONResponse:
    body = await request.json() if (
        request.headers.get("content-length")
        and int(request.headers["content-length"]) > 0) else {}
    iss = body.get("iss")
    launch = body.get("launch")
    client_id = body.get("client_id") or os.environ.get(
        "TRUSTEDRISK_SMART_CLIENT_ID", "trustedrisk-app")
    scope = body.get("scope", "launch patient/*.read openid fhirUser")
    redirect_uri = body.get("redirect_uri") or \
        f"http://localhost:8765/smart/callback"

    if not iss:
        return JSONResponse(status_code=400,
                              content={"error": "missing_iss"})
    if not launch:
        return JSONResponse(status_code=400,
                              content={"error": "missing_launch"})

    try:
        smart_cfg = await _fetch_smart_config(iss)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=502, content={
            "error": "smart_config_fetch_failed",
            "detail": str(exc),
        })

    auth_ep = smart_cfg.get("authorization_endpoint")
    token_ep = smart_cfg.get("token_endpoint")
    if not auth_ep or not token_ep:
        return JSONResponse(status_code=502, content={
            "error": "smart_config_missing_endpoints",
            "smart_configuration": smart_cfg,
        })

    state = secrets.token_urlsafe(16)
    session_id = secrets.token_urlsafe(16)
    code_verifier = body.get("code_verifier")  # caller-supplied PKCE

    _PENDING[session_id] = {
        "iss": iss,
        "launch": launch,
        "client_id": client_id,
        "scope": scope,
        "state": state,
        "redirect_uri": redirect_uri,
        "token_endpoint": token_ep,
        "authorization_endpoint": auth_ep,
        "code_verifier": code_verifier,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    qs = urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "aud": iss,
        "launch": launch,
    })
    authorize_url = f"{auth_ep}?{qs}"

    return JSONResponse(content={
        "session_id": session_id,
        "state": state,
        "authorize_url": authorize_url,
        "iss": iss,
        "scope": scope,
    })


async def smart_callback(request: Request) -> JSONResponse:
    if request.method == "GET":
        params = dict(request.query_params)
    else:
        params = await request.json() if (
            request.headers.get("content-length")
            and int(request.headers["content-length"]) > 0) else {}

    code = params.get("code")
    state = params.get("state")
    if not code or not state:
        return JSONResponse(status_code=400, content={
            "error": "missing_code_or_state"})

    pending = next((p for p in _PENDING.values()
                       if p["state"] == state), None)
    session_id = next((sid for sid, p in _PENDING.items()
                          if p["state"] == state), None)
    if pending is None or session_id is None:
        return JSONResponse(status_code=400,
                              content={"error": "unknown_state"})

    try:
        token = await _token_exchange(
            pending["token_endpoint"],
            code=code,
            redirect_uri=pending["redirect_uri"],
            client_id=pending["client_id"],
            code_verifier=pending.get("code_verifier"),
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=502, content={
            "error": "token_exchange_failed",
            "detail": str(exc),
        })

    if "access_token" not in token:
        return JSONResponse(status_code=502, content={
            "error": "token_response_missing_access_token",
            "token_response": token,
        })

    expires_in = int(token.get("expires_in", 3600))
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    # Resolve patient_id from the standard `patient` claim or `fhirContext`
    patient_id = token.get("patient")
    if not patient_id:
        for ctx in (token.get("fhirContext") or []):
            ref = (ctx.get("reference") or "")
            if ref.startswith("Patient/"):
                patient_id = ref.split("/", 1)[1]
                break

    _RESOLVED[session_id] = {
        "server_url": pending["iss"],
        "access_token": token["access_token"],
        "scope": token.get("scope", pending["scope"]),
        "patient_id": patient_id,
        "expires_at": expires_at.isoformat(),
        "token_type": token.get("token_type", "Bearer"),
    }
    _PENDING.pop(session_id, None)

    return JSONResponse(content={
        "session_id": session_id,
        "server_url": pending["iss"],
        "patient_id": patient_id,
        "scope": token.get("scope", pending["scope"]),
        "expires_at": expires_at.isoformat(),
    })


async def smart_session(request: Request) -> JSONResponse:
    session_id = request.path_params.get("session_id")
    if not session_id or session_id not in _RESOLVED:
        return JSONResponse(status_code=404,
                              content={"error": "unknown_session"})
    sess = _RESOLVED[session_id]
    # NEVER return the raw access_token -- only metadata + redacted prefix
    token = sess["access_token"]
    redacted = (token[:6] + "..." + token[-4:]) if len(token) > 12 else "***"
    return JSONResponse(content={
        "session_id": session_id,
        "server_url": sess["server_url"],
        "scope": sess["scope"],
        "patient_id": sess["patient_id"],
        "expires_at": sess["expires_at"],
        "token_redacted": redacted,
        "token_type": sess["token_type"],
    })


def smart_routes() -> list[Route]:
    """Return the Starlette Route objects to mount on the MCP HTTP app."""
    return [
        Route("/smart/launch", smart_launch, methods=["POST"]),
        Route("/smart/callback", smart_callback, methods=["GET", "POST"]),
        Route("/smart/session/{session_id}", smart_session,
              methods=["GET"]),
    ]
