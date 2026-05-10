"""Phase 3.2 -- A2A AUTH_REQUIRED + auto-refresh tests.

Verifies that `with_auth_retry`:
  1. Passes calls through normally when no 401 fires
  2. On 401, attempts refresh via X-FHIR-Refresh-Token + X-FHIR-Refresh-Url
  3. After successful refresh, retries the call once with the new token
  4. On refresh failure (no token / refresh endpoint error /
     refreshed-token-also-401), raises FhirAuthRequired with a
     structured AuthRequiredHint
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from mcp_server.fhir.auth_retry import (
    FhirAuthRequired,
    with_auth_retry,
)
from mcp_server.sharp.headers import FHIRContext, _fhir_ctx


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Helpers ───────────────────────

class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        message=f"{status}",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(status),
    )


def _bind_ctx(**kw):
    """Bind a FHIRContext for the duration of one test."""
    defaults = dict(
        server_url="http://hapi/fhir",
        access_token="orig-tok",
        patient_id="Patient/1",
        refresh_token=None,
        refresh_token_url=None,
    )
    defaults.update(kw)
    return _fhir_ctx.set(FHIRContext(**defaults))


# ─────────────────────── Pass-through ───────────────────────

def test_with_auth_retry_passes_through_on_success():
    token = _bind_ctx()
    try:
        async def fake_call():
            return "ok"
        result = _run(with_auth_retry(fake_call))
        assert result == "ok"
    finally:
        _fhir_ctx.reset(token)


def test_with_auth_retry_propagates_non_401():
    token = _bind_ctx()
    try:
        async def fake_call():
            raise ValueError("not a 401")
        with pytest.raises(ValueError, match="not a 401"):
            _run(with_auth_retry(fake_call))
    finally:
        _fhir_ctx.reset(token)


# ─────────────────────── 401 + refresh ───────────────────────

def test_401_without_refresh_token_raises_auth_required():
    """No X-FHIR-Refresh-Token in context -> cannot recover, must
    promote to AUTH_REQUIRED."""
    token = _bind_ctx()
    try:
        async def fake_call():
            raise _http_status_error(401)
        with pytest.raises(FhirAuthRequired) as ei:
            _run(with_auth_retry(fake_call))
        assert ei.value.hint.refresh_attempted is False
        assert "refresh token" in ei.value.hint.message.lower()
    finally:
        _fhir_ctx.reset(token)


def test_401_with_refresh_token_succeeds_on_retry(monkeypatch):
    """Successful refresh + retry returns the wrapped call's result."""
    from mcp_server.sharp import refresh as refresh_mod

    async def fake_refresh(refresh_url, refresh_token, **kw):
        return refresh_mod.RefreshedTokens(
            access_token="new-access-tok",
            refresh_token="rotated-refresh-tok",
        )
    monkeypatch.setattr(
        "mcp_server.fhir.auth_retry.refresh_fhir_token_async", fake_refresh,
    )

    token = _bind_ctx(
        refresh_token="orig-refresh", refresh_token_url="http://idp/refresh",
    )
    try:
        attempt = {"count": 0}

        async def fake_call():
            attempt["count"] += 1
            if attempt["count"] == 1:
                raise _http_status_error(401)
            return "ok-after-refresh"

        result = _run(with_auth_retry(fake_call))
        assert result == "ok-after-refresh"
        assert attempt["count"] == 2
        # Context should now carry the new access token
        ctx = _fhir_ctx.get()
        assert ctx.access_token == "new-access-tok"
    finally:
        _fhir_ctx.reset(token)


def test_401_with_failed_refresh_raises_auth_required(monkeypatch):
    """Refresh endpoint returns error -> AuthRequiredHint with
    refresh_attempted=True and a reason."""
    from mcp_server.sharp.refresh import RefreshTokenExchangeError

    async def fake_refresh(refresh_url, refresh_token, **kw):
        raise RefreshTokenExchangeError("idp returned 500")
    monkeypatch.setattr(
        "mcp_server.fhir.auth_retry.refresh_fhir_token_async", fake_refresh,
    )

    token = _bind_ctx(
        refresh_token="orig-refresh", refresh_token_url="http://idp/refresh",
    )
    try:
        async def fake_call():
            raise _http_status_error(401)
        with pytest.raises(FhirAuthRequired) as ei:
            _run(with_auth_retry(fake_call))
        assert ei.value.hint.refresh_attempted is True
        assert "500" in (ei.value.hint.refresh_failure_reason or "")
    finally:
        _fhir_ctx.reset(token)


def test_refreshed_token_also_401_raises_auth_required(monkeypatch):
    """Even the freshly-refreshed token gets a 401 -> unrecoverable,
    promote to AUTH_REQUIRED."""
    from mcp_server.sharp import refresh as refresh_mod

    async def fake_refresh(refresh_url, refresh_token, **kw):
        return refresh_mod.RefreshedTokens(
            access_token="new-tok", refresh_token="rotated",
        )
    monkeypatch.setattr(
        "mcp_server.fhir.auth_retry.refresh_fhir_token_async", fake_refresh,
    )

    token = _bind_ctx(
        refresh_token="orig-refresh", refresh_token_url="http://idp/refresh",
    )
    try:
        async def fake_call():
            raise _http_status_error(401)
        with pytest.raises(FhirAuthRequired) as ei:
            _run(with_auth_retry(fake_call))
        assert ei.value.hint.refresh_attempted is True
        assert (
            "also" in (ei.value.hint.refresh_failure_reason or "").lower()
        )
    finally:
        _fhir_ctx.reset(token)


def test_401_outside_request_raises_auth_required():
    """No FHIRContext bound at all -> cannot recover."""
    async def fake_call():
        raise _http_status_error(401)
    with pytest.raises(FhirAuthRequired) as ei:
        _run(with_auth_retry(fake_call))
    assert "no SHARP" in ei.value.hint.message or \
           "no FHIR" in ei.value.hint.message.lower()
