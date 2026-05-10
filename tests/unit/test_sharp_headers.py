"""Unit tests for SHARP-on-MCP context middleware."""
from __future__ import annotations

import asyncio
import json

import pytest

from mcp_server.sharp.headers import (
    FHIRContext,
    _fhir_ctx,
    get_fhir_context,
    initialize_capabilities,
    sharp_context_middleware,
)


class _StubRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── FHIRContext dataclass ───────────────────────

def test_fhir_context_constructs():
    ctx = FHIRContext(server_url="http://hapi/fhir", access_token="tok", patient_id="pt-1")
    assert ctx.server_url == "http://hapi/fhir"
    assert ctx.patient_id == "pt-1"


def test_fhir_context_no_patient_id():
    ctx = FHIRContext(server_url="http://hapi/fhir", access_token="tok")
    assert ctx.patient_id is None


def test_fhir_context_offline_access_fields_default_none():
    ctx = FHIRContext(server_url="http://hapi/fhir", access_token="tok")
    assert ctx.refresh_token is None
    assert ctx.refresh_token_url is None


def test_fhir_context_offline_access_fields_populate():
    ctx = FHIRContext(
        server_url="http://hapi/fhir",
        access_token="tok",
        refresh_token="r-tok",
        refresh_token_url="https://idp/refresh",
    )
    assert ctx.refresh_token == "r-tok"
    assert ctx.refresh_token_url == "https://idp/refresh"


# ─────────────────────── get_fhir_context ───────────────────────

def test_get_fhir_context_outside_request_raises():
    with pytest.raises(LookupError):
        get_fhir_context()


def test_get_fhir_context_inside_request():
    token = _fhir_ctx.set(FHIRContext(server_url="u", access_token="t"))
    try:
        ctx = get_fhir_context()
        assert ctx.server_url == "u"
    finally:
        _fhir_ctx.reset(token)


# ─────────────────────── sharp_context_middleware ───────────────────────

def test_middleware_blocks_missing_server_url():
    req = _StubRequest({"X-FHIR-Access-Token": "tok"})

    async def call_next(_):
        raise AssertionError("call_next must NOT be invoked when context is missing")

    response = _run(sharp_context_middleware(req, call_next))
    assert response.status_code == 403
    body = json.loads(response.body)
    assert "X-FHIR-Server-URL" in body["missing_headers"]
    assert body["error"] == "missing_fhir_context"


def test_middleware_blocks_missing_access_token():
    req = _StubRequest({"X-FHIR-Server-URL": "http://hapi"})

    async def call_next(_):
        raise AssertionError("call_next must NOT be invoked")

    response = _run(sharp_context_middleware(req, call_next))
    assert response.status_code == 403
    body = json.loads(response.body)
    assert "X-FHIR-Access-Token" in body["missing_headers"]


def test_middleware_passes_with_valid_headers():
    req = _StubRequest({
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
        "X-Patient-ID": "pt-9",
    })

    async def call_next(r):
        # Inside the call_next, the context should be set
        ctx = get_fhir_context()
        assert ctx.server_url == "http://hapi/fhir"
        assert ctx.access_token == "tok"
        assert ctx.patient_id == "pt-9"
        return "ok"

    result = _run(sharp_context_middleware(req, call_next))
    assert result == "ok"


def test_middleware_resets_context_after_request():
    req = _StubRequest({
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
    })

    async def call_next(r):
        return "ok"

    _run(sharp_context_middleware(req, call_next))
    # After the middleware returns, get_fhir_context() should raise LookupError
    with pytest.raises(LookupError):
        get_fhir_context()


def test_middleware_captures_offline_access_headers():
    """When the client supplies the offline-access headers, the
    FHIRContext should carry them through."""
    req = _StubRequest({
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
        "X-FHIR-Refresh-Token": "refresh-xyz",
        "X-FHIR-Refresh-Url": "https://idp/refresh",
    })

    captured: dict = {}

    async def call_next(_):
        ctx = get_fhir_context()
        captured["refresh_token"] = ctx.refresh_token
        captured["refresh_token_url"] = ctx.refresh_token_url
        return "ok"

    _run(sharp_context_middleware(req, call_next))
    assert captured["refresh_token"] == "refresh-xyz"
    assert captured["refresh_token_url"] == "https://idp/refresh"


def test_middleware_partial_offline_access_does_not_403():
    """If only one of the two offline-access headers is present, the
    middleware should still pass (offline-access is opt-in; the server
    simply won't be able to refresh)."""
    req = _StubRequest({
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
        "X-FHIR-Refresh-Token": "refresh-xyz",
        # X-FHIR-Refresh-Url omitted
    })

    async def call_next(_):
        return "ok"

    result = _run(sharp_context_middleware(req, call_next))
    assert result == "ok"


def test_middleware_patient_id_optional():
    """X-Patient-ID is optional per spec -- middleware should NOT reject when only it is missing."""
    req = _StubRequest({
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
        # X-Patient-ID intentionally omitted
    })

    async def call_next(r):
        ctx = get_fhir_context()
        assert ctx.patient_id is None
        return "ok"

    assert _run(sharp_context_middleware(req, call_next)) == "ok"


# ─────────────────────── initialize_capabilities ───────────────────────

def test_initialize_capabilities_declares_fhir_context():
    caps = initialize_capabilities()
    assert "experimental" in caps
    assert "fhir_context_required" in caps["experimental"]
    decl = caps["experimental"]["fhir_context_required"]
    assert decl["value"] is True
    assert "X-FHIR-Server-URL" in decl["headers"]
    assert "X-FHIR-Access-Token" in decl["headers"]
    assert "X-Patient-ID" in decl["headers"]
    assert decl["patient_id_required"] is False
