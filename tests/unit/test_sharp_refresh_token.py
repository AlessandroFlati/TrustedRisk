"""Unit tests for the SHARP refresh-token exchange utility.

Network is mocked via httpx.MockTransport -- no real HTTP traffic.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from mcp_server.sharp.refresh import (
    RefreshTokenExchangeError,
    RefreshedTokens,
    refresh_fhir_token,
    refresh_fhir_token_async,
)


def _stub_handler(payload: dict, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(payload).encode()
        return httpx.Response(status, content=body,
                              headers={"content-type": "application/json"})
    return handler


def _patch_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_post = httpx.post
    real_async_client = httpx.AsyncClient

    def fake_post(url, **kwargs):
        with httpx.Client(transport=transport) as c:
            return c.post(url, **kwargs)

    class _PatchedAsync(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, transport=transport, **kw)

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("httpx.AsyncClient", _PatchedAsync)


def test_refresh_token_round_trip(monkeypatch):
    _patch_transport(monkeypatch, _stub_handler({
        "accessToken": "new-access-tok",
        "refreshToken": "rotated-refresh-tok",
    }))
    out = refresh_fhir_token(
        refresh_url="https://idp.example.org/refresh",
        refresh_token="orig-refresh",
    )
    assert isinstance(out, RefreshedTokens)
    assert out.access_token == "new-access-tok"
    assert out.refresh_token == "rotated-refresh-tok"


def test_refresh_token_async_round_trip(monkeypatch):
    _patch_transport(monkeypatch, _stub_handler({
        "accessToken": "new-access-tok",
        "refreshToken": "rotated-refresh-tok",
    }))
    out = asyncio.run(refresh_fhir_token_async(
        refresh_url="https://idp.example.org/refresh",
        refresh_token="orig-refresh",
    ))
    assert out.access_token == "new-access-tok"
    assert out.refresh_token == "rotated-refresh-tok"


def test_refresh_token_missing_args_rejected():
    with pytest.raises(RefreshTokenExchangeError, match="required"):
        refresh_fhir_token("", "refresh")
    with pytest.raises(RefreshTokenExchangeError, match="required"):
        refresh_fhir_token("https://x", "")


def test_refresh_token_non_2xx_response(monkeypatch):
    _patch_transport(monkeypatch, _stub_handler({"error": "denied"}, status=401))
    with pytest.raises(RefreshTokenExchangeError, match="401"):
        refresh_fhir_token(
            refresh_url="https://idp.example.org/refresh",
            refresh_token="x",
        )


def test_refresh_token_missing_access_token_field(monkeypatch):
    _patch_transport(monkeypatch, _stub_handler({"refreshToken": "r"}))
    with pytest.raises(RefreshTokenExchangeError, match="accessToken"):
        refresh_fhir_token(
            refresh_url="https://idp.example.org/refresh",
            refresh_token="x",
        )


def test_refresh_token_missing_refresh_token_field(monkeypatch):
    _patch_transport(monkeypatch, _stub_handler({"accessToken": "a"}))
    with pytest.raises(RefreshTokenExchangeError, match="refreshToken"):
        refresh_fhir_token(
            refresh_url="https://idp.example.org/refresh",
            refresh_token="x",
        )


def test_refresh_token_payload_must_be_object(monkeypatch):
    def handler(_request):
        return httpx.Response(
            200, content=b'["not", "an", "object"]',
            headers={"content-type": "application/json"})
    _patch_transport(monkeypatch, handler)
    with pytest.raises(RefreshTokenExchangeError, match="not an object"):
        refresh_fhir_token(
            refresh_url="https://idp.example.org/refresh",
            refresh_token="x",
        )
