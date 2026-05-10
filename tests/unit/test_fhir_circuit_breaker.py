"""Phase 8.4 -- FHIR circuit breaker + retry + pagination tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from mcp_server.fhir.circuit_breaker import (
    CircuitOpen,
    default_breaker_for,
    paginate_bundle,
    reset_all,
    with_circuit_breaker,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean():
    reset_all()
    yield
    reset_all()


def _http_status_error(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        message=f"{code}",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(code),
    )


# ─────────────────────── Pass-through ───────────────────────


def test_passthrough_on_success():
    async def ok():
        return "ok"
    assert _run(with_circuit_breaker("http://x", ok)) == "ok"


def test_breaker_resets_failures_on_success(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_FHIR_FAILURE_THRESHOLD", "3")
    reset_all()
    breaker = default_breaker_for("http://x")
    breaker.failures = [1.0, 2.0]   # 2 prior failures
    assert breaker.check_state() == "closed"

    async def ok():
        return "ok"
    _run(with_circuit_breaker("http://x", ok))
    assert breaker.failures == []
    assert breaker.state == "closed"


# ─────────────────────── Retry transient ───────────────────────


def test_retries_on_5xx_then_succeeds(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_FHIR_MAX_RETRY", "3")
    n = {"attempts": 0}

    async def flaky():
        n["attempts"] += 1
        if n["attempts"] < 2:
            raise _http_status_error(503)
        return "ok"

    out = _run(with_circuit_breaker("http://x", flaky))
    assert out == "ok"
    assert n["attempts"] == 2


def test_does_not_retry_on_4xx_auth(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_FHIR_MAX_RETRY", "3")
    n = {"attempts": 0}

    async def auth_fail():
        n["attempts"] += 1
        raise _http_status_error(401)

    with pytest.raises(httpx.HTTPStatusError):
        _run(with_circuit_breaker("http://x", auth_fail))
    assert n["attempts"] == 1


# ─────────────────────── Open circuit ───────────────────────


def test_breaker_opens_after_n_failures(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_FHIR_FAILURE_THRESHOLD", "3")
    monkeypatch.setenv("TRUSTEDRISK_FHIR_MAX_RETRY", "1")
    reset_all()

    async def fail():
        raise _http_status_error(503)

    # 3 failed calls (each only 1 attempt due to MAX_RETRY=1)
    for _ in range(3):
        with pytest.raises((CircuitOpen, httpx.HTTPStatusError)):
            _run(with_circuit_breaker("http://x", fail))

    breaker = default_breaker_for("http://x")
    assert breaker.state == "open"

    # Subsequent call fails fast with CircuitOpen
    async def ok():
        raise AssertionError("must not be called when breaker is open")
    with pytest.raises(CircuitOpen):
        _run(with_circuit_breaker("http://x", ok))


def test_breaker_per_server_isolation(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_FHIR_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("TRUSTEDRISK_FHIR_MAX_RETRY", "1")
    reset_all()

    async def fail():
        raise _http_status_error(502)

    for _ in range(2):
        with pytest.raises((CircuitOpen, httpx.HTTPStatusError)):
            _run(with_circuit_breaker("http://server-a", fail))

    # Server A is open; server B should be unaffected
    async def ok():
        return "ok"
    assert _run(with_circuit_breaker("http://server-b", ok)) == "ok"


# ─────────────────────── Half-open recovery ───────────────────────


def test_half_open_after_cooldown_then_recover(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_FHIR_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("TRUSTEDRISK_FHIR_MAX_RETRY", "1")
    monkeypatch.setenv("TRUSTEDRISK_FHIR_COOLDOWN_S", "0.05")
    reset_all()

    async def fail():
        raise _http_status_error(503)

    for _ in range(2):
        with pytest.raises((CircuitOpen, httpx.HTTPStatusError)):
            _run(with_circuit_breaker("http://srv", fail))

    breaker = default_breaker_for("http://srv")
    assert breaker.state == "open"

    # Wait for cooldown
    import time
    time.sleep(0.08)
    # Half-open transition triggered by check_state()
    assert breaker.check_state() == "half_open"

    async def ok():
        return "ok"
    assert _run(with_circuit_breaker("http://srv", ok)) == "ok"
    assert default_breaker_for("http://srv").state == "closed"


# ─────────────────────── Bundle pagination ───────────────────────


def test_paginate_walks_link_chain():
    pages: dict[str | None, dict] = {
        None: {
            "resourceType": "Bundle",
            "entry": [{"resource": {"id": "p1-r1"}},
                          {"resource": {"id": "p1-r2"}}],
            "link": [{"relation": "next", "url": "page2"}],
        },
        "page2": {
            "resourceType": "Bundle",
            "entry": [{"resource": {"id": "p2-r1"}}],
            "link": [{"relation": "next", "url": "page3"}],
        },
        "page3": {
            "resourceType": "Bundle",
            "entry": [{"resource": {"id": "p3-r1"}}],
            "link": [],
        },
    }

    async def fetch(url=None):
        return pages[url]

    out = _run(paginate_bundle(fetch))
    ids = [e["resource"]["id"] for e in out["entry"]]
    assert ids == ["p1-r1", "p1-r2", "p2-r1", "p3-r1"]
    assert out["_n_pages_walked"] == 3


def test_paginate_caps_at_max_pages():
    pages: dict[str | None, dict] = {None: {
        "resourceType": "Bundle", "entry": [],
        "link": [{"relation": "next", "url": "loop"}],
    }, "loop": {
        "resourceType": "Bundle", "entry": [],
        "link": [{"relation": "next", "url": "loop"}],
    }}

    async def fetch(url=None):
        return pages.get(url, pages[None])

    out = _run(paginate_bundle(fetch, max_pages=3))
    assert out["_n_pages_walked"] == 3


def test_paginate_handles_no_link():
    bundle = {"resourceType": "Bundle",
                  "entry": [{"resource": {"id": "only"}}]}

    async def fetch(url=None):
        return bundle

    out = _run(paginate_bundle(fetch))
    assert out["_n_pages_walked"] == 1
    assert len(out["entry"]) == 1
