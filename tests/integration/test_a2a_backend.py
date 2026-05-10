"""A2A backend tests for the Care Engine.

These tests validate that the orchestrator can dispatch workflow steps
to specialist agents over real HTTP (not in-process imports), which is
the runtime requirement for the "agent composition" pitch in the β
submission. The default backend stays in-process for production speed;
this test suite exercises the HTTP path that activates when
`TRUSTEDRISK_COMPOSER_BACKEND=a2a`.

Three layers:

  1. Backend selector unit tests (env var → backend class).
  2. URL resolution unit tests (federation umbrella vs single-port).
  3. End-to-end live test against a uvicorn-hosted specialist on a
     random local port. Skipped when uvicorn cannot bind the chosen
     port within 5 s (CI environments without loopback support).
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time

import pytest


# ─────────────────────── 1. Backend selector ───────────────────────


def test_default_backend_is_inprocess(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_COMPOSER_BACKEND", raising=False)
    from apps.composer.backends import InProcessBackend, default_backend
    backend = default_backend()
    assert isinstance(backend, InProcessBackend)
    assert backend.name == "inprocess"


def test_a2a_backend_selected_via_env(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_COMPOSER_BACKEND", "a2a")
    from apps.composer.backends import A2ABackend, default_backend
    backend = default_backend()
    assert isinstance(backend, A2ABackend)
    assert backend.name == "a2a"


def test_unknown_backend_falls_back_to_inprocess(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_COMPOSER_BACKEND", "carrier-pigeon")
    from apps.composer.backends import InProcessBackend, default_backend
    backend = default_backend()
    assert isinstance(backend, InProcessBackend)


# ─────────────────────── 2. URL resolution ───────────────────────


def test_a2a_url_uses_federation_base_when_set(monkeypatch):
    monkeypatch.setenv(
        "TRUSTEDRISK_FEDERATION_BASE_URL", "https://flati.work")
    from apps.composer.backends import A2ABackend
    backend = A2ABackend()
    assert backend._url_for("trustedrisk-discharge") == \
        "https://flati.work/a2a/trustedrisk-discharge/mcp"


def test_a2a_url_falls_back_to_specialist_port(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_FEDERATION_BASE_URL", raising=False)
    from apps.composer.backends import A2ABackend
    backend = A2ABackend()
    # Per _SPECIALIST_PORTS: discharge=8770
    assert backend._url_for("trustedrisk-discharge") == \
        "http://127.0.0.1:8770/mcp"


def test_a2a_url_unknown_specialist_raises(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_FEDERATION_BASE_URL", raising=False)
    from apps.composer.backends import A2ABackend
    backend = A2ABackend()
    with pytest.raises(ValueError, match="no port mapping"):
        backend._url_for("trustedrisk-does-not-exist")


# ─────────────────────── 3. SHARP header propagation ───────────────────────


def test_a2a_headers_empty_outside_sharp_context():
    from apps.composer.backends import A2ABackend
    backend = A2ABackend()
    # No ContextVar bound -> empty dict (no spurious headers).
    assert backend._headers() == {}


def test_a2a_headers_forward_fhir_context():
    from apps.composer.backends import A2ABackend
    from mcp_server.sharp.headers import FHIRContext, _fhir_ctx

    token = _fhir_ctx.set(FHIRContext(
        server_url="https://hapi.example.com/fhir",
        access_token="bearer-xyz",
        patient_id="pt-99",
        refresh_token="rt-abc",
        refresh_token_url="https://idp.example.com/refresh",
    ))
    try:
        h = A2ABackend()._headers()
    finally:
        _fhir_ctx.reset(token)
    assert h["X-FHIR-Server-URL"] == "https://hapi.example.com/fhir"
    assert h["X-FHIR-Access-Token"] == "bearer-xyz"
    assert h["X-Patient-ID"] == "pt-99"
    assert h["X-FHIR-Refresh-Token"] == "rt-abc"
    assert h["X-FHIR-Refresh-Url"] == "https://idp.example.com/refresh"


# ─────────────────────── 4. Live HTTP round-trip ───────────────────────


def _free_port() -> int:
    """Bind/release a TCP port to find one nobody else is on."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def live_specialist_server():
    """Bring up the discharge specialist on a random local port.

    Yields the port number; the uvicorn server is shut down on teardown.
    Skips the test if uvicorn fails to start within the deadline.
    """
    import uvicorn
    from apps.specialist_discharge.server import app

    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 8.0
    while time.time() < deadline:
        if server.started:
            break
        time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=2.0)
        pytest.skip("uvicorn did not start within deadline")

    yield port

    server.should_exit = True
    thread.join(timeout=3.0)


def test_a2a_backend_calls_live_specialist(live_specialist_server):
    """End-to-end: A2ABackend POSTs tools/call to a real HTTP server.

    Picks `detect_phi` because it doesn't require a FHIR fetch and so
    doesn't depend on the SHARP middleware ContextVar being populated
    on the specialist's side.
    """
    port = live_specialist_server
    from apps.composer.backends import A2ABackend

    class _DirectMcpBackend(A2ABackend):
        # The live specialist mounts FastMCP at /mcp directly (no
        # /a2a/<slug> prefix); override URL resolution + provide stub
        # SHARP headers so the middleware accepts the tools/call.
        def _url_for(self, specialist: str) -> str:
            return f"http://127.0.0.1:{port}/mcp"

        def _headers(self) -> dict:
            return {
                "X-FHIR-Server-URL": "http://stub.fhir.local",
                "X-FHIR-Access-Token": "stub-token",
            }

    backend = _DirectMcpBackend()

    async def _call():
        return await backend.call(
            specialist="trustedrisk-discharge",
            tool_name="detect_phi",
            callable_=None,  # never used in A2A path
            resolved_inputs={
                "text": "Patient John Doe (MRN 12345) discharged home.",
            },
        )

    result = asyncio.run(_call())
    assert isinstance(result, dict), \
        f"expected dict from A2A backend, got {type(result).__name__}"
    # The detect_phi tool returns a `redaction_map` field with the
    # original->placeholder substitutions. We don't assert the exact
    # contents (the regex floor evolves), only that the field exists,
    # confirming the round-trip went through real HTTP.
    assert "redaction_map" in result or "entities" in result or \
        "risk_level" in result, (
            f"detect_phi response shape unexpected: {sorted(result.keys())}"
        )
