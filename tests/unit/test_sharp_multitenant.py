"""Multi-tenant SHARP-on-MCP isolation tests.

The SHARP middleware uses a `ContextVar` to bind the (server_url,
access_token, patient_id) triple to the in-flight request. ContextVar is
asyncio-aware: each task sees its own snapshot, even if multiple tasks run
concurrently in the same event loop. These tests verify that property
end-to-end against the actual middleware code, so a regression that
accidentally introduced shared module state would be caught.

Why it matters: a clinical decision-support service that's deployed for
multiple hospitals (different FHIR endpoints, different bearer tokens) MUST
guarantee that tenant A's context cannot be observed by tenant B. The
ContextVar pattern enforces this at the language level, but only if the
middleware doesn't accidentally cache the request context anywhere.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import pytest

from mcp_server.sharp.headers import (
    FHIRContext,
    get_fhir_context,
    sharp_context_middleware,
)


class _StubRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def _make_request(tenant: str) -> _StubRequest:
    return _StubRequest({
        "X-FHIR-Server-URL": f"https://hapi.{tenant}.example.com/fhir",
        "X-FHIR-Access-Token": f"tok-{tenant}",
        "X-Patient-ID": f"{tenant}-patient-1",
    })


# ─────────────────────── Sequential isolation ───────────────────────

def test_sequential_two_tenants_no_leak():
    """Tenant A's context must not leak into tenant B's request."""
    seen: list[FHIRContext] = []

    async def echo_handler(_):
        seen.append(get_fhir_context())
        return "ok"

    async def run_both():
        await sharp_context_middleware(_make_request("alpha"), echo_handler)
        # After A completes, the ContextVar is reset
        with pytest.raises(LookupError):
            get_fhir_context()
        await sharp_context_middleware(_make_request("beta"), echo_handler)

    asyncio.run(run_both())
    assert len(seen) == 2
    assert seen[0].server_url == "https://hapi.alpha.example.com/fhir"
    assert seen[1].server_url == "https://hapi.beta.example.com/fhir"
    assert seen[0].access_token != seen[1].access_token


# ─────────────────────── Concurrent isolation ───────────────────────

@pytest.mark.parametrize("n_concurrent", [10, 50])
def test_concurrent_tenants_no_leak(n_concurrent: int):
    """Run N concurrent requests with different tenants. Each task must see
    only its own context, never another tenant's."""
    observed: list[tuple[str, str]] = []  # (declared_tenant, observed_token)
    lock = asyncio.Lock()

    async def slow_handler(declared: str):
        # Sleep a random amount to force interleaving
        await asyncio.sleep(random.uniform(0.001, 0.01))
        ctx = get_fhir_context()
        async with lock:
            observed.append((declared, ctx.access_token))
        return "ok"

    async def make_call(tenant: str):
        # Wrap slow_handler with the tenant info passed at call time
        async def handler(_):
            return await slow_handler(tenant)
        await sharp_context_middleware(_make_request(tenant), handler)

    async def run_all():
        tenants = [f"tenant-{i}" for i in range(n_concurrent)]
        await asyncio.gather(*(make_call(t) for t in tenants))

    asyncio.run(run_all())

    # Every observation must be (tenant-i, tok-tenant-i) -- no cross-leak
    assert len(observed) == n_concurrent
    for declared, observed_token in observed:
        assert observed_token == f"tok-{declared}", (
            f"Cross-tenant leak: declared {declared!r} but saw token {observed_token!r}"
        )


# ─────────────────────── Context cleanup on error path ───────────────────────

def test_context_resets_after_handler_exception():
    """If the call_next raises, the ContextVar must still be reset cleanly."""

    async def failing_handler(_):
        raise RuntimeError("downstream blew up")

    async def run():
        try:
            await sharp_context_middleware(_make_request("alpha"), failing_handler)
        except RuntimeError:
            pass
        # The context MUST be reset even on the error path
        with pytest.raises(LookupError):
            get_fhir_context()

    asyncio.run(run())


# ─────────────────────── Missing-context returns 403, no propagation ───────────────────────

def test_missing_context_does_not_set_contextvar():
    """If headers are missing, the middleware returns 403 and the ContextVar
    must NOT be set (otherwise downstream callers in the same event loop
    might mistakenly believe a context exists)."""

    async def must_not_run(_):
        raise AssertionError("call_next must not run when context is missing")

    async def run():
        # No headers
        req = _StubRequest({})
        resp = await sharp_context_middleware(req, must_not_run)
        assert resp.status_code == 403
        body = json.loads(resp.body)
        assert body["error"] == "missing_fhir_context"
        # ContextVar must remain unset
        with pytest.raises(LookupError):
            get_fhir_context()

    asyncio.run(run())


# ─────────────────────── Middleware composes with itself (stress) ───────────────────────

def test_nested_middleware_inner_overrides_outer():
    """Calling the middleware from within an outer middleware should give
    the inner the inner's context -- not the outer's. (Defensive; not a real
    deployment pattern but verifies ContextVar.set/reset semantics.)"""
    inner_seen: dict[str, str] = {}

    async def inner_handler(_):
        ctx = get_fhir_context()
        inner_seen["server_url"] = ctx.server_url
        return "inner-ok"

    async def outer_handler(_):
        # This handler IS the call_next under the outer middleware. It then
        # re-invokes the middleware with a different tenant.
        return await sharp_context_middleware(_make_request("inner"), inner_handler)

    async def run():
        await sharp_context_middleware(_make_request("outer"), outer_handler)

    asyncio.run(run())
    assert inner_seen["server_url"] == "https://hapi.inner.example.com/fhir"
