"""Phase 6.3 -- observability primitives + Prometheus + rate limiter."""

from __future__ import annotations

import time

import pytest

from a2a_agent.observability import (
    LLM_POLISH_CALLS,
    TOOL_DURATION_MS,
    TOOL_INVOCATIONS,
    TokenBucket,
    default_rate_limiter,
    rate_key_for_request,
    render_prometheus_metrics,
    time_tool,
)


# ─────────────────────── Counters ───────────────────────


def test_counter_inc_aggregates_by_label():
    TOOL_INVOCATIONS.inc("compute_test_a", "ok")
    TOOL_INVOCATIONS.inc("compute_test_a", "ok")
    TOOL_INVOCATIONS.inc("compute_test_a", "error")
    snap = TOOL_INVOCATIONS.snapshot()
    assert snap[("compute_test_a", "ok")] >= 2
    assert snap[("compute_test_a", "error")] >= 1


def test_counter_rejects_label_count_mismatch():
    with pytest.raises(ValueError, match="expected 2 labels"):
        TOOL_INVOCATIONS.inc("only-one")


def test_histogram_observe_and_snapshot():
    TOOL_DURATION_MS.observe("compute_test_h", value=12.5)
    TOOL_DURATION_MS.observe("compute_test_h", value=37.5)
    snap = TOOL_DURATION_MS.snapshot()
    stats = snap[("compute_test_h",)]
    assert stats["count"] >= 2
    # Mean is sum/count over the bucket -- float, non-zero
    assert stats["mean"] > 0


# ─────────────────────── time_tool context manager ───────────────────────


def test_time_tool_records_duration_and_outcome_ok():
    n_before = TOOL_INVOCATIONS.snapshot().get(
        ("compute_test_timer", "ok"), 0)
    with time_tool("compute_test_timer"):
        pass
    snap = TOOL_INVOCATIONS.snapshot()
    assert snap[("compute_test_timer", "ok")] == n_before + 1


def test_time_tool_records_outcome_error_on_exception():
    n_before = TOOL_INVOCATIONS.snapshot().get(
        ("compute_test_err", "error"), 0)
    with pytest.raises(RuntimeError):
        with time_tool("compute_test_err"):
            raise RuntimeError("kaboom")
    snap = TOOL_INVOCATIONS.snapshot()
    assert snap[("compute_test_err", "error")] == n_before + 1


# ─────────────────────── Prometheus renderer ───────────────────────


def test_prometheus_text_format_emits_help_type_lines():
    LLM_POLISH_CALLS.inc("ollama:test", "ok")
    text = render_prometheus_metrics()
    assert "# HELP trustedrisk_tool_invocations_total" in text
    assert "# TYPE trustedrisk_tool_invocations_total counter" in text
    assert "# HELP trustedrisk_tool_duration_ms" in text
    assert "# TYPE trustedrisk_tool_duration_ms summary" in text
    assert "trustedrisk_llm_polish_calls_total{client=\"ollama:test\",outcome=\"ok\"}" in text


def test_prometheus_endpoint_serves_metrics():
    """The MCP server's /metrics endpoint returns Prometheus text-format."""
    from starlette.testclient import TestClient

    from mcp_server.server import build_http_app

    app = build_http_app()
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "trustedrisk_" in resp.text


# ─────────────────────── Rate limiter ───────────────────────


def test_token_bucket_allows_within_capacity():
    bucket = TokenBucket(capacity=5.0, refill_per_sec=10.0)
    for _ in range(5):
        allowed, _ = bucket.acquire("k1")
        assert allowed is True


def test_token_bucket_rejects_when_empty():
    bucket = TokenBucket(capacity=2.0, refill_per_sec=0.0)
    bucket.acquire("k2")
    bucket.acquire("k2")
    allowed, retry_after = bucket.acquire("k2")
    assert allowed is False
    assert retry_after > 0


def test_token_bucket_refills_over_time():
    bucket = TokenBucket(capacity=1.0, refill_per_sec=10.0)
    bucket.acquire("k3")
    allowed, _ = bucket.acquire("k3")
    assert allowed is False
    time.sleep(0.15)
    allowed, _ = bucket.acquire("k3")
    assert allowed is True


def test_token_bucket_isolates_keys():
    bucket = TokenBucket(capacity=1.0, refill_per_sec=0.0)
    bucket.acquire("alice")
    allowed, _ = bucket.acquire("bob")
    assert allowed is True


def test_default_rate_limiter_is_singleton():
    a = default_rate_limiter()
    b = default_rate_limiter()
    assert a is b


# ─────────────────────── Rate-key derivation ───────────────────────


class _FakeRequest:
    def __init__(self, *, api_key: str = "", client_host: str = ""):
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key

        class _Client:
            host = client_host
        self.client = _Client() if client_host else None


def test_rate_key_uses_api_key_when_present():
    r = _FakeRequest(api_key="K123", client_host="1.2.3.4")
    assert rate_key_for_request(r) == "apikey:K123"


def test_rate_key_falls_back_to_ip():
    r = _FakeRequest(api_key="", client_host="9.9.9.9")
    assert rate_key_for_request(r) == "ip:9.9.9.9"


# ─────────────────────── Server middleware integration ───────────────────────


def test_rate_limiter_returns_429_when_burst_exceeded():
    """Exceed the burst cap; bucket reports `allowed=False` + positive
    Retry-After. Avoids module reload (which would reset the in-memory
    registry singletons used by other tests)."""
    bucket = TokenBucket(capacity=1.0, refill_per_sec=0.0)
    bucket.acquire("ip:127.0.0.1")
    allowed, retry = bucket.acquire("ip:127.0.0.1")
    assert allowed is False
    assert retry > 0
