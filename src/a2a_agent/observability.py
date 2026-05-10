"""Phase 6.3 -- Observability primitives (Prometheus + OpenTelemetry).

Lightweight in-process counters + an optional OTLP exporter hook.
Designed so a Cloud Run deployment with `OTEL_EXPORTER_OTLP_ENDPOINT`
configured streams traces to a managed collector (Cloud Trace, Honeycomb,
etc.) without any code changes -- the env var is the only switch.

Pure-Python; no hard dependency on `prometheus-client` or
`opentelemetry-sdk`. When those packages are absent, the module
exposes no-op stubs so production-runtime behaviour stays identical.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from threading import Lock
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# In-process counters (always available -- Prometheus optional)
# ─────────────────────────────────────────────────────────────────────


class _Counter:
    def __init__(self, name: str, description: str,
                  label_names: tuple[str, ...] = ()):
        self.name = name
        self.description = description
        self.label_names = label_names
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = Lock()

    def inc(self, *label_values: str, value: float = 1.0) -> None:
        if len(label_values) != len(self.label_names):
            raise ValueError(
                f"counter {self.name!r}: expected "
                f"{len(self.label_names)} labels, got {len(label_values)}"
            )
        with self._lock:
            self._values[label_values] = (
                self._values.get(label_values, 0.0) + value
            )

    def snapshot(self) -> dict[tuple[str, ...], float]:
        with self._lock:
            return dict(self._values)


class _Histogram:
    def __init__(self, name: str, description: str,
                  label_names: tuple[str, ...] = ()):
        self.name = name
        self.description = description
        self.label_names = label_names
        self._counts: dict[tuple[str, ...], int] = {}
        self._sums: dict[tuple[str, ...], float] = {}
        self._lock = Lock()

    def observe(self, *label_values: str, value: float) -> None:
        if len(label_values) != len(self.label_names):
            raise ValueError(
                f"histogram {self.name!r}: label count mismatch"
            )
        with self._lock:
            self._counts[label_values] = self._counts.get(label_values, 0) + 1
            self._sums[label_values] = (
                self._sums.get(label_values, 0.0) + value
            )

    def snapshot(self) -> dict[tuple[str, ...], dict[str, float]]:
        with self._lock:
            out = {}
            for k, n in self._counts.items():
                out[k] = {
                    "count": float(n),
                    "sum": self._sums.get(k, 0.0),
                    "mean": (self._sums.get(k, 0.0) / n) if n else 0.0,
                }
            return out


# Module-level metric registry (populated lazily)
TOOL_INVOCATIONS = _Counter(
    "trustedrisk_tool_invocations_total",
    "MCP tool invocations grouped by tool name + outcome",
    ("tool", "outcome"),
)

TOOL_DURATION_MS = _Histogram(
    "trustedrisk_tool_duration_ms",
    "MCP tool wall-clock duration in milliseconds",
    ("tool",),
)

A2A_TASKS = _Counter(
    "trustedrisk_a2a_tasks_total",
    "A2A tasks grouped by terminal state",
    ("specialist", "state"),
)

LLM_POLISH_CALLS = _Counter(
    "trustedrisk_llm_polish_calls_total",
    "LLM polish calls grouped by client + outcome",
    ("client", "outcome"),
)

REGISTRY = {
    "counters": [TOOL_INVOCATIONS, A2A_TASKS, LLM_POLISH_CALLS],
    "histograms": [TOOL_DURATION_MS],
}


# ─────────────────────────────────────────────────────────────────────
# Prometheus-format renderer (no external deps)
# ─────────────────────────────────────────────────────────────────────


def render_prometheus_metrics() -> str:
    """Render the in-process registry to a Prometheus text-format
    payload. Compatible with `prometheus-client` parsers.
    """
    lines: list[str] = []
    for c in REGISTRY["counters"]:
        lines.append(f"# HELP {c.name} {c.description}")
        lines.append(f"# TYPE {c.name} counter")
        for labels, val in c.snapshot().items():
            label_str = ""
            if c.label_names:
                kvs = [f'{k}="{v}"' for k, v in zip(c.label_names, labels)]
                label_str = "{" + ",".join(kvs) + "}"
            lines.append(f"{c.name}{label_str} {val}")
    for h in REGISTRY["histograms"]:
        lines.append(f"# HELP {h.name} {h.description}")
        lines.append(f"# TYPE {h.name} summary")
        for labels, stats in h.snapshot().items():
            label_str = ""
            if h.label_names:
                kvs = [f'{k}="{v}"' for k, v in zip(h.label_names, labels)]
                label_str = "{" + ",".join(kvs) + "}"
            lines.append(f"{h.name}_count{label_str} {stats['count']}")
            lines.append(f"{h.name}_sum{label_str} {stats['sum']}")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────
# Tool invocation timer
# ─────────────────────────────────────────────────────────────────────


@contextmanager
def time_tool(tool_name: str):
    """Context manager that records duration + outcome for a tool call.

    Usage:
        with time_tool("compute_readmission_risk"):
            result = await tool_function(...)
    """
    start = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        dur_ms = (time.perf_counter() - start) * 1000.0
        TOOL_INVOCATIONS.inc(tool_name, outcome)
        TOOL_DURATION_MS.observe(tool_name, value=dur_ms)


# ─────────────────────────────────────────────────────────────────────
# OpenTelemetry hook (optional)
# ─────────────────────────────────────────────────────────────────────


_OTEL_INITIALISED = False


def maybe_init_otel() -> bool:
    """Initialise the OpenTelemetry tracer + OTLP exporter when
    `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Idempotent.

    Returns True when OTel was wired, False otherwise.
    """
    global _OTEL_INITIALISED
    if _OTEL_INITIALISED:
        return True
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return False
    try:
        from opentelemetry import trace                       # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )                                                       # type: ignore
        from opentelemetry.sdk.resources import Resource       # type: ignore
        from opentelemetry.sdk.trace import TracerProvider    # type: ignore
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )                                                       # type: ignore
    except ImportError:
        return False

    resource = Resource.create({"service.name": os.environ.get(
        "OTEL_SERVICE_NAME", "trustedrisk")})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _OTEL_INITIALISED = True
    return True


# ─────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter
# ─────────────────────────────────────────────────────────────────────


class TokenBucket:
    """Per-key token bucket. Async-safe via a single Lock.

    Production deployments should use a Redis-backed implementation
    (multi-process consistency); this in-process bucket is sufficient
    for a single-replica Cloud Run deployment with `min-instances=1`.
    """

    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = capacity
        self.refill = refill_per_sec
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._lock = Lock()

    def acquire(self, key: str, *, cost: float = 1.0) -> tuple[bool, float]:
        """Try to consume `cost` tokens for `key`.

        Returns (allowed, retry_after_seconds). When `allowed` is False
        the caller should respond 429 with a `Retry-After` header.
        """
        now = time.monotonic()
        with self._lock:
            tokens = self._tokens.get(key, self.capacity)
            last = self._last.get(key, now)
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= cost:
                self._tokens[key] = tokens - cost
                self._last[key] = now
                return (True, 0.0)
            self._tokens[key] = tokens
            self._last[key] = now
            deficit = cost - tokens
            return (False, deficit / max(self.refill, 1e-6))


# Default bucket sizing -- 60 req / sec sustained, 120 burst
_DEFAULT_BUCKET = TokenBucket(
    capacity=float(os.environ.get(
        "TRUSTEDRISK_RATE_LIMIT_BURST", "120")),
    refill_per_sec=float(os.environ.get(
        "TRUSTEDRISK_RATE_LIMIT_PER_SEC", "60")),
)


def default_rate_limiter() -> TokenBucket:
    return _DEFAULT_BUCKET


def rate_key_for_request(request: Any) -> str:
    """Derive the rate-limit key from a Starlette Request -- defaults to
    the API key (when present) or the client IP."""
    api_key = request.headers.get("X-API-Key", "").strip()
    if api_key:
        return f"apikey:{api_key}"
    client_host = (request.client.host
                       if getattr(request, "client", None) else "unknown")
    return f"ip:{client_host}"
