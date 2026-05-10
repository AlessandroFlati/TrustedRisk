"""Phase 8.4 -- Production-grade circuit breaker for FHIR upstreams.

Per-FHIR-server circuit-breaker with three states (closed / open /
half-open), exponential-backoff retry on transient errors, and a
single point of integration with `fhirpy`'s async client.

Pure-Python; no hard dep on a circuit-breaker library. The state is
process-local (no Redis); this is sufficient for a single-replica
Cloud Run deployment with `min-instances=1`. Multi-replica deployments
should bind a Redis-backed implementation by replacing
`default_breaker_for()`.

States
------
- **closed**: requests pass through. Failures are counted; when the
  failure rate exceeds the threshold, transition to `open`.
- **open**: requests fail fast (raises `CircuitOpen`) without hitting
  the upstream. After `cooldown_s` elapses, transition to `half_open`.
- **half_open**: a single probe request is allowed; on success ->
  `closed`; on failure -> back to `open` with cooldown reset.

Thresholds
----------
- `failure_threshold` = 5 errors in `window_s` = 60s window.
- `cooldown_s` = 30s.
- `max_retry_attempts` = 3 with exponential backoff (50ms × 2^n).

Configurable via env:
- `TRUSTEDRISK_FHIR_FAILURE_THRESHOLD`
- `TRUSTEDRISK_FHIR_WINDOW_S`
- `TRUSTEDRISK_FHIR_COOLDOWN_S`
- `TRUSTEDRISK_FHIR_MAX_RETRY`
"""

from __future__ import annotations

import asyncio
import os
import time
from threading import Lock
from typing import Any, Awaitable, Callable, TypeVar


T = TypeVar("T")


class CircuitOpen(RuntimeError):
    """Raised when the breaker is open and the request fails fast."""


class _Breaker:
    def __init__(self, key: str):
        self.key = key
        self.failure_threshold = int(os.environ.get(
            "TRUSTEDRISK_FHIR_FAILURE_THRESHOLD", "5"))
        self.window_s = float(os.environ.get(
            "TRUSTEDRISK_FHIR_WINDOW_S", "60"))
        self.cooldown_s = float(os.environ.get(
            "TRUSTEDRISK_FHIR_COOLDOWN_S", "30"))
        self.failures: list[float] = []
        self.state: str = "closed"
        self.opened_at: float = 0.0
        self._lock = Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        self.failures = [t for t in self.failures if t >= cutoff]

    def record_failure(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            self.failures.append(now)
            if len(self.failures) >= self.failure_threshold:
                self.state = "open"
                self.opened_at = now

    def record_success(self) -> None:
        with self._lock:
            self.failures.clear()
            self.state = "closed"

    def check_state(self) -> str:
        now = time.monotonic()
        with self._lock:
            if self.state == "open" and (
                    now - self.opened_at) >= self.cooldown_s:
                self.state = "half_open"
            return self.state


_breakers: dict[str, _Breaker] = {}
_registry_lock = Lock()


def default_breaker_for(fhir_server_url: str) -> _Breaker:
    """Return (or create) the per-server breaker. Keyed by URL."""
    with _registry_lock:
        if fhir_server_url not in _breakers:
            _breakers[fhir_server_url] = _Breaker(fhir_server_url)
        return _breakers[fhir_server_url]


def reset_all() -> None:
    """For tests."""
    with _registry_lock:
        _breakers.clear()


# ─────────────────────────────────────────────────────────────────────
# Retry + circuit-breaker wrapper
# ─────────────────────────────────────────────────────────────────────


def _is_transient(exc: Exception) -> bool:
    """Detect retryable upstream errors (5xx, network)."""
    status = getattr(exc, "response", None)
    if status is not None:
        code = getattr(status, "status_code", 0) or 0
        if 500 <= code < 600:
            return True
        # 408 Request Timeout
        if code == 408:
            return True
    code = getattr(exc, "status_code", 0) or 0
    if 500 <= code < 600 or code == 408:
        return True
    msg = str(exc).lower()
    return any(tok in msg for tok in (
        "connection reset", "timed out", "connect error",
        "service unavailable", "bad gateway",
    ))


async def with_circuit_breaker(
    fhir_server_url: str,
    func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute `func(*args, **kwargs)` through the per-server circuit
    breaker + retry wrapper.

    Behaviour:
    - state=open -> CircuitOpen raised immediately.
    - state=closed/half_open -> at most `max_retry_attempts` tries
      with exponential backoff (50ms, 100ms, 200ms).
    - successful call -> reset breaker (closed).
    - non-transient error (4xx auth, ValueError, ...) -> raised raw,
      not counted as breaker failure.
    """
    breaker = default_breaker_for(fhir_server_url)
    state = breaker.check_state()
    if state == "open":
        raise CircuitOpen(
            f"Circuit open for FHIR server {fhir_server_url!r}; "
            f"cooldown for ~{breaker.cooldown_s:.0f}s before retry."
        )

    max_retry = int(os.environ.get("TRUSTEDRISK_FHIR_MAX_RETRY", "3"))
    last_exc: Exception | None = None
    for attempt in range(max_retry):
        try:
            result = await func(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception as exc:
            if _is_transient(exc):
                breaker.record_failure()
                last_exc = exc
                if attempt < max_retry - 1:
                    await asyncio.sleep(0.05 * (2 ** attempt))
                    continue
                # All retries exhausted; if breaker tripped, raise CircuitOpen
                if breaker.check_state() == "open":
                    raise CircuitOpen(
                        f"Circuit opened on FHIR server "
                        f"{fhir_server_url!r} after {max_retry} retries: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                raise
            # Non-transient -- surface raw, do not count as breaker failure
            raise
    # Should not reach
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("with_circuit_breaker: unreachable")


# ─────────────────────────────────────────────────────────────────────
# FHIR Bundle pagination helper
# ─────────────────────────────────────────────────────────────────────


async def paginate_bundle(
    fetch_first_page: Callable[..., Awaitable[dict]],
    *,
    max_pages: int = 50,
) -> dict:
    """Walk the `link[rel='next']` chain of a paginated FHIR Bundle and
    return a single concatenated Bundle.

    `fetch_first_page` must accept `url=...` for subsequent pages and
    return the same Bundle shape.
    """
    bundle = await fetch_first_page()
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        return bundle

    all_entries = list(bundle.get("entry", []) or [])
    seen_pages = 1
    while seen_pages < max_pages:
        links = bundle.get("link", []) or []
        next_url = next(
            (lk.get("url") for lk in links if lk.get("relation") == "next"),
            None,
        )
        if not next_url:
            break
        bundle = await fetch_first_page(url=next_url)
        all_entries.extend(bundle.get("entry", []) or [])
        seen_pages += 1

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": all_entries,
        "_n_pages_walked": seen_pages,
    }
