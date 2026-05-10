"""Phase 13.13 G2 -- Saga composer cross-agent.

Implements a saga-pattern composer that chains multiple specialist
calls with idempotency keys + compensating actions on failure.

Why a saga? In a federation each specialist is a separate process
(production: separate Cloud Run services). A multi-step clinical
workflow (e.g. discharge -> PA -> patient counseling -> write-back to
FHIR) cannot be a single 2-phase-commit transaction. The saga
pattern is the textbook fix:

  - **Forward steps** are individual specialist calls, each guarded
    by an idempotency key derived from the saga id.
  - **Compensating actions** are inverse operations (e.g. "delete
    the Composition we wrote at step 4") that the composer runs in
    reverse order when a downstream step fails.
  - **State machine** persists per-saga progress so retries are
    deterministic.

This is an in-process implementation: each step is an async
callable, compensations are the reverse-mapped callables, and the
state machine is a simple dict log keyed by saga id. A future
production deployment would persist the log to a Postgres table
or an event-sourced ledger.

References:
  - Garcia-Molina H, Salem K. Sagas. ACM SIGMOD 1987.
  - Richardson C. Microservices Patterns. Manning 2018, ch. 4.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal


SagaStepStatus = Literal["pending", "running", "succeeded", "failed",
                                "compensated", "compensation_failed"]


@dataclass
class SagaStep:
    name: str
    forward: Callable[[dict[str, Any]], Awaitable[Any]]
    compensate: Callable[[dict[str, Any], Any], Awaitable[Any]] | None = None
    max_retries: int = 0
    retry_backoff_seconds: float = 0.5


@dataclass
class SagaStepLog:
    name: str
    status: SagaStepStatus = "pending"
    output: Any = None
    error: str | None = None
    n_retries: int = 0
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class SagaResult:
    saga_id: str
    posture: Literal["committed", "compensated", "compensation_failed"]
    n_steps_total: int
    n_steps_completed: int
    n_steps_compensated: int
    duration_seconds: float
    steps: list[SagaStepLog] = field(default_factory=list)
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────────
# Public runner
# ─────────────────────────────────────────────────────────────────────


async def run_saga(
    steps: list[SagaStep],
    *,
    saga_id: str | None = None,
    initial_context: dict[str, Any] | None = None,
) -> SagaResult:
    """Execute a saga.

    Each `forward` call receives a `context` dict that accumulates the
    previous steps' outputs (`context[step.name] = output`). On any
    forward failure the composer walks back through completed steps
    in reverse order and invokes `compensate(context, output)` for
    each. Idempotency keys are derived as `f"{saga_id}-{step.name}"`
    and made available to forward callables under
    `context["__idempotency_key"]`.
    """
    sid = saga_id or f"saga-{uuid.uuid4().hex[:12]}"
    context: dict[str, Any] = dict(initial_context or {})
    logs: list[SagaStepLog] = [SagaStepLog(name=s.name) for s in steps]
    completed: list[tuple[SagaStep, SagaStepLog]] = []
    t0 = time.perf_counter()

    posture: str = "committed"

    for idx, step in enumerate(steps):
        log = logs[idx]
        log.started_at = time.time()
        log.status = "running"
        context["__idempotency_key"] = f"{sid}-{step.name}"

        attempts = 0
        while True:
            try:
                out = await step.forward(context)
                log.output = out
                log.status = "succeeded"
                log.finished_at = time.time()
                context[step.name] = out
                completed.append((step, log))
                break
            except Exception as exc:    # noqa: BLE001
                if attempts < step.max_retries:
                    attempts += 1
                    log.n_retries = attempts
                    await asyncio.sleep(step.retry_backoff_seconds)
                    continue
                log.status = "failed"
                log.error = f"{type(exc).__name__}: {exc}"
                log.finished_at = time.time()
                posture = "compensated"
                break

        if log.status == "failed":
            break

    n_completed = sum(1 for log in logs if log.status == "succeeded")
    n_compensated = 0

    if posture == "compensated":
        # Walk back through completed steps in reverse and run compensations
        for step, log in reversed(completed):
            if step.compensate is None:
                # No compensator -> still mark as compensated (no-op)
                log.status = "compensated"
                n_compensated += 1
                continue
            try:
                await step.compensate(context, log.output)
                log.status = "compensated"
                n_compensated += 1
            except Exception as exc:    # noqa: BLE001
                log.status = "compensation_failed"
                log.error = (
                    (log.error or "") + f"; compensation: {exc}"
                )
                posture = "compensation_failed"

    duration = time.perf_counter() - t0
    rationale = (
        f"Saga {sid} {posture} after {duration:.3f}s; "
        f"{n_completed}/{len(steps)} forward steps completed; "
        f"{n_compensated} compensations executed."
    )
    return SagaResult(
        saga_id=sid, posture=posture,                     # type: ignore[arg-type]
        n_steps_total=len(steps),
        n_steps_completed=n_completed,
        n_steps_compensated=n_compensated,
        duration_seconds=round(duration, 4),
        steps=logs, rationale=rationale,
    )


# ─────────────────────────────────────────────────────────────────────
# Pre-built clinical saga: discharge -> PA -> patient counseling
# ─────────────────────────────────────────────────────────────────────


def build_discharge_saga() -> list[SagaStep]:
    """A representative cross-agent saga: discharge counseling -> PA
    evidence pack -> patient FAQ -> idempotent FHIR write-back.

    Each step is wired to the real MCP tools; the saga runner takes
    care of idempotency keys and compensation on failure."""

    async def step_discharge(ctx: dict[str, Any]) -> Any:
        from mcp_server.tools.discharge_counseling import (
            compute_discharge_counseling,
        )
        return await compute_discharge_counseling(
            medications=ctx.get("medications", []),
            lace_score=ctx.get("lace_score", 6),
            recommendation_action=ctx.get(
                "recommendation_action", "discharge_home",
            ),
        )

    async def step_pa_evidence(ctx: dict[str, Any]) -> Any:
        # Skip PA when no service is requested -- keeps the saga
        # generic for non-PA discharges.
        if not ctx.get("requested_service"):
            return {"skipped": True}
        from mcp_server.tools.pa_evidence_pack import (
            compute_pa_evidence_pack,
        )
        return await compute_pa_evidence_pack(
            requested_service=ctx["requested_service"],
            patient_age=ctx.get("patient_age", 65),
        )

    async def step_patient_faq(ctx: dict[str, Any]) -> Any:
        from mcp_server.tools.patient_faq import compute_patient_faq
        return await compute_patient_faq(
            question=ctx.get(
                "patient_question",
                "When can I go home?",
            ),
        )

    async def step_writeback(ctx: dict[str, Any]) -> Any:
        # Optional -- only invoked when a FHIR context is bound. We
        # surface a no-op result when the SHARP context is missing,
        # so the saga is testable in pure-deterministic CI.
        from mcp_server.sharp.headers import get_fhir_context
        try:
            ctx_fhir = get_fhir_context()
        except LookupError:
            return {"skipped": True, "reason": "no_fhir_context"}
        if not ctx_fhir.server_url:
            return {"skipped": True, "reason": "no_server_url"}
        from mcp_server.tools.fhir_writeback import (
            compute_write_decision_to_fhir,
        )
        return await compute_write_decision_to_fhir(
            decision_summary=str(ctx.get("recommendation_action") or "")[:60],
            decision_card_logical_id=ctx["__idempotency_key"],
            recommended_action=ctx.get("recommendation_action"),
            rationale_text=ctx.get("rationale_text", ""),
        )

    async def compensate_writeback(ctx: dict[str, Any], output: Any) -> Any:
        """If write-back succeeded but a later step failed, the
        composer would normally DELETE the Composition. Here we
        simply tag the output with `compensated=True` for the audit
        trail; production deployments would issue a FHIR DELETE."""
        if isinstance(output, dict) and output.get("skipped"):
            return None
        return {"compensated": True, "original_id": getattr(
            output, "composition_id", None,
        )}

    return [
        SagaStep("discharge_counseling", step_discharge),
        SagaStep("pa_evidence_pack", step_pa_evidence),
        SagaStep("patient_faq", step_patient_faq),
        SagaStep(
            "fhir_writeback", step_writeback,
            compensate=compensate_writeback,
        ),
    ]
