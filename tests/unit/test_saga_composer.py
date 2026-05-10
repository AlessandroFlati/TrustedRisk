"""Phase 13.13 G2 -- Saga composer cross-agent tests."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.saga_composer import (
    SagaStep, build_discharge_saga, run_saga,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Fundamental saga semantics
# ─────────────────────────────────────────────────────────────────────

def test_saga_commits_when_all_steps_succeed():
    async def step_a(ctx): return "a-out"
    async def step_b(ctx): return "b-out"
    async def step_c(ctx): return "c-out"
    res = _run(run_saga([
        SagaStep("a", step_a), SagaStep("b", step_b), SagaStep("c", step_c),
    ]))
    assert res.posture == "committed"
    assert res.n_steps_completed == 3
    assert res.n_steps_compensated == 0


def test_saga_compensates_in_reverse_order_on_step_failure():
    """When step C fails, the composer compensates B then A in reverse."""
    compensate_log: list[str] = []

    async def step_a(ctx): return "a-out"
    async def step_b(ctx): return "b-out"
    async def step_c(ctx): raise RuntimeError("boom")

    async def comp_a(ctx, out): compensate_log.append("a")
    async def comp_b(ctx, out): compensate_log.append("b")

    res = _run(run_saga([
        SagaStep("a", step_a, compensate=comp_a),
        SagaStep("b", step_b, compensate=comp_b),
        SagaStep("c", step_c),
    ]))
    assert res.posture == "compensated"
    assert res.n_steps_completed == 2
    assert res.n_steps_compensated == 2
    # Compensations in reverse order: b then a
    assert compensate_log == ["b", "a"]


def test_saga_step_without_compensator_marked_compensated_no_op():
    async def step_a(ctx): return "a-out"
    async def step_b(ctx): raise ValueError("fail")
    res = _run(run_saga([
        SagaStep("a", step_a),    # no compensate
        SagaStep("b", step_b),
    ]))
    assert res.posture == "compensated"
    a_log = next(s for s in res.steps if s.name == "a")
    assert a_log.status == "compensated"


def test_saga_compensation_failed_propagates_to_posture():
    async def step_a(ctx): return "a-out"
    async def step_b(ctx): raise RuntimeError("boom")
    async def comp_a(ctx, out): raise RuntimeError("compensation failed")
    res = _run(run_saga([
        SagaStep("a", step_a, compensate=comp_a),
        SagaStep("b", step_b),
    ]))
    assert res.posture == "compensation_failed"
    a_log = next(s for s in res.steps if s.name == "a")
    assert a_log.status == "compensation_failed"


# ─────────────────────────────────────────────────────────────────────
# Idempotency keys
# ─────────────────────────────────────────────────────────────────────

def test_idempotency_key_is_passed_into_forward_callable():
    captured: list[str] = []

    async def step_a(ctx):
        captured.append(ctx["__idempotency_key"])
        return "a-out"

    res = _run(run_saga(
        [SagaStep("a", step_a)],
        saga_id="my-saga",
    ))
    assert res.posture == "committed"
    assert captured == ["my-saga-a"]


def test_idempotency_keys_are_unique_across_steps():
    captured: list[str] = []

    async def step(ctx):
        captured.append(ctx["__idempotency_key"])
        return None

    _run(run_saga(
        [SagaStep("first", step), SagaStep("second", step)],
        saga_id="abc",
    ))
    assert captured == ["abc-first", "abc-second"]


def test_saga_id_auto_generated_when_omitted():
    async def step_a(ctx): return "x"
    res = _run(run_saga([SagaStep("a", step_a)]))
    assert res.saga_id.startswith("saga-")


# ─────────────────────────────────────────────────────────────────────
# Retries with backoff
# ─────────────────────────────────────────────────────────────────────

def test_saga_retries_failing_step_up_to_max_retries():
    attempts = {"n": 0}

    async def flaky(ctx):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    res = _run(run_saga([
        SagaStep("flaky", flaky, max_retries=3,
                     retry_backoff_seconds=0.01),
    ]))
    assert res.posture == "committed"
    assert res.steps[0].n_retries == 2
    assert attempts["n"] == 3


def test_saga_retries_exhausted_triggers_compensation():
    attempts = {"n": 0}

    async def always_fails(ctx):
        attempts["n"] += 1
        raise RuntimeError("permanent")

    res = _run(run_saga([
        SagaStep("a", always_fails, max_retries=2,
                     retry_backoff_seconds=0.01),
    ]))
    assert res.posture == "compensated"
    assert attempts["n"] == 3   # initial + 2 retries


# ─────────────────────────────────────────────────────────────────────
# Context propagation
# ─────────────────────────────────────────────────────────────────────

def test_step_outputs_accumulate_into_context_for_downstream_steps():
    captured = []

    async def step_a(ctx): return "alpha"
    async def step_b(ctx):
        captured.append(ctx.get("a"))
        return "beta"

    _run(run_saga([
        SagaStep("a", step_a), SagaStep("b", step_b),
    ]))
    assert captured == ["alpha"]


def test_initial_context_visible_to_first_step():
    captured = []

    async def step_a(ctx):
        captured.append(ctx.get("seed"))
        return None

    _run(run_saga(
        [SagaStep("a", step_a)],
        initial_context={"seed": "VALUE"},
    ))
    assert captured == ["VALUE"]


# ─────────────────────────────────────────────────────────────────────
# Pre-built clinical saga
# ─────────────────────────────────────────────────────────────────────

def test_build_discharge_saga_yields_four_steps():
    saga = build_discharge_saga()
    assert len(saga) == 4
    names = [s.name for s in saga]
    assert names == [
        "discharge_counseling", "pa_evidence_pack",
        "patient_faq", "fhir_writeback",
    ]


def test_clinical_saga_runs_to_committed_without_fhir_context():
    """The fhir_writeback step skips gracefully when no FHIR context
    is bound, so the saga commits end-to-end in pure-CI mode."""
    res = _run(run_saga(
        build_discharge_saga(),
        initial_context={
            "medications": [
                {"name": "warfarin", "dose_mg": 5, "status": "active"},
            ],
            "lace_score": 6,
            "recommendation_action": "discharge_home",
            "patient_question": "When can I go home?",
            "patient_age": 65,
        },
    ))
    assert res.posture == "committed"
    assert res.n_steps_completed == 4
    # The write-back step skipped since there's no SHARP context
    wb = next(s for s in res.steps if s.name == "fhir_writeback")
    assert isinstance(wb.output, dict) and wb.output.get("skipped") is True
