"""Phase 12.5 B3 -- per-scenario counterfactual analysis tests."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.scenario_counterfactuals import (
    list_counterfactuals,
    run_all_counterfactuals,
    run_counterfactual,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Per-scenario contract
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sid", [
    "acute_stroke_lvo",
    "sepsis_bundle",
    "polytrauma_mtp",
    "geriatric_polypharmacy",
    "mental_health_crisis",
])
def test_each_scenario_evaluates_at_least_3_perturbations(sid: str):
    rep = _run(run_counterfactual(sid))
    assert rep.scenario_id == sid
    assert rep.perturbations_evaluated >= 3


@pytest.mark.parametrize("sid", [
    "acute_stroke_lvo",
    "sepsis_bundle",
    "polytrauma_mtp",
    "geriatric_polypharmacy",
    "mental_health_crisis",
])
def test_each_scenario_finds_at_least_one_flip(sid: str):
    rep = _run(run_counterfactual(sid))
    assert rep.n_flips >= 1, (
        f"{sid}: expected ≥1 flip, got {rep.n_flips}. "
        f"Rationale: {rep.rationale}"
    )


@pytest.mark.parametrize("sid", [
    "acute_stroke_lvo",
    "sepsis_bundle",
    "polytrauma_mtp",
    "geriatric_polypharmacy",
    "mental_health_crisis",
])
def test_each_scenario_baseline_outcome_non_empty(sid: str):
    rep = _run(run_counterfactual(sid))
    assert rep.baseline_outcome.strip()


# ─────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────

def test_run_all_returns_five():
    reps = _run(run_all_counterfactuals())
    ids = {r.scenario_id for r in reps}
    assert ids == {
        "acute_stroke_lvo", "sepsis_bundle", "polytrauma_mtp",
        "geriatric_polypharmacy", "mental_health_crisis",
    }


def test_cumulative_flip_count_at_least_5():
    """Across the 5 scenarios, at least 5 of the 15 perturbations
    must produce a tier change. Sets a non-trivial floor."""
    reps = _run(run_all_counterfactuals())
    total_flips = sum(r.n_flips for r in reps)
    assert total_flips >= 5


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        _run(run_counterfactual("nope"))


def test_list_matches_run_all():
    listed = set(list_counterfactuals())
    reps = _run(run_all_counterfactuals())
    found = {r.scenario_id for r in reps}
    assert listed == found


# ─────────────────────────────────────────────────────────────────────
# Flip semantics
# ─────────────────────────────────────────────────────────────────────

def test_each_flip_has_distinct_outcomes():
    """Every emitted flip must have original_outcome != modified_outcome."""
    reps = _run(run_all_counterfactuals())
    for rep in reps:
        for flip in rep.flips_found:
            assert flip.original_outcome != flip.modified_outcome, (
                f"{rep.scenario_id}: flip {flip.factor_name!r} same outcome "
                f"on both sides"
            )


def test_each_flip_has_positive_distance():
    reps = _run(run_all_counterfactuals())
    for rep in reps:
        for flip in rep.flips_found:
            assert flip.flip_distance > 0
