"""Phase 11.6 -- clinical scenario smoke tests.

Validates that all 5 end-to-end scenarios run, chain at least 5
deterministic-floor tool calls, and emit a non-empty final summary.
No FHIR, no network -- every scenario provides its inputs inline.
"""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.clinical_scenarios import (
    list_scenarios,
    run_all_scenarios,
    run_scenario,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Per-scenario smoke
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario_id", [
    "acute_stroke_lvo",
    "sepsis_bundle",
    "polytrauma_mtp",
    "geriatric_polypharmacy",
    "mental_health_crisis",
])
def test_scenario_runs_and_chains_tools(scenario_id: str):
    run = _run(run_scenario(scenario_id))
    assert run.scenario_id == scenario_id
    assert run.n_tools_invoked >= 5, (
        f"{scenario_id}: expected ≥ 5 tool calls, got "
        f"{run.n_tools_invoked}"
    )
    assert run.final_summary.strip(), (
        f"{scenario_id}: empty final summary"
    )
    # Every step has a non-empty tool name + summary
    for step in run.steps:
        assert step.tool
        assert step.summary


# ─────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────

def test_run_all_returns_five_scenarios():
    runs = _run(run_all_scenarios())
    ids = {r.scenario_id for r in runs}
    assert ids == {
        "acute_stroke_lvo",
        "sepsis_bundle",
        "polytrauma_mtp",
        "geriatric_polypharmacy",
        "mental_health_crisis",
    }


def test_aggregate_tool_calls_at_least_25():
    """Floor for the cumulative tool surface across the 5 scenarios."""
    runs = _run(run_all_scenarios())
    total = sum(r.n_tools_invoked for r in runs)
    assert total >= 25, (
        f"Cumulative tool calls {total} < 25 -- scenarios are too thin"
    )


def test_unknown_scenario_id_raises():
    with pytest.raises(KeyError):
        _run(run_scenario("nonexistent"))


def test_list_scenarios_matches_run_all():
    ids_listed = set(list_scenarios())
    runs = _run(run_all_scenarios())
    ids_run = {r.scenario_id for r in runs}
    assert ids_listed == ids_run
