"""Phase 14.15 Q2 -- Planner revision feedback loop tests."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.planner_revision import (
    critique_plan, revise_plan, run_with_planner_revision,
)
from shared.schemas import ToolUsePlan, ToolUsePlanStep


def _run(coro):
    return asyncio.run(coro)


def _step(step_id: str, tool: str, *,
          bundle: str = "core_discharge", mandatory: bool = False,
          specialist: str = "trustedrisk-discharge") -> ToolUsePlanStep:
    return ToolUsePlanStep(
        step_id=step_id, specialist=specialist, bundle=bundle,
        tool=tool, args={}, mandatory=mandatory,
        rationale="test step",
    )


def _plan(user_query: str, steps: list[ToolUsePlanStep]) -> ToolUsePlan:
    return ToolUsePlan(
        user_query=user_query,
        steps=steps,
        n_steps=len(steps),
        confidence=0.7,
        safety_floor_engaged=any(s.mandatory for s in steps),
        llm_router_used=False,
        llm_model_id=None,
        rationale="test plan",
        references=[],
    )


# ─────────────────────────────────────────────────────────────────────
# Critic -- verdict cases
# ─────────────────────────────────────────────────────────────────────

def test_clean_plan_is_approved():
    plan = _plan(
        "stroke nihss assessment",
        [_step("floor-int-001", "compute_stroke_severity",
               bundle="stroke_acs")],
    )
    crit = critique_plan(plan)
    assert crit.verdict == "approved"
    assert crit.issues == []


def test_empty_plan_force_abstain():
    plan = _plan("ambiguous query", [])
    crit = critique_plan(plan)
    assert crit.verdict == "force_abstain"
    assert any(i.code == "no_steps_planned" for i in crit.issues)


def test_phi_query_without_scrub_revises():
    plan = _plan(
        "Look up MRN 12345 for the patient with chest pain",
        [_step("floor-int-001", "compute_heart_score",
               bundle="stroke_acs")],
    )
    crit = critique_plan(plan)
    assert crit.verdict == "revise"
    assert any(
        i.code == "missing_phi_scrub_for_phi_query" for i in crit.issues
    )


def test_duplicate_tool_revises():
    plan = _plan(
        "compute risk twice",
        [
            _step("floor-int-001", "compute_readmission_risk"),
            _step("floor-int-002", "compute_readmission_risk"),
        ],
    )
    crit = critique_plan(plan)
    assert crit.verdict == "revise"
    issue = next(
        i for i in crit.issues if i.code == "duplicate_tool_in_plan"
    )
    assert issue.affected_step_ids == ["floor-int-002"]


def test_orphan_bundle_revises():
    plan = _plan(
        "test bundle drift",
        [_step("floor-int-001", "compute_readmission_risk",
               bundle="not_a_real_bundle_xyz")],
    )
    crit = critique_plan(plan)
    assert any(
        i.code == "orphan_bundle_for_specialist" for i in crit.issues
    )


def test_redundant_tfidf_after_intent_revises():
    plan = _plan(
        "stroke",
        [
            _step("floor-int-001", "compute_stroke_severity",
                  bundle="stroke_acs"),
            _step("floor-tfidf-002", "compute_heart_score",
                  bundle="stroke_acs"),
        ],
    )
    crit = critique_plan(plan)
    assert any(
        i.code == "redundant_optional_step_after_intent"
        for i in crit.issues
    )


# ─────────────────────────────────────────────────────────────────────
# Revisor -- applies critique
# ─────────────────────────────────────────────────────────────────────

def test_revise_drops_duplicate_steps():
    plan = _plan(
        "x",
        [
            _step("floor-int-001", "compute_readmission_risk"),
            _step("floor-int-002", "compute_readmission_risk"),
        ],
    )
    revised = revise_plan(plan, critique_plan(plan))
    assert revised.n_steps == 1
    assert revised.steps[0].step_id == "floor-int-001"


def test_revise_injects_phi_scrub():
    plan = _plan(
        "Look up MRN 12345",
        [_step("floor-int-001", "compute_readmission_risk")],
    )
    revised = revise_plan(plan, critique_plan(plan))
    assert revised.steps[0].tool == "detect_phi"
    assert revised.steps[0].mandatory is True


def test_revise_drops_orphan_bundle_steps():
    plan = _plan(
        "x",
        [
            _step("floor-int-001", "compute_readmission_risk",
                  bundle="core_discharge"),
            _step("floor-int-002", "compute_decision_utility",
                  bundle="not_a_real_bundle_xyz"),
        ],
    )
    revised = revise_plan(plan, critique_plan(plan))
    tools = [s.tool for s in revised.steps]
    assert "compute_readmission_risk" in tools
    assert "compute_decision_utility" not in tools


def test_revise_drops_redundant_tfidf_steps():
    plan = _plan(
        "x",
        [
            _step("floor-int-001", "compute_stroke_severity",
                  bundle="stroke_acs"),
            _step("floor-tfidf-002", "compute_heart_score",
                  bundle="stroke_acs"),
        ],
    )
    revised = revise_plan(plan, critique_plan(plan))
    assert all(
        not s.step_id.startswith("floor-tfidf-") for s in revised.steps
    )


def test_revise_pass_through_when_approved():
    plan = _plan(
        "stroke nihss",
        [_step("floor-int-001", "compute_stroke_severity",
               bundle="stroke_acs")],
    )
    crit = critique_plan(plan)
    assert crit.verdict == "approved"
    revised = revise_plan(plan, crit)
    assert revised == plan


# ─────────────────────────────────────────────────────────────────────
# Bounded loop
# ─────────────────────────────────────────────────────────────────────

def test_loop_approves_clean_plan_in_one_iteration():
    async def plan_fn(q: str) -> ToolUsePlan:
        return _plan(q, [_step("floor-int-001", "compute_stroke_severity",
                                bundle="stroke_acs")])
    trace = _run(run_with_planner_revision(
        plan_fn, "stroke nihss", max_iterations=2,
    ))
    assert trace.terminated_reason == "approved"
    assert trace.iterations_used == 0
    assert len(trace.attempts) == 1


def test_loop_revises_then_approves():
    async def plan_fn(q: str) -> ToolUsePlan:
        # Plan with duplicate tool -- needs one revision
        return _plan(q, [
            _step("floor-int-001", "compute_readmission_risk"),
            _step("floor-int-002", "compute_readmission_risk"),
        ])
    trace = _run(run_with_planner_revision(
        plan_fn, "discharge readmission", max_iterations=2,
    ))
    assert trace.terminated_reason == "approved"
    assert trace.iterations_used == 1
    assert trace.final_plan.n_steps == 1


def test_loop_force_abstains_on_empty_plan():
    async def plan_fn(q: str) -> ToolUsePlan:
        return _plan(q, [])
    trace = _run(run_with_planner_revision(
        plan_fn, "ambiguous query", max_iterations=3,
    ))
    assert trace.terminated_reason == "force_abstain"
    assert trace.iterations_used == 0


def test_loop_records_every_attempt():
    async def plan_fn(q: str) -> ToolUsePlan:
        return _plan(q, [
            _step("floor-int-001", "compute_readmission_risk"),
            _step("floor-int-002", "compute_readmission_risk"),
        ])
    trace = _run(run_with_planner_revision(
        plan_fn, "x", max_iterations=2,
    ))
    iterations = [a.iteration for a in trace.attempts]
    assert iterations == [0, 1]


def test_loop_rejects_negative_budget():
    async def plan_fn(q: str) -> ToolUsePlan:
        return _plan(q, [])
    with pytest.raises(ValueError):
        _run(run_with_planner_revision(
            plan_fn, "x", max_iterations=-1,
        ))


def test_loop_zero_budget_returns_initial_plan_unrevised():
    async def plan_fn(q: str) -> ToolUsePlan:
        return _plan(q, [
            _step("floor-int-001", "compute_readmission_risk"),
            _step("floor-int-002", "compute_readmission_risk"),
        ])
    trace = _run(run_with_planner_revision(
        plan_fn, "x", max_iterations=0,
    ))
    # Budget exhausted before any revision
    assert trace.terminated_reason == "budget_exhausted"
    assert trace.iterations_used == 0


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────

def test_revise_is_deterministic():
    plan = _plan(
        "Look up MRN 12345",
        [_step("floor-int-001", "compute_readmission_risk")],
    )
    a = revise_plan(plan, critique_plan(plan))
    b = revise_plan(plan, critique_plan(plan))
    assert a.model_dump() == b.model_dump()
