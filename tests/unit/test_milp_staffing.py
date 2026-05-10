"""Phase 17.AV - MILP nurse-staffing tests."""

from __future__ import annotations

import pytest

from a2a_agent.milp_staffing import (
    MILPStaffingPlan, solve_milp_nurse_staffing,
)


_SHIFTS = ["day", "evening", "night"]
_ROLES = ["RN", "LPN", "tech"]
_CAP = {"RN": 5, "LPN": 4, "tech": 3}
_COST = {"RN": 600.0, "LPN": 380.0, "tech": 220.0}


def test_solver_finds_feasible_plan_for_simple_demand():
    plan = solve_milp_nurse_staffing(
        shifts=_SHIFTS, roles=_ROLES,
        capacity_per_role=_CAP, cost_per_role=_COST,
        demand_per_shift={"day": 12, "evening": 9, "night": 6},
    )
    assert plan.feasible
    assert plan.total_cost_usd > 0


def test_solver_meets_demand_per_shift():
    plan = solve_milp_nurse_staffing(
        shifts=_SHIFTS, roles=_ROLES,
        capacity_per_role=_CAP, cost_per_role=_COST,
        demand_per_shift={"day": 10, "evening": 8, "night": 5},
    )
    assert plan.feasible
    for shift, slack in plan.demand_slack_per_shift.items():
        assert slack >= 0


def test_solver_respects_minimum_RN_constraint():
    plan = solve_milp_nurse_staffing(
        shifts=_SHIFTS, roles=_ROLES,
        capacity_per_role=_CAP, cost_per_role=_COST,
        demand_per_shift={"day": 10, "evening": 8, "night": 5},
        min_per_shift_role={"day": {"RN": 2}},
    )
    assert plan.feasible
    assert plan.n_per_shift_role["day"]["RN"] >= 2


def test_solver_marks_infeasible_when_budget_too_tight():
    plan = solve_milp_nurse_staffing(
        shifts=_SHIFTS, roles=_ROLES,
        capacity_per_role=_CAP, cost_per_role=_COST,
        demand_per_shift={"day": 10, "evening": 8, "night": 5},
        total_budget_usd=100.0,
    )
    assert plan.feasible is False


def test_solver_picks_lower_cost_when_two_plans_meet_demand():
    """Demand=4, RN(cap 5, $600) vs LPN(cap 4, $380). Should pick
    LPN."""
    plan = solve_milp_nurse_staffing(
        shifts=["day"], roles=_ROLES,
        capacity_per_role=_CAP, cost_per_role=_COST,
        demand_per_shift={"day": 4},
    )
    assert plan.feasible
    chosen = plan.n_per_shift_role["day"]
    assert chosen["LPN"] >= 1 or chosen["tech"] >= 2


def test_solver_rejects_empty_shifts():
    with pytest.raises(ValueError):
        solve_milp_nurse_staffing(
            shifts=[], roles=_ROLES,
            capacity_per_role=_CAP, cost_per_role=_COST,
            demand_per_shift={},
        )


def test_solver_rejects_role_set_mismatch():
    with pytest.raises(ValueError):
        solve_milp_nurse_staffing(
            shifts=_SHIFTS, roles=_ROLES,
            capacity_per_role={"RN": 5},  # missing LPN, tech
            cost_per_role=_COST,
            demand_per_shift={
                "day": 10, "evening": 8, "night": 5,
            },
        )


def test_solver_rejects_demand_set_mismatch():
    with pytest.raises(ValueError):
        solve_milp_nurse_staffing(
            shifts=_SHIFTS, roles=_ROLES,
            capacity_per_role=_CAP, cost_per_role=_COST,
            demand_per_shift={"day": 10},   # missing other shifts
        )


def test_solver_round_trip_through_pydantic():
    plan = solve_milp_nurse_staffing(
        shifts=_SHIFTS, roles=_ROLES,
        capacity_per_role=_CAP, cost_per_role=_COST,
        demand_per_shift={"day": 10, "evening": 8, "night": 5},
    )
    payload = plan.model_dump(mode="json")
    rebuilt = MILPStaffingPlan.model_validate(payload)
    assert rebuilt.total_cost_usd == plan.total_cost_usd


def test_solver_returns_feasible_false_for_too_high_demand():
    plan = solve_milp_nurse_staffing(
        shifts=["day"], roles=_ROLES,
        capacity_per_role=_CAP, cost_per_role=_COST,
        demand_per_shift={"day": 1000},   # impossible w/ default cap
        max_per_role_default=2,           # tight upper bound
    )
    assert plan.feasible is False
