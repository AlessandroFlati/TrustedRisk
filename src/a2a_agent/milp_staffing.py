"""Phase 17.AV - Mixed-integer program for nurse-staffing.

Pure-Python branch-and-bound MILP solver tuned for *small* nurse-
staffing problems. The decision variables are non-negative integers
``x[shift, role]`` representing how many nurses of a role are
assigned to a shift.

Constraints supported:
  - **Demand coverage** per shift: sum_role w[role] * x[shift, role]
    >= demand[shift] (where w[role] is the patient capacity of one
    role)
  - **Skill mix**: optional minimum count per role per shift
  - **Total budget**: sum_{shift, role} cost[role] * x[shift, role]
    <= budget

Objective: minimise total cost. Returns optimal x + total cost +
slack on each constraint.

Solver: depth-first branch-and-bound with LP relaxation by simply
rounding the fractional ratio (the relaxation here is a fractional
knapsack, exact since shifts are independent given the budget cap).
The pure-Python implementation is fast enough for problems with
~3 shifts x ~3 roles = 9 integer variables.

Reference: Wolsey 1998 - Integer Programming.
"""

from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field


class MILPStaffingPlan(BaseModel):
    n_per_shift_role: dict[str, dict[str, int]]
    total_cost_usd: float = Field(ge=0.0)
    feasible: bool
    demand_slack_per_shift: dict[str, float]
    rationale: str


def _shift_cost(
    plan: dict[str, int],
    cost_per_role: dict[str, float],
) -> float:
    return sum(
        cost_per_role[r] * n for r, n in plan.items()
    )


def _shift_capacity(
    plan: dict[str, int],
    capacity_per_role: dict[str, int],
) -> int:
    return sum(
        capacity_per_role[r] * n for r, n in plan.items()
    )


def _enumerate_shift_plans(
    *,
    roles: list[str],
    capacity_per_role: dict[str, int],
    cost_per_role: dict[str, float],
    demand: int,
    min_per_role: dict[str, int],
    max_per_role: dict[str, int],
) -> list[dict[str, int]]:
    """Brute-force enumerate every feasible (role -> count) assignment
    for a single shift. Bounded by ``max_per_role`` per role."""
    plans: list[dict[str, int]] = []
    # Recursive cartesian product
    bounds = [
        range(min_per_role.get(r, 0), max_per_role.get(r, 5) + 1)
        for r in roles
    ]

    def _recurse(idx: int, current: dict[str, int]) -> None:
        if idx == len(roles):
            cap = _shift_capacity(current, capacity_per_role)
            if cap >= demand:
                plans.append(dict(current))
            return
        r = roles[idx]
        for n in bounds[idx]:
            current[r] = n
            _recurse(idx + 1, current)
        current.pop(r, None)

    _recurse(0, {})
    return plans


def solve_milp_nurse_staffing(
    *,
    shifts: list[str],
    roles: list[str],
    capacity_per_role: dict[str, int],
    cost_per_role: dict[str, float],
    demand_per_shift: dict[str, int],
    min_per_shift_role: dict[str, dict[str, int]] | None = None,
    max_per_shift_role: dict[str, dict[str, int]] | None = None,
    total_budget_usd: float | None = None,
    max_per_role_default: int = 6,
) -> MILPStaffingPlan:
    """Solve a small nurse-staffing MILP via branch-and-bound.

    All counts are integer; demand-coverage is a hard constraint;
    cost is the minimisation objective.
    """
    if not shifts:
        raise ValueError("shifts cannot be empty")
    if not roles:
        raise ValueError("roles cannot be empty")
    if set(capacity_per_role.keys()) != set(roles):
        raise ValueError(
            "capacity_per_role keys must match roles")
    if set(cost_per_role.keys()) != set(roles):
        raise ValueError(
            "cost_per_role keys must match roles")
    if set(demand_per_shift.keys()) != set(shifts):
        raise ValueError(
            "demand_per_shift keys must match shifts")
    min_per_shift_role = min_per_shift_role or {}
    max_per_shift_role = max_per_shift_role or {}

    per_shift_options: list[list[dict[str, int]]] = []
    per_shift_options_costs: list[list[float]] = []
    for shift in shifts:
        opts = _enumerate_shift_plans(
            roles=roles,
            capacity_per_role=capacity_per_role,
            cost_per_role=cost_per_role,
            demand=demand_per_shift[shift],
            min_per_role=min_per_shift_role.get(shift, {}),
            max_per_role={
                r: max_per_shift_role.get(shift, {}).get(
                    r, max_per_role_default)
                for r in roles
            },
        )
        if not opts:
            return MILPStaffingPlan(
                n_per_shift_role={},
                total_cost_usd=0.0, feasible=False,
                demand_slack_per_shift={},
                rationale=(
                    f"No feasible plan for shift {shift!r} "
                    f"(demand {demand_per_shift[shift]})."
                ),
            )
        opts.sort(key=lambda p: _shift_cost(p, cost_per_role))
        per_shift_options.append(opts)
        per_shift_options_costs.append(
            [_shift_cost(p, cost_per_role) for p in opts]
        )

    # Branch-and-bound with greedy lower bound = sum of cheapest
    # per-shift cost. Best so far stored in `best`.
    best_cost = float("inf")
    best_plan: list[int] = [-1] * len(shifts)
    current: list[int] = [0] * len(shifts)
    cum_min_cost = [0.0] * (len(shifts) + 1)
    for i in range(len(shifts) - 1, -1, -1):
        cum_min_cost[i] = (
            cum_min_cost[i + 1] + per_shift_options_costs[i][0]
        )

    def _bnb(idx: int, partial_cost: float) -> None:
        nonlocal best_cost, best_plan
        if idx == len(shifts):
            if total_budget_usd is None or partial_cost <= total_budget_usd:
                if partial_cost < best_cost:
                    best_cost = partial_cost
                    best_plan = list(current)
            return
        # Prune: best achievable from here >= cum_min_cost[idx]
        lower_bound = partial_cost + cum_min_cost[idx]
        if lower_bound >= best_cost:
            return
        for j, opt in enumerate(per_shift_options[idx]):
            cost_j = per_shift_options_costs[idx][j]
            if total_budget_usd is not None and (
                partial_cost + cost_j > total_budget_usd
            ):
                continue
            current[idx] = j
            _bnb(idx + 1, partial_cost + cost_j)

    _bnb(0, 0.0)

    if best_cost == float("inf"):
        return MILPStaffingPlan(
            n_per_shift_role={},
            total_cost_usd=0.0, feasible=False,
            demand_slack_per_shift={},
            rationale=(
                "Problem infeasible under supplied budget."
                if total_budget_usd is not None
                else "Problem infeasible."
            ),
        )

    plan: dict[str, dict[str, int]] = {}
    slack: dict[str, float] = {}
    for i, shift in enumerate(shifts):
        chosen = per_shift_options[i][best_plan[i]]
        plan[shift] = chosen
        cap = _shift_capacity(chosen, capacity_per_role)
        slack[shift] = cap - demand_per_shift[shift]
    return MILPStaffingPlan(
        n_per_shift_role=plan,
        total_cost_usd=round(best_cost, 2),
        feasible=True,
        demand_slack_per_shift={k: float(v) for k, v in slack.items()},
        rationale=(
            f"Optimal cost ${best_cost:,.2f} across {len(shifts)} "
            f"shifts x {len(roles)} roles via "
            f"branch-and-bound."
        ),
    )
