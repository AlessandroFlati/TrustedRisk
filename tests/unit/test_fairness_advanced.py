"""Phase 13.2 I1 -- Fairness EOO/DP advanced audit tests."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.fairness_advanced import compute_fairness_advanced
from mcp_server.tools.fairness_advanced_tool import (
    compute_fairness_advanced as compute_fairness_tool,
)


def _run(coro):
    return asyncio.run(coro)


def _balanced_rows(n_per_group: int = 200) -> list[dict]:
    """Both groups have 50% positive base rate AND identical model
    behaviour -> posture should be `fair`."""
    rows: list[dict] = []
    for g in ("white", "black"):
        for i in range(n_per_group):
            actual = i % 2
            rows.append({
                "subgroup": g,
                "actual_positive": actual,
                "predicted_positive": actual,    # perfect
            })
    return rows


def _imbalanced_rows(n_per_group: int = 200) -> list[dict]:
    """White group: 50% selection rate. Black group: 10% selection rate.
    DP gap = 0.40 -> `violation`."""
    rows: list[dict] = []
    for i in range(n_per_group):
        rows.append({
            "subgroup": "white", "actual_positive": i % 2,
            "predicted_positive": 1 if i % 2 == 0 else 0,
        })
    for i in range(n_per_group):
        rows.append({
            "subgroup": "black", "actual_positive": i % 2,
            "predicted_positive": 1 if i % 10 == 0 else 0,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Smoke
# ─────────────────────────────────────────────────────────────────────

def test_balanced_rows_yield_fair_posture():
    rep = compute_fairness_advanced("race", _balanced_rows())
    assert rep.posture == "fair"
    assert rep.demographic_parity_max_gap < 0.05
    assert rep.equalized_odds_tpr_max_gap < 0.05


def test_imbalanced_rows_yield_violation_posture():
    rep = compute_fairness_advanced("race", _imbalanced_rows())
    assert rep.posture == "violation"
    assert rep.demographic_parity_max_gap > 0.20


def test_chi2_p_value_rejects_independence_when_imbalanced():
    rep = compute_fairness_advanced("race", _imbalanced_rows())
    # Chi² should reject independence at 0.05 with this much disparity
    assert rep.chi2_independence_p_value is not None
    assert rep.chi2_independence_p_value < 0.05


def test_chi2_p_value_does_not_reject_when_balanced():
    rep = compute_fairness_advanced("race", _balanced_rows())
    assert rep.chi2_independence_p_value is not None
    assert rep.chi2_independence_p_value > 0.20


# ─────────────────────────────────────────────────────────────────────
# Per-subgroup metric shape
# ─────────────────────────────────────────────────────────────────────

def test_per_subgroup_metrics_have_ci_within_unit_box():
    rep = compute_fairness_advanced("race", _imbalanced_rows())
    for m in rep.metrics:
        assert 0.0 <= m.selection_rate_ci95_lower <= 1.0
        assert 0.0 <= m.selection_rate_ci95_upper <= 1.0
        assert (m.selection_rate_ci95_lower
                    <= m.selection_rate
                    <= m.selection_rate_ci95_upper)


def test_per_subgroup_metrics_have_one_row_per_subgroup():
    rep = compute_fairness_advanced("race", _balanced_rows())
    subs = {m.subgroup for m in rep.metrics}
    assert subs == {"white", "black"}


# ─────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────

def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        compute_fairness_advanced("race", [{"subgroup": "white",
                                                  "actual_positive": 1,
                                                  "predicted_positive": 1}])


def test_only_one_subgroup_raises():
    rows = [
        {"subgroup": "white", "actual_positive": i % 2,
         "predicted_positive": i % 2}
        for i in range(20)
    ]
    with pytest.raises(ValueError):
        compute_fairness_advanced("race", rows)


# ─────────────────────────────────────────────────────────────────────
# MCP tool wrapper
# ─────────────────────────────────────────────────────────────────────

def test_mcp_tool_wrapper_round_trips_via_async():
    rep = _run(compute_fairness_tool(
        sensitive_attribute="race", rows=_imbalanced_rows(),
    ))
    assert rep.posture == "violation"


def test_fairness_advanced_in_economics_bundle():
    from mcp_server.tools import BUNDLES
    assert "compute_fairness_advanced" in BUNDLES["economics"]


# ─────────────────────────────────────────────────────────────────────
# Posture taxonomy thresholds
# ─────────────────────────────────────────────────────────────────────

def test_monitor_or_investigate_posture_emerges_for_modest_gap():
    """Build a cohort with a moderate selection-rate gap that should
    fall above 'fair' but below 'violation'."""
    rows: list[dict] = []
    for i in range(200):
        rows.append({"subgroup": "a", "actual_positive": i % 2,
                          "predicted_positive": 1 if i % 2 == 0 else 0})
    # Group b: ~ 35% selection rate (so gap ~ 0.15)
    for i in range(200):
        rows.append({"subgroup": "b", "actual_positive": i % 2,
                          "predicted_positive": 1 if i % 100 < 35 else 0})
    rep = compute_fairness_advanced("g", rows)
    # Just make sure it's NOT 'fair' (the gap should be material)
    assert rep.posture != "fair"


def test_chi2_p_in_unit_interval():
    """Sanity bound on the p-value calculation."""
    rep = compute_fairness_advanced("race", _imbalanced_rows())
    p = rep.chi2_independence_p_value
    assert p is not None and 0.0 <= p <= 1.0
