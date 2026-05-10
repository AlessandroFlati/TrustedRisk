"""Phase 17.AH - Monte Carlo hospital-year simulator tests."""

from __future__ import annotations

import pytest

from a2a_agent.hospital_year_simulator import (
    HospitalYearReport, _percentile, _summarize,
    simulate_hospital_year,
)


# ─────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────

def test_percentile_returns_lower_bound_for_q_zero():
    assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0


def test_percentile_returns_upper_bound_for_q_one():
    assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0


def test_percentile_returns_median_for_q_half():
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_summarize_yields_zero_on_empty_list():
    s = _summarize([])
    assert s.point_estimate == 0.0
    assert s.ci95_lower == 0.0


def test_summarize_yields_correct_mean_and_ci_widths():
    s = _summarize([1.0, 2.0, 3.0, 4.0, 5.0,
                    6.0, 7.0, 8.0, 9.0, 10.0])
    assert abs(s.point_estimate - 5.5) < 1e-6
    # CI95 must enclose the median
    assert s.ci95_lower <= s.median <= s.ci95_upper


# ─────────────────────────────────────────────────────────────────────
# Hospital-year simulator
# ─────────────────────────────────────────────────────────────────────

def test_simulator_runs_with_small_grid():
    rep = simulate_hospital_year(
        n_trajectories=20, n_per_trajectory=200,
    )
    assert rep.n_trajectories == 20
    assert rep.n_per_trajectory == 200


def test_simulator_yields_non_negative_averted_count():
    rep = simulate_hospital_year(
        n_trajectories=10, n_per_trajectory=200,
    )
    assert rep.readmissions_averted.point_estimate >= 0
    assert rep.readmissions_averted.ci95_lower >= 0


def test_simulator_ci_lower_at_or_below_upper():
    rep = simulate_hospital_year(
        n_trajectories=10, n_per_trajectory=200,
    )
    for stat in (
        rep.readmissions_averted, rep.cost_saved_usd,
        rep.qalys_gained, rep.abstain_rate,
    ):
        assert stat.ci95_lower <= stat.ci95_upper


def test_simulator_rejects_zero_n():
    with pytest.raises(ValueError):
        simulate_hospital_year(
            n_trajectories=0, n_per_trajectory=10,
        )


def test_simulator_round_trip_through_pydantic():
    rep = simulate_hospital_year(
        n_trajectories=5, n_per_trajectory=100,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = HospitalYearReport.model_validate(payload)
    assert rebuilt.n_trajectories == rep.n_trajectories


def test_simulator_is_deterministic_for_seed_grid():
    a = simulate_hospital_year(
        n_trajectories=5, n_per_trajectory=100, base_seed=99,
    )
    b = simulate_hospital_year(
        n_trajectories=5, n_per_trajectory=100, base_seed=99,
    )
    assert (
        a.cost_saved_usd.point_estimate
        == b.cost_saved_usd.point_estimate
    )


def test_simulator_iqr_inside_ci():
    rep = simulate_hospital_year(
        n_trajectories=15, n_per_trajectory=200,
    )
    for stat in (
        rep.readmissions_averted, rep.cost_saved_usd,
        rep.qalys_gained,
    ):
        assert stat.ci95_lower <= stat.iqr[0] <= stat.iqr[1]
        assert stat.iqr[1] <= stat.ci95_upper
