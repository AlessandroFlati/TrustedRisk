"""Phase 17.AA - Cost simulator tests."""

from __future__ import annotations

import pytest

from a2a_agent.cost_simulator import (
    CostSimulationReport, _expected_outcome,
    simulate_cost_impact,
)


def test_expected_outcome_lower_for_more_intensive_action():
    base = 0.30
    home = _expected_outcome("discharge_home", base)
    homecare = _expected_outcome("discharge_with_homecare", base)
    inpatient = _expected_outcome("continued_admission", base)
    assert home > homecare > inpatient


def test_expected_outcome_clipped_to_unit():
    out = _expected_outcome("discharge_home", 1.5)
    assert out <= 1.0


def test_simulator_yields_overall_with_n_matching_cohort():
    rep = simulate_cost_impact(n_encounters=500, seed=1)
    assert rep.cohort_n == 500
    assert rep.overall.n == 500


def test_simulator_emits_per_payer_buckets():
    rep = simulate_cost_impact(n_encounters=2000, seed=2)
    assert "medicare" in rep.by_payer
    assert "medicaid" in rep.by_payer
    assert "commercial" in rep.by_payer
    total_n = sum(b.n for b in rep.by_payer.values())
    assert total_n == rep.cohort_n


def test_simulator_yields_non_negative_readmissions_averted():
    rep = simulate_cost_impact(n_encounters=2000, seed=3)
    assert rep.overall.readmissions_averted_vs_baseline >= 0


def test_simulator_cost_saved_scales_with_averted_count():
    rep = simulate_cost_impact(n_encounters=2000, seed=4)
    if rep.overall.readmissions_averted_vs_baseline > 0:
        assert rep.overall.cost_saved_usd > 0


def test_simulator_qalys_gained_scales_with_averted():
    rep = simulate_cost_impact(n_encounters=2000, seed=5)
    if rep.overall.readmissions_averted_vs_baseline > 0:
        assert rep.overall.qalys_gained > 0


def test_simulator_rejects_zero_n():
    with pytest.raises(ValueError):
        simulate_cost_impact(n_encounters=0)


def test_simulator_round_trip_through_pydantic():
    rep = simulate_cost_impact(n_encounters=500, seed=6)
    payload = rep.model_dump(mode="json")
    rebuilt = CostSimulationReport.model_validate(payload)
    assert rebuilt.cohort_n == rep.cohort_n


def test_simulator_is_deterministic_for_seed():
    a = simulate_cost_impact(n_encounters=400, seed=7)
    b = simulate_cost_impact(n_encounters=400, seed=7)
    assert (
        a.overall.cost_saved_usd
        == b.overall.cost_saved_usd
    )
    assert (
        a.overall.readmissions_averted_vs_baseline
        == b.overall.readmissions_averted_vs_baseline
    )
