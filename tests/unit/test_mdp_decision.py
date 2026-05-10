"""SCALE-3 unit tests for the sequential MDP decision modeler."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent.mdp_decision import (
    _transition_matrix,
    compute_sequential_mdp_value,
)


# ─────────────────────── Validation ───────────────────────

def test_missing_baseline_and_risk_raises():
    with pytest.raises(ValueError, match="baseline"):
        compute_sequential_mdp_value()


def test_baseline_out_of_range_raises():
    with pytest.raises(ValueError, match="baseline"):
        compute_sequential_mdp_value(baseline_readmission_30d_prob=1.5)


def test_horizon_out_of_range_raises():
    with pytest.raises(ValueError, match="horizon"):
        compute_sequential_mdp_value(
            baseline_readmission_30d_prob=0.20, horizon_days=400)


def test_invalid_discount_raises():
    with pytest.raises(ValueError, match="discount"):
        compute_sequential_mdp_value(
            baseline_readmission_30d_prob=0.20, discount_factor=1.5)


def test_no_supported_actions_raises():
    with pytest.raises(ValueError, match="actions"):
        compute_sequential_mdp_value(
            baseline_readmission_30d_prob=0.20,
            actions=["fly_to_mars"])


# ─────────────────────── Transition matrix invariants ───────────────────────

def test_transition_matrix_rows_sum_to_one():
    T = _transition_matrix(0.30, "discharge_home")
    row_sums = T.sum(axis=1)
    assert all(abs(s - 1.0) < 1e-9 for s in row_sums)


def test_transition_matrix_no_negative_entries():
    T = _transition_matrix(0.30, "discharge_home")
    assert (T >= 0).all()


def test_deceased_state_is_absorbing():
    T = _transition_matrix(0.30, "snf")
    # State 3 = deceased; row sums to 1 with self-transition
    assert T[3, 3] == pytest.approx(1.0)
    assert T[3, 0] == pytest.approx(0.0)


def test_higher_rrr_lowers_p_readmission():
    T_home = _transition_matrix(0.30, "discharge_home")
    T_snf = _transition_matrix(0.30, "snf")
    # Row 0 (home_well), column 2 (readmitted)
    assert T_snf[0, 2] < T_home[0, 2]


# ─────────────────────── Action values + ranking ───────────────────────

def test_high_risk_prefers_aggressive_action():
    """A high-baseline-risk patient should rank continued_admission/SNF
    above discharge_home on expected value."""
    r = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.55, horizon_days=90,
    )
    # Optimal action shouldn't be discharge_home for very high risk
    assert r.optimal_action != "discharge_home"


def test_low_risk_prefers_discharge_home():
    """Very low-risk patients shouldn't waste resources on continued_admission."""
    r = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.05, horizon_days=90,
    )
    # discharge_home should win OR be tied -- accept either of the
    # 2 lightest actions
    assert r.optimal_action in ("discharge_home", "home_with_care")


def test_action_values_complete_set():
    r = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.20)
    actions = {av.action for av in r.action_values}
    assert actions == {"discharge_home", "home_with_care",
                          "snf", "continued_admission"}


def test_action_values_have_well_formed_metrics():
    r = compute_sequential_mdp_value(baseline_readmission_30d_prob=0.20)
    for av in r.action_values:
        assert 0.0 <= av.probability_alive_at_horizon <= 1.0
        assert 0.0 <= av.probability_readmitted_in_window <= 1.0
        assert av.expected_qaly_in_window >= 0.0


# ─────────────────────── Horizon scaling ───────────────────────

def test_longer_horizon_increases_readmission_chance():
    r30 = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.20, horizon_days=30)
    r90 = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.20, horizon_days=90)
    home30 = next(av for av in r30.action_values
                     if av.action == "discharge_home")
    home90 = next(av for av in r90.action_values
                     if av.action == "discharge_home")
    assert home90.probability_readmitted_in_window >= \
           home30.probability_readmitted_in_window


def test_n_stages_scales_with_horizon():
    r30 = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.20, horizon_days=30)
    r60 = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.20, horizon_days=60)
    assert r30.n_stages == 1
    assert r60.n_stages == 2


# ─────────────────────── Risk estimate input path ───────────────────────

def test_risk_estimate_dict_accepted():
    r = compute_sequential_mdp_value(
        risk_estimate={"probability_mean": 0.40})
    assert r.baseline_readmission_30d_prob == pytest.approx(0.40)


def test_risk_estimate_object_accepted():
    from datetime import datetime, timezone
    from shared.schemas import Factor, RiskEstimate
    risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="0.7.0",
        outcome_id="readmission_30d",
        horizon_days=30,
        lace_raw_score=12,
        probability_mean=0.35,
        probability_ci95=(0.30, 0.40),
        probability_ci_width=0.10,
        contributing_factors=[
            Factor(name="LACE_length_of_stay", raw_value=6.0,
                     lace_points=2, weight=0.25),
            Factor(name="LACE_acuity", raw_value=1.0,
                     lace_points=3, weight=0.30),
            Factor(name="LACE_comorbidity", raw_value=4.0,
                     lace_points=3, weight=0.30),
            Factor(name="LACE_ed_visits_6mo", raw_value=2.0,
                     lace_points=4, weight=0.15),
        ],
        fhir_observations_used=[],
        computed_at=datetime.now(timezone.utc),
        confidence="preferred",
        valid_for_minutes=60,
        valid_until=datetime.now(timezone.utc),
    )
    r = compute_sequential_mdp_value(risk_estimate=risk)
    assert r.baseline_readmission_30d_prob == pytest.approx(0.35)


# ─────────────────────── Initial-state path ───────────────────────

def test_home_sick_initial_state_lowers_alive_probability():
    """Starting from home_sick (already morbid) should yield lower p_alive."""
    well = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.20, initial_state="home_well")
    sick = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.20, initial_state="home_sick")
    well_home = next(av for av in well.action_values
                        if av.action == "discharge_home")
    sick_home = next(av for av in sick.action_values
                        if av.action == "discharge_home")
    assert sick_home.probability_alive_at_horizon <= \
           well_home.probability_alive_at_horizon


# ─────────────────────── Output shape + references ───────────────────────

def test_rationale_includes_optimal_action():
    r = compute_sequential_mdp_value(baseline_readmission_30d_prob=0.20)
    assert r.optimal_action in r.rationale


def test_references_include_bellman_and_krumholz():
    r = compute_sequential_mdp_value(baseline_readmission_30d_prob=0.20)
    refs = " ".join(r.references)
    assert "Bellman" in refs
    assert "Krumholz" in refs


def test_optimal_action_matches_max_expected_value():
    r = compute_sequential_mdp_value(baseline_readmission_30d_prob=0.30)
    max_value = max(av.expected_value for av in r.action_values)
    assert r.optimal_action_expected_value == pytest.approx(max_value)
    optimal = next(av for av in r.action_values
                      if av.action == r.optimal_action)
    assert optimal.expected_value == pytest.approx(max_value)


# ─────────────────────── /api/mdp/decision endpoint ───────────────────────

@pytest.fixture(scope="module")
def app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod
    spec.loader.exec_module(mod)
    return mod.app


def test_mdp_endpoint_returns_report(app):
    from fastapi.testclient import TestClient
    body = {
        "baseline_readmission_30d_prob": 0.30,
        "horizon_days": 90,
    }
    with TestClient(app) as c:
        r = c.post("/api/mdp/decision", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["baseline_readmission_30d_prob"] == 0.30
    assert "action_values" in payload
    assert "optimal_action" in payload


def test_mdp_endpoint_rejects_invalid_baseline(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/mdp/decision",
                     json={"baseline_readmission_30d_prob": 5.0})
    assert r.status_code == 400
