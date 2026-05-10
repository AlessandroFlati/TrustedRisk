"""SIM-1 unit tests for the Monte Carlo hospital outcomes simulator + endpoint."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent.outcomes_simulator import simulate_hospital_year
from shared.schemas import CaseMixSegment


# ─────────────────────── Validation guards ───────────────────────

def test_empty_case_mix_raises():
    with pytest.raises(ValueError, match="case_mix"):
        simulate_hospital_year(case_mix=[])


def test_invalid_rrr_raises():
    with pytest.raises(ValueError, match="rrr"):
        simulate_hospital_year(
            case_mix=[CaseMixSegment(name="x", n_patients_per_year=10,
                                          baseline_event_probability=0.2)],
            intervention_relative_risk_reduction=1.5,
        )


def test_negative_event_cost_raises():
    with pytest.raises(ValueError, match="event_cost"):
        simulate_hospital_year(
            case_mix=[CaseMixSegment(name="x", n_patients_per_year=10,
                                          baseline_event_probability=0.2)],
            avoided_event_cost_usd=-1.0,
        )


def test_invalid_n_iterations_raises():
    with pytest.raises(ValueError, match="n_iterations"):
        simulate_hospital_year(
            case_mix=[CaseMixSegment(name="x", n_patients_per_year=10,
                                          baseline_event_probability=0.2)],
            n_iterations=0,
        )


def test_dict_case_mix_coerced():
    """Dicts must be coerced to CaseMixSegment."""
    r = simulate_hospital_year(
        case_mix=[{"name": "CHF", "n_patients_per_year": 50,
                     "baseline_event_probability": 0.30}],
        n_iterations=200, seed=1,
    )
    assert r.cohort_size_per_year == 50


# ─────────────────────── Reproducibility ───────────────────────

def test_seed_makes_simulation_reproducible():
    case_mix = [
        CaseMixSegment(name="A", n_patients_per_year=200,
                          baseline_event_probability=0.30),
        CaseMixSegment(name="B", n_patients_per_year=100,
                          baseline_event_probability=0.10),
    ]
    a = simulate_hospital_year(case_mix=case_mix, n_iterations=500, seed=7)
    b = simulate_hospital_year(case_mix=case_mix, n_iterations=500, seed=7)
    assert a.total_events_avoided_mean == b.total_events_avoided_mean
    assert a.total_events_avoided_ci95 == b.total_events_avoided_ci95


def test_different_seeds_diverge():
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=300,
                                    baseline_event_probability=0.20)]
    a = simulate_hospital_year(case_mix=case_mix, n_iterations=1000, seed=1)
    b = simulate_hospital_year(case_mix=case_mix, n_iterations=1000, seed=99)
    # Means won't match exactly -- at least one should differ
    assert a.total_events_avoided_mean != b.total_events_avoided_mean or \
           a.total_events_avoided_ci95 != b.total_events_avoided_ci95


# ─────────────────────── Closed-form sanity checks ───────────────────────

def test_zero_rrr_means_zero_avoided():
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=500,
                                    baseline_event_probability=0.30)]
    r = simulate_hospital_year(
        case_mix=case_mix,
        intervention_relative_risk_reduction=0.0,
        n_iterations=1000, seed=42,
    )
    # Variance ≈ 2 * n * p * (1-p) ≈ 210, std ≈ 14.5; 95% CI half-width ≈ 28
    # Mean should be near 0 -- sample CI should bracket 0
    lo, hi = r.total_events_avoided_ci95
    assert lo <= 0 <= hi


def test_full_rrr_eliminates_events_in_with_intervention_arm():
    """With RRR=1, post-intervention events should be 0, so events_avoided
    equals the no-intervention draw."""
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=200,
                                    baseline_event_probability=0.50)]
    r = simulate_hospital_year(
        case_mix=case_mix,
        intervention_relative_risk_reduction=1.0,
        n_iterations=2000, seed=42,
    )
    seg = r.segments[0]
    assert seg.expected_events_with_intervention == pytest.approx(0.0)
    # Mean events avoided should be near n*p = 100
    assert 90 <= r.total_events_avoided_mean <= 110


def test_arr_scales_linearly_with_cohort_size():
    """Doubling cohort size should ~double expected events avoided."""
    small = [CaseMixSegment(name="A", n_patients_per_year=100,
                                  baseline_event_probability=0.30)]
    big = [CaseMixSegment(name="A", n_patients_per_year=1000,
                                baseline_event_probability=0.30)]
    a = simulate_hospital_year(case_mix=small, n_iterations=2000,
                                  intervention_relative_risk_reduction=0.25,
                                  seed=42)
    b = simulate_hospital_year(case_mix=big, n_iterations=2000,
                                  intervention_relative_risk_reduction=0.25,
                                  seed=42)
    ratio = b.total_events_avoided_mean / max(0.01, a.total_events_avoided_mean)
    assert 7.0 <= ratio <= 13.0  # Roughly 10× ± noise


def test_cost_savings_match_events_times_avoided_cost():
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=500,
                                    baseline_event_probability=0.30)]
    r = simulate_hospital_year(
        case_mix=case_mix, n_iterations=2000,
        avoided_event_cost_usd=10_000.0, seed=42)
    expected_savings = r.total_events_avoided_mean * 10_000.0
    assert abs(r.total_avoided_event_cost_mean_usd - expected_savings) < 1.0


def test_net_cost_equation():
    """net = intervention_cost - avoided_event_cost (mean over iterations)."""
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=200,
                                    baseline_event_probability=0.30)]
    r = simulate_hospital_year(
        case_mix=case_mix, n_iterations=1000,
        intervention_cost_per_patient_usd=100.0,
        avoided_event_cost_usd=10_000.0,
        seed=42,
    )
    expected_net = r.total_intervention_cost_usd - \
        r.total_avoided_event_cost_mean_usd
    assert abs(r.net_cost_mean_usd - expected_net) < 1.0


# ─────────────────────── QALY pathway ───────────────────────

def test_qaly_gained_when_qaly_provided():
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=300,
                                    baseline_event_probability=0.30)]
    r = simulate_hospital_year(
        case_mix=case_mix, n_iterations=2000,
        qaly_gained_per_avoided_event=0.10, seed=42,
    )
    assert r.total_qaly_gained_mean is not None
    assert r.total_qaly_gained_mean > 0
    assert r.cost_per_qaly_mean_usd is not None


def test_no_qaly_when_qaly_none():
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=300,
                                    baseline_event_probability=0.30)]
    r = simulate_hospital_year(
        case_mix=case_mix, n_iterations=500,
        qaly_gained_per_avoided_event=None, seed=42,
    )
    assert r.total_qaly_gained_mean is None
    assert r.cost_per_qaly_mean_usd is None


# ─────────────────────── Sensitivity ───────────────────────

def test_sensitivity_grid_includes_default_three_rrr_three_cost_rows():
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=100,
                                    baseline_event_probability=0.20)]
    r = simulate_hospital_year(case_mix=case_mix, n_iterations=500, seed=42)
    rrr_rows = [s for s in r.sensitivity
                  if s.parameter == "intervention_rrr"]
    cost_rows = [s for s in r.sensitivity
                   if s.parameter == "avoided_event_cost_usd"]
    assert len(rrr_rows) == 3
    assert len(cost_rows) == 3


def test_higher_rrr_yields_more_avoided_events_in_sensitivity():
    """Sensitivity table should be monotonic in RRR."""
    case_mix = [CaseMixSegment(name="A", n_patients_per_year=500,
                                    baseline_event_probability=0.30)]
    r = simulate_hospital_year(
        case_mix=case_mix, n_iterations=2000,
        rrr_grid=[0.10, 0.25, 0.50], seed=42,
    )
    rrr_rows = sorted([s for s in r.sensitivity
                          if s.parameter == "intervention_rrr"],
                         key=lambda x: x.value)
    means = [s.events_avoided_mean for s in rrr_rows]
    assert means == sorted(means)


# ─────────────────────── Per-segment reporting ───────────────────────

def test_each_segment_returned_in_order():
    case_mix = [
        CaseMixSegment(name="alpha", n_patients_per_year=50,
                          baseline_event_probability=0.20),
        CaseMixSegment(name="beta", n_patients_per_year=100,
                          baseline_event_probability=0.40),
    ]
    r = simulate_hospital_year(case_mix=case_mix, n_iterations=200, seed=1)
    assert [s.name for s in r.segments] == ["alpha", "beta"]
    assert r.segments[0].n_patients == 50
    assert r.segments[1].n_patients == 100


# ─────────────────────── /api/simulate/year endpoint ───────────────────────

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


def test_endpoint_runs_simulation(app):
    from fastapi.testclient import TestClient
    body = {
        "case_mix": [
            {"name": "CHF", "n_patients_per_year": 200,
             "baseline_event_probability": 0.30},
            {"name": "AKI", "n_patients_per_year": 100,
             "baseline_event_probability": 0.20},
        ],
        "n_iterations": 500, "seed": 42,
    }
    with TestClient(app) as c:
        r = c.post("/api/simulate/year", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["cohort_size_per_year"] == 300
    assert "total_events_avoided_mean" in payload
    assert isinstance(payload["sensitivity"], list)


def test_endpoint_rejects_empty_case_mix(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/simulate/year", json={"case_mix": []})
    assert r.status_code == 400


def test_endpoint_rejects_invalid_rrr(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/simulate/year", json={
            "case_mix": [{"name": "x", "n_patients_per_year": 10,
                            "baseline_event_probability": 0.2}],
            "intervention_relative_risk_reduction": 5.0,
        })
    assert r.status_code == 400
