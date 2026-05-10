"""SIM-2 unit tests for the population equity dashboard + endpoint."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent.equity_dashboard import compute_equity_dashboard


def _entry(*, action="discharge_home", confidence="high",
              risk=0.20, age=70, race="white", sex="male",
              insurance="medicare"):
    return {
        "decision_card": {
            "recommendation": ({"action": action, "confidence": confidence}
                                  if action else None),
            "reasoning": {"risk_estimate": {"probability_mean": risk}},
            "audit": {}, "validation": {}, "abstain": [],
        },
        "demographics": {"age": age, "race": race, "sex": sex,
                            "insurance_type": insurance},
    }


# ─────────────────────── Validation ───────────────────────

def test_invalid_threshold_raises():
    with pytest.raises(ValueError, match="disparity_threshold_pct"):
        compute_equity_dashboard([], disparity_threshold_pct=2.0)


def test_empty_cohort_returns_zeros():
    d = compute_equity_dashboard([])
    assert d.n_total_decisions == 0
    assert d.segments == []
    assert d.flagged_segments == []


# ─────────────────────── Slicing dimensions ───────────────────────

def test_slices_by_age_race_sex_insurance():
    cohort = [_entry(age=70, race="white", sex="male", insurance="medicare")]
    d = compute_equity_dashboard(cohort)
    dims = {s.subgroup_dimension for s in d.segments}
    assert dims == {"age_band", "race", "sex", "insurance_type"}


def test_age_band_assignment():
    cohort = [
        _entry(age=12),  # 0-17
        _entry(age=35),  # 18-44
        _entry(age=50),  # 45-64
        _entry(age=70),  # 65-74
        _entry(age=80),  # 75-84
        _entry(age=92),  # 85+
    ]
    d = compute_equity_dashboard(cohort)
    age_seg_values = sorted({
        s.subgroup_value for s in d.segments
        if s.subgroup_dimension == "age_band"
    })
    assert age_seg_values == ["0-17", "18-44", "45-64",
                                  "65-74", "75-84", "85+"]


def test_missing_demographics_skipped():
    """Entries without demographics shouldn't fall into any segment."""
    cohort = [
        _entry(age=70),
        {"decision_card": {"recommendation":
                                {"action": "home_with_care",
                                 "confidence": "high"}},
         "demographics": {}},
    ]
    d = compute_equity_dashboard(cohort)
    # n_total = 2 but only 1 contributed to segments
    assert d.n_total_decisions == 2
    age_seg = next(s for s in d.segments
                      if s.subgroup_dimension == "age_band")
    assert age_seg.n_decisions == 1


# ─────────────────────── Action distribution ───────────────────────

def test_action_counts_per_segment():
    cohort = [
        _entry(action="home_with_care", race="black"),
        _entry(action="snf", race="black"),
        _entry(action="discharge_home", race="black"),
    ]
    d = compute_equity_dashboard(cohort)
    black_seg = next(s for s in d.segments
                        if s.subgroup_dimension == "race"
                        and s.subgroup_value == "black")
    assert black_seg.action_counts == {
        "home_with_care": 1, "snf": 1, "discharge_home": 1,
    }


def test_intervention_rate_correct():
    """2 of 4 decisions in a segment are non-discharge_home -> 0.5"""
    cohort = [
        _entry(action="home_with_care", race="black"),
        _entry(action="snf", race="black"),
        _entry(action="discharge_home", race="black"),
        _entry(action="discharge_home", race="black"),
    ]
    d = compute_equity_dashboard(cohort)
    black_seg = next(s for s in d.segments
                        if s.subgroup_dimension == "race"
                        and s.subgroup_value == "black")
    assert black_seg.intervention_rate == pytest.approx(0.5)


def test_abstention_rate_correct():
    """1 of 2 decisions abstain -> 0.5"""
    cohort = [
        _entry(action="home_with_care", race="black"),
        _entry(action=None, race="black"),
    ]
    d = compute_equity_dashboard(cohort)
    black_seg = next(s for s in d.segments
                        if s.subgroup_dimension == "race"
                        and s.subgroup_value == "black")
    assert black_seg.abstention_rate == pytest.approx(0.5)


# ─────────────────────── Disparity flags ───────────────────────

def test_flagged_when_intervention_rate_diverges_above_threshold():
    """Black 1.0 vs white 0.0 with threshold 0.15 -> flagged."""
    cohort = (
        [_entry(action="home_with_care", race="black") for _ in range(5)]
        + [_entry(action="discharge_home", race="white") for _ in range(5)]
    )
    d = compute_equity_dashboard(cohort, disparity_threshold_pct=0.15)
    flagged_set = set(d.flagged_segments)
    # Both segments diverge from cohort mean (0.5) by 0.5 -> above threshold
    assert "race=black|intervention_rate" in flagged_set
    assert "race=white|intervention_rate" in flagged_set


def test_no_flags_when_balanced():
    cohort = (
        [_entry(action="home_with_care", race="black") for _ in range(5)]
        + [_entry(action="home_with_care", race="white") for _ in range(5)]
    )
    d = compute_equity_dashboard(cohort, disparity_threshold_pct=0.10)
    assert d.flagged_segments == []


def test_disparity_max_metrics():
    cohort = (
        [_entry(action="home_with_care", race="black", risk=0.40) for _ in range(5)]
        + [_entry(action="discharge_home", race="white", risk=0.10) for _ in range(5)]
    )
    d = compute_equity_dashboard(cohort, disparity_threshold_pct=0.10)
    assert d.max_intervention_rate_disparity > 0.0
    assert d.max_avg_risk_disparity == pytest.approx(0.30, rel=1e-3)


# ─────────────────────── Confidence high rate ───────────────────────

def test_confidence_high_rate():
    cohort = [
        _entry(action="home_with_care", confidence="high", race="x"),
        _entry(action="home_with_care", confidence="medium", race="x"),
        _entry(action="home_with_care", confidence="medium", race="x"),
    ]
    d = compute_equity_dashboard(cohort)
    seg = next(s for s in d.segments
                  if s.subgroup_dimension == "race" and s.subgroup_value == "x")
    assert seg.confidence_high_rate == pytest.approx(1 / 3)


# ─────────────────────── References + disclaimer ───────────────────────

def test_disclaimer_present():
    d = compute_equity_dashboard([_entry()])
    assert "disparity does not" in d.disclaimer.lower() or \
           "not causal" in d.disclaimer.lower()


def test_references_include_obermeyer():
    d = compute_equity_dashboard([_entry()])
    assert any("Obermeyer" in r for r in d.references)


# ─────────────────────── /api/equity/cohort endpoint ───────────────────────

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


def test_endpoint_returns_dashboard(app):
    from fastapi.testclient import TestClient
    cohort = [_entry(race="black"), _entry(race="white")]
    with TestClient(app) as c:
        r = c.post("/api/equity/cohort", json={"cohort": cohort})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_total_decisions"] == 2
    assert "segments" in body and len(body["segments"]) > 0


def test_endpoint_rejects_non_list(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/equity/cohort", json={"cohort": "nope"})
    assert r.status_code == 400


def test_endpoint_rejects_invalid_threshold(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/equity/cohort", json={
            "cohort": [_entry()],
            "disparity_threshold_pct": 2.5,
        })
    assert r.status_code == 400
