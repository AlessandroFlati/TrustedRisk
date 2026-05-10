"""SIM-3 unit tests for the cost-effectiveness ladder."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent.cost_effectiveness_ladder import (
    build_cost_effectiveness_ladder,
)


def _build(**overrides):
    base = {
        "target_intervention_name": "TrustedRisk pharmacist counseling",
        "target_arr_30d_percentage_points": 3.5,
        "target_cost_per_patient_usd": 75.0,
        "avoided_event_cost_usd": 14_000.0,
        "qaly_gained_per_avoided_event": 0.05,
    }
    base.update(overrides)
    return build_cost_effectiveness_ladder(**base)


# ─────────────────────── Validation ───────────────────────

def test_negative_arr_raises():
    with pytest.raises(ValueError, match="arr"):
        _build(target_arr_30d_percentage_points=-1.0)


def test_arr_above_100_raises():
    with pytest.raises(ValueError, match="arr"):
        _build(target_arr_30d_percentage_points=110.0)


def test_negative_cost_raises():
    with pytest.raises(ValueError, match="cost"):
        _build(target_cost_per_patient_usd=-1.0)


def test_negative_avoided_event_cost_raises():
    with pytest.raises(ValueError, match="event_cost"):
        _build(avoided_event_cost_usd=-100.0)


def test_negative_qaly_raises():
    with pytest.raises(ValueError, match="qaly"):
        _build(qaly_gained_per_avoided_event=-0.01)


# ─────────────────────── Benchmark catalogue ───────────────────────

def test_five_benchmarks_returned():
    """Default ladder has 5 published RCTs."""
    ladder = _build()
    assert len(ladder.benchmarks) == 5
    intervention_names = {b.intervention for b in ladder.benchmarks}
    assert any("Pharmacist" in n for n in intervention_names)
    assert any("Care Transitions" in n for n in intervention_names)
    assert any("Naylor" in n.replace(" ", "") or
                  "Advanced Practice Nurse" in n
                  for n in intervention_names)


def test_benchmark_citations_present():
    ladder = _build()
    for b in ladder.benchmarks:
        assert b.citation
        assert b.notes


def test_benchmark_arr_in_range():
    ladder = _build()
    for b in ladder.benchmarks:
        assert 0.0 <= b.arr_30d_percentage_points <= 100.0


# ─────────────────────── Target metrics ───────────────────────

def test_target_cost_per_event_correct():
    """Target cost-per-event = cost_per_patient / ARR."""
    ladder = _build(target_arr_30d_percentage_points=5.0,
                       target_cost_per_patient_usd=50.0)
    # 50 / 0.05 = 1000
    assert ladder.target_cost_per_event_avoided_usd == pytest.approx(1_000.0)


def test_target_cost_per_qaly_correct():
    """Target $/QALY = (cost − avoided*ARR) / (ARR*QALY_per_event)."""
    ladder = _build(
        target_arr_30d_percentage_points=5.0,
        target_cost_per_patient_usd=50.0,
        avoided_event_cost_usd=10_000.0,
        qaly_gained_per_avoided_event=0.05,
    )
    # net_per_patient = 50 - 10000*0.05 = -450
    # qaly = 0.05 * 0.05 = 0.0025
    # cost_per_qaly = -450 / 0.0025 = -180000
    assert ladder.target_cost_per_qaly_usd == pytest.approx(-180_000.0)


def test_zero_arr_yields_none_cost_per_qaly():
    ladder = _build(target_arr_30d_percentage_points=0.0)
    assert ladder.target_cost_per_qaly_usd is None
    assert ladder.target_cost_per_event_avoided_usd is None


def test_zero_qaly_weight_yields_none_cost_per_qaly():
    ladder = _build(qaly_gained_per_avoided_event=0.0)
    assert ladder.target_cost_per_qaly_usd is None


# ─────────────────────── Ranking ───────────────────────

def test_dominant_target_ranks_first():
    """A cost-saving target with strong ARR should rank #1."""
    ladder = _build(
        target_arr_30d_percentage_points=10.0,
        target_cost_per_patient_usd=50.0,  # low cost, high ARR -> cost-saving
        avoided_event_cost_usd=14_000.0,
    )
    assert ladder.target_rank_among_benchmarks == 1


def test_weak_target_ranks_after_benchmarks():
    """Tiny ARR, high cost -> ranks behind most benchmarks."""
    ladder = _build(
        target_arr_30d_percentage_points=0.5,
        target_cost_per_patient_usd=2_000.0,
        avoided_event_cost_usd=14_000.0,
    )
    # Should be at least mid-pack or worse
    assert ladder.target_rank_among_benchmarks >= 3


def test_target_rank_in_valid_range():
    """Rank is always in [1, n_benchmarks + 1]."""
    ladder = _build()
    assert 1 <= ladder.target_rank_among_benchmarks <= len(ladder.benchmarks) + 1


# ─────────────────────── Rationale + references ───────────────────────

def test_rationale_mentions_qaly_when_present():
    ladder = _build()
    assert "$/QALY" in ladder.rationale or "QALY" in ladder.rationale


def test_references_include_canonical_papers():
    ladder = _build()
    refs = " ".join(ladder.references)
    assert "Schnipper" in refs
    assert "Coleman" in refs
    assert "Naylor" in refs


# ─────────────────────── /api/economics/ladder endpoint ───────────────────────

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


def test_endpoint_returns_ladder(app):
    from fastapi.testclient import TestClient
    body = {
        "target_intervention_name": "Test intervention",
        "target_arr_30d_percentage_points": 5.0,
        "target_cost_per_patient_usd": 100.0,
    }
    with TestClient(app) as c:
        r = c.post("/api/economics/ladder", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["target_intervention_name"] == "Test intervention"
    assert "benchmarks" in payload and len(payload["benchmarks"]) >= 4


def test_endpoint_rejects_invalid_arr(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/economics/ladder",
                     json={"target_arr_30d_percentage_points": -5.0,
                            "target_cost_per_patient_usd": 100.0})
    assert r.status_code == 400
