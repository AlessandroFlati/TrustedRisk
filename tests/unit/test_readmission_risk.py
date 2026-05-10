"""Unit tests for compute_readmission_risk -- LACE extraction + lookup."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mcp_server.tools.readmission_risk import (
    _compute_lace_components,
    _score_charlson,
    _score_los,
    _weight_for_component,
    _extract_observation_ids,
    compute_readmission_risk,
)


def _run(coro):
    return asyncio.run(coro)


def _bundle(*, los_days: int = 5, n_conditions: int = 4, n_ed: int = 2,
            is_acute: bool = True, observations: int = 0) -> dict:
    """Build a minimal Synthea-like FHIR bundle for testing."""
    entries = [{"resource": {"resourceType": "Patient", "id": "pt-test", "birthDate": "1955-03-15"}}]
    if is_acute:
        entries.append({"resource": {"resourceType": "Encounter", "id": "imp",
                                      "class": {"code": "IMP"},
                                      "period": {"start": "2025-12-01T08:00:00Z",
                                                 "end": f"2025-12-{1+los_days:02d}T11:00:00Z"}}})
    for i in range(n_ed):
        entries.append({"resource": {"resourceType": "Encounter", "id": f"ed-{i}", "class": {"code": "EMER"}}})
    for i in range(n_conditions):
        entries.append({"resource": {"resourceType": "Condition", "id": f"c-{i}"}})
    for i in range(observations):
        entries.append({"resource": {"resourceType": "Observation", "id": f"obs-{i}"}})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


# ─────────────────────── LACE-L scoring ───────────────────────

def test_los_below_1_zero():
    assert _score_los(0.5) == 0


def test_los_1_day():
    assert _score_los(1.0) == 1


def test_los_3_days():
    assert _score_los(3.0) == 3


def test_los_5_days():
    assert _score_los(5.0) == 4


def test_los_10_days():
    assert _score_los(10.0) == 5


def test_los_14_days():
    assert _score_los(14.0) == 7


def test_los_20_days_capped():
    assert _score_los(20.0) == 7


# ─────────────────────── LACE-C scoring ───────────────────────

def test_charlson_zero_zero_points():
    assert _score_charlson(0) == 0


def test_charlson_3_three_points():
    assert _score_charlson(3) == 3


def test_charlson_4_caps_at_5():
    assert _score_charlson(4) == 5


def test_charlson_negative_zero_points():
    assert _score_charlson(-1) == 0


# ─────────────────────── Bundle parsing ───────────────────────

def test_compute_lace_components_basic():
    b = _bundle(los_days=5, n_conditions=4, n_ed=2, is_acute=True)
    c = _compute_lace_components(b)
    assert c["l"] == 4  # 5 days -> 4 points
    assert c["a"] == 3  # acute IMP
    assert c["c"] == 5  # 4 conditions -> 5 points (cap)
    assert c["e"] == 2  # 2 EMER -> 2 points


def test_compute_lace_components_low_risk():
    b = _bundle(los_days=1, n_conditions=1, n_ed=0, is_acute=True)
    c = _compute_lace_components(b)
    assert c["l"] == 1
    assert c["a"] == 3
    assert c["c"] == 1
    assert c["e"] == 0


def test_compute_lace_components_non_acute():
    b = _bundle(los_days=3, n_conditions=2, n_ed=1, is_acute=False)
    c = _compute_lace_components(b)
    assert c["a"] == 0  # no IMP encounter
    # n_ed=1 EMER encounter -- counted in E since non-IMP
    assert c["e"] == 1


def test_compute_lace_components_ed_caps_at_4():
    b = _bundle(los_days=2, n_conditions=2, n_ed=10, is_acute=True)
    c = _compute_lace_components(b)
    assert c["e"] == 4


def test_extract_observation_ids():
    b = _bundle(observations=3)
    ids = _extract_observation_ids(b)
    assert len(ids) == 3
    assert "obs-0" in ids


# ─────────────────────── Weight computation ───────────────────────

def test_weight_for_component_total_zero():
    assert _weight_for_component(3, 0) == 0.0


def test_weight_for_component_normal():
    assert _weight_for_component(4, 14) == pytest.approx(4 / 14)


def test_weight_for_component_capped_at_1():
    assert _weight_for_component(20, 14) == 1.0


# ─────────────────────── End-to-end via in-memory FHIR stub ───────────────────────

def test_compute_readmission_risk_returns_calibrated_estimate(monkeypatch):
    """Patch fetch_patient_bundle + resolve_patient_id to bypass live FHIR."""
    bundle = _bundle(los_days=5, n_conditions=4, n_ed=2, is_acute=True, observations=2)

    async def stub_fetch(pid):
        return bundle

    async def stub_resolve(explicit):
        return explicit or "pt-test"

    from mcp_server.tools import readmission_risk as rr
    monkeypatch.setattr(rr, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(rr, "resolve_patient_id", stub_resolve)

    risk = _run(compute_readmission_risk(horizon_days=30, patient_id="pt-test"))
    assert risk.model_name == "lace-plus-bayesian-v1"
    assert risk.horizon_days == 30
    assert 0.0 <= risk.probability_mean <= 1.0
    lo, hi = risk.probability_ci95
    assert 0.0 <= lo <= risk.probability_mean <= hi <= 1.0
    assert risk.lace_raw_score == 14  # L=4 + A=3 + C=5 + E=2
    assert len(risk.contributing_factors) == 4
    factor_names = {f.name for f in risk.contributing_factors}
    assert factor_names == {
        "LACE_length_of_stay", "LACE_acuity",
        "LACE_comorbidity", "LACE_ed_visits_6mo",
    }


def test_compute_readmission_risk_low_lace_lookup(monkeypatch):
    bundle = _bundle(los_days=1, n_conditions=1, n_ed=0, is_acute=True)
    from mcp_server.tools import readmission_risk as rr

    async def stub_fetch(pid): return bundle
    async def stub_resolve(explicit): return explicit or "pt-test"
    monkeypatch.setattr(rr, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(rr, "resolve_patient_id", stub_resolve)

    risk = _run(compute_readmission_risk(patient_id="pt-low"))
    assert risk.lace_raw_score == 5  # L=1 + A=3 + C=1 + E=0
    # Low LACE -> low probability
    assert risk.probability_mean < 0.20


def test_compute_readmission_risk_max_lace_saturates(monkeypatch):
    bundle = _bundle(los_days=14, n_conditions=10, n_ed=5, is_acute=True)
    from mcp_server.tools import readmission_risk as rr

    async def stub_fetch(pid): return bundle
    async def stub_resolve(explicit): return explicit or "pt-test"
    monkeypatch.setattr(rr, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(rr, "resolve_patient_id", stub_resolve)

    risk = _run(compute_readmission_risk(patient_id="pt-high"))
    # L=7 + A=3 + C=5 + E=4 = 19 (max)
    assert risk.lace_raw_score == 19
