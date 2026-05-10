"""Unit tests for compute_clinical_deterioration_score (NEWS2)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from mcp_server.tools.clinical_deterioration_score import (
    _bracket_score,
    _classify_severity,
    _classify_trend,
    _NEWS2_HR,
    _NEWS2_RR,
    _NEWS2_SBP,
    _NEWS2_SPO2_SCALE_1,
    _NEWS2_TEMP,
    compute_clinical_deterioration_score,
)


def _run(coro):
    return asyncio.run(coro)


def _vital(t: str, v, mins_ago: int = 0) -> dict:
    return {
        "type": t,
        "value": v,
        "observed_at": (datetime.now(timezone.utc)
                         - timedelta(minutes=mins_ago)).isoformat(),
    }


# ─────────────────────── NEWS2 bracket scoring ───────────────────────

@pytest.mark.parametrize("rr,expected", [
    (8, 3), (9, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3), (30, 3),
])
def test_rr_brackets(rr, expected):
    assert _bracket_score(rr, _NEWS2_RR) == expected


@pytest.mark.parametrize("spo2,expected", [
    (89, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (99, 0),
])
def test_spo2_brackets(spo2, expected):
    assert _bracket_score(spo2, _NEWS2_SPO2_SCALE_1) == expected


@pytest.mark.parametrize("sbp,expected", [
    (85, 3), (95, 2), (105, 1), (115, 0), (200, 0), (220, 3), (240, 3),
])
def test_sbp_brackets(sbp, expected):
    assert _bracket_score(sbp, _NEWS2_SBP) == expected


@pytest.mark.parametrize("hr,expected", [
    (35, 3), (45, 1), (60, 0), (90, 0), (95, 1), (115, 2), (135, 3),
])
def test_hr_brackets(hr, expected):
    assert _bracket_score(hr, _NEWS2_HR) == expected


@pytest.mark.parametrize("t,expected", [
    (34.5, 3), (35.5, 1), (37.0, 0), (38.5, 1), (39.5, 2),
])
def test_temp_brackets(t, expected):
    assert _bracket_score(t, _NEWS2_TEMP) == expected


# ─────────────────────── Severity classification ───────────────────────

def test_severity_low():
    sev, resp = _classify_severity(0, [])
    assert sev == "low"
    assert resp == "routine_12h_obs"


def test_severity_low_medium():
    sev, resp = _classify_severity(3, [])
    assert sev == "low_medium"
    assert resp == "increase_obs_4_6h"


def test_severity_medium_at_5():
    sev, resp = _classify_severity(5, [])
    assert sev == "medium"
    assert resp == "urgent_review_1h"


def test_severity_high_at_7():
    sev, resp = _classify_severity(7, [])
    assert sev == "high"
    assert resp == "emergency_response"


def test_single_three_escalates():
    """A single parameter of 3 should escalate to medium even at total 3-4."""
    from mcp_server.tools.clinical_deterioration_score import DeteriorationFactor
    contribs = [DeteriorationFactor(parameter="spo2", value=88, points=3,
                                      rationale="hypoxic")]
    sev, resp = _classify_severity(3, contribs)
    assert sev == "medium"


# ─────────────────────── Trend classification ───────────────────────

def test_trend_unknown_when_no_prior():
    delta, direction = _classify_trend(5, None)
    assert delta == 0
    assert direction == "unknown"


def test_trend_worsening():
    delta, direction = _classify_trend(7, 4)
    assert delta == 3
    assert direction == "worsening"


def test_trend_stable():
    delta, direction = _classify_trend(5, 4)
    assert delta == 1
    assert direction == "stable"


def test_trend_improving():
    delta, direction = _classify_trend(2, 5)
    assert delta == -3
    assert direction == "improving"


# ─────────────────────── End-to-end ───────────────────────

def test_e2e_normal_patient():
    vitals = [
        _vital("respiratory_rate", 16),
        _vital("spo2", 98),
        _vital("systolic_bp", 130),
        _vital("heart_rate", 75),
        _vital("temperature", 37.0),
        _vital("consciousness", "A"),
    ]
    rep = _run(compute_clinical_deterioration_score(vital_signs=vitals))
    assert rep.score_total == 0
    assert rep.severity_tier == "low"
    assert rep.recommended_response == "routine_12h_obs"


def test_e2e_septic_patient():
    """Septic shock pattern: high RR, low SpO2, hypotension, tachycardia, fever."""
    vitals = [
        _vital("respiratory_rate", 28),       # +2
        _vital("spo2", 91),                    # +3
        _vital("supplemental_oxygen", True),   # +2
        _vital("systolic_bp", 88),             # +3
        _vital("heart_rate", 122),             # +2
        _vital("temperature", 39.2),           # +2
        _vital("consciousness", "V"),          # +3
    ]
    rep = _run(compute_clinical_deterioration_score(vital_signs=vitals))
    assert rep.score_total >= 15
    assert rep.severity_tier == "high"
    assert rep.recommended_response == "emergency_response"


def test_e2e_picks_most_recent_per_type():
    base = datetime.now(timezone.utc)
    vitals = [
        {"type": "heart_rate", "value": 95, "observed_at": (base - timedelta(hours=4)).isoformat()},
        {"type": "heart_rate", "value": 78, "observed_at": (base - timedelta(hours=1)).isoformat()},
    ]
    rep = _run(compute_clinical_deterioration_score(vital_signs=vitals))
    # 78 is normal (0 points); 95 would have been +1.
    assert rep.score_total == 0


def test_e2e_trend_worsening_from_prior():
    vitals = [
        _vital("respiratory_rate", 25),
        _vital("heart_rate", 115),
        _vital("systolic_bp", 95),
    ]
    rep = _run(compute_clinical_deterioration_score(vital_signs=vitals, prior_score=3))
    assert rep.trend_direction == "worsening"
    assert rep.trend_delta >= 2


def test_e2e_no_vitals_returns_unknown():
    """Empty vital_signs -> abstain. The absence of vitals is NOT a
    low-acuity finding; the placeholder score must be flagged as
    abstain_recommended so a downstream consumer cannot present it
    as a clean NEWS2=0 reading."""
    rep = _run(compute_clinical_deterioration_score(vital_signs=[]))
    assert rep.score_total == 0
    assert rep.trend_direction == "unknown"
    assert rep.abstain_recommended is True
    assert rep.abstain_reason is not None
    assert "missing_vital_signs" in rep.abstain_reason


def test_e2e_avpu_anything_but_alert_flags():
    vitals = [_vital("consciousness", "P")]
    rep = _run(compute_clinical_deterioration_score(vital_signs=vitals))
    contributions = {c.parameter: c.points for c in rep.parameter_contributions}
    assert contributions["consciousness"] == 3


def test_e2e_supplemental_oxygen_strings():
    """The boolean field accepts Boolean or truthy strings."""
    for val in (True, "true", "yes", 1, "supplemental"):
        vitals = [_vital("supplemental_oxygen", val)]
        rep = _run(compute_clinical_deterioration_score(vital_signs=vitals))
        contribs = {c.parameter: c.points for c in rep.parameter_contributions}
        assert contribs["supplemental_oxygen"] == 2, f"failed for value {val!r}"


def test_e2e_via_fhir_bundle(monkeypatch):
    bundle = {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Observation",
                          "code": {"text": "Heart rate"},
                          "valueQuantity": {"value": 130, "unit": "/min"},
                          "effectiveDateTime": "2026-04-27T10:00:00Z"}},
            {"resource": {"resourceType": "Observation",
                          "code": {"text": "SpO2"},
                          "valueQuantity": {"value": 90, "unit": "%"},
                          "effectiveDateTime": "2026-04-27T10:00:00Z"}},
            {"resource": {"resourceType": "Observation",
                          "code": {"text": "Respiratory rate"},
                          "valueQuantity": {"value": 26, "unit": "/min"},
                          "effectiveDateTime": "2026-04-27T10:00:00Z"}},
        ],
    }

    async def stub_fetch(pid):
        return bundle

    async def stub_resolve(explicit):
        return explicit or "pt-1"

    from mcp_server.tools import clinical_deterioration_score as cds
    monkeypatch.setattr(cds, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(cds, "resolve_patient_id", stub_resolve)

    rep = _run(compute_clinical_deterioration_score(patient_id="pt-1"))
    assert rep.score_total >= 5
    assert rep.severity_tier in ("medium", "high")
