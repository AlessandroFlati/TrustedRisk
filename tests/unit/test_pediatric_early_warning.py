"""Unit tests for compute_pediatric_early_warning."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.pediatric_early_warning import (
    _age_band,
    _classify_severity,
    compute_pediatric_early_warning,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Age-band selection ───────────────────────

@pytest.mark.parametrize("months,expected", [
    (0, "0-11mo"), (6, "0-11mo"), (11, "0-11mo"),
    (12, "1-4y"), (36, "1-4y"), (59, "1-4y"),
    (60, "5-11y"), (100, "5-11y"), (143, "5-11y"),
    (144, "12-17y"), (200, "12-17y"),
])
def test_age_band(months, expected):
    band, _ = _age_band(months)
    assert band == expected


# ─────────────────────── Severity classification ───────────────────────

def test_severity_low():
    sev, resp = _classify_severity(2, [])
    assert sev == "low"
    assert resp == "routine_obs"


def test_severity_medium():
    sev, resp = _classify_severity(4, [])
    assert sev == "medium"
    assert resp == "increase_to_hourly_obs"


def test_severity_high_at_5():
    sev, resp = _classify_severity(5, [])
    assert sev == "high"
    assert resp == "urgent_pediatric_review"


def test_severity_picu_at_7():
    sev, resp = _classify_severity(7, [])
    assert sev == "high"
    assert resp == "rapid_response_picu_consult"


# ─────────────────────── End-to-end ───────────────────────

def test_e2e_normal_toddler():
    rep = _run(compute_pediatric_early_warning(
        age_months=24,
        behavior="appropriate",
        heart_rate=110,
        respiratory_rate=28,
        spo2=99,
        systolic_bp=95,
    ))
    assert rep.score_total == 0
    assert rep.severity_tier == "low"
    assert rep.age_band == "1-4y"


def test_e2e_septic_infant():
    rep = _run(compute_pediatric_early_warning(
        age_months=4,
        behavior="lethargic",
        heart_rate=190,
        respiratory_rate=70,
        spo2=89,
        capillary_refill_seconds=4.5,
        accessory_muscle_use=True,
        on_supplemental_oxygen=True,
        parental_or_nurse_concern=True,
    ))
    assert rep.score_total >= 8
    assert rep.severity_tier == "high"
    assert rep.recommended_response == "rapid_response_picu_consult"


def test_e2e_parental_concern_alone_meaningful():
    rep = _run(compute_pediatric_early_warning(
        age_months=24,
        behavior="appropriate",
        heart_rate=110,
        parental_or_nurse_concern=True,
    ))
    # +2 for parental concern alone
    assert rep.score_total == 2


def test_e2e_age_band_boundary():
    """At 12 months -> 1-4y band; at 11 months -> 0-11mo band."""
    a = _run(compute_pediatric_early_warning(age_months=12, heart_rate=140))
    b = _run(compute_pediatric_early_warning(age_months=11, heart_rate=140))
    assert a.age_band == "1-4y"
    assert b.age_band == "0-11mo"


def test_e2e_irritable_behavior_scores_2():
    rep = _run(compute_pediatric_early_warning(age_months=24, behavior="irritable",
                                                  heart_rate=110))
    behavior = next(c for c in rep.components if c.component == "behavior")
    assert behavior.points == 2


def test_e2e_age_band_specific_normals():
    """Same HR=140 is normal for an infant but tachycardic for an 11-year-old."""
    infant = _run(compute_pediatric_early_warning(age_months=6, heart_rate=140))
    older = _run(compute_pediatric_early_warning(age_months=132, heart_rate=140))  # 11y
    cv_infant = next(c for c in infant.components if c.component == "cardiovascular").points
    cv_older = next(c for c in older.components if c.component == "cardiovascular").points
    assert cv_older > cv_infant
