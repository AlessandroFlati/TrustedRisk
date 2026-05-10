"""Unit tests for compute_maternal_early_warning + compute_preeclampsia_assessment."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.maternal_early_warning import compute_maternal_early_warning
from mcp_server.tools.preeclampsia_assessment import compute_preeclampsia_assessment


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── MEOWS ───────────────────────

def test_meows_normal_pregnancy_green():
    rep = _run(compute_maternal_early_warning(
        gestational_age_weeks=28, pregnancy_phase="antepartum",
        respiratory_rate=18, spo2=98, heart_rate=88,
        systolic_bp=115, diastolic_bp=70, temperature=37.0, avpu="A",
    ))
    assert rep.severity_tier == "green"
    assert rep.recommended_response == "routine_obs_4h"


def test_meows_severe_headache_pulls_red():
    """Severe headache + visual disturbance is a red flag regardless of vitals."""
    rep = _run(compute_maternal_early_warning(
        gestational_age_weeks=32, pregnancy_phase="antepartum",
        respiratory_rate=18, spo2=98, heart_rate=92,
        systolic_bp=158, diastolic_bp=95,
        severe_headache_or_visual=True, proteinuria_present=True,
    ))
    assert rep.severity_tier == "red"
    assert rep.recommended_response == "obstetric_emergency_response"


def test_meows_excess_bleeding_red():
    rep = _run(compute_maternal_early_warning(
        gestational_age_weeks=39, pregnancy_phase="postpartum",
        heart_rate=120, systolic_bp=85,
        excess_bleeding=True,
    ))
    assert rep.severity_tier == "red"


def test_meows_yellow_with_one_param_outside():
    rep = _run(compute_maternal_early_warning(
        gestational_age_weeks=28, pregnancy_phase="antepartum",
        respiratory_rate=18, spo2=94, heart_rate=92,
        systolic_bp=115, diastolic_bp=70, temperature=37.0,
    ))
    # SpO2 94 is outside yellow threshold (95-100)
    assert rep.severity_tier in ("yellow", "red")


def test_meows_pregnancy_higher_hr_normal():
    """HR 100 in 3rd trimester should be GREEN under MEOWS (vs yellow under NEWS2)."""
    rep = _run(compute_maternal_early_warning(
        gestational_age_weeks=36, pregnancy_phase="antepartum",
        respiratory_rate=18, spo2=98, heart_rate=100,
        systolic_bp=115, diastolic_bp=70, temperature=37.0,
    ))
    assert rep.severity_tier == "green"


# ─────────────────────── Preeclampsia ───────────────────────

def test_preeclampsia_normal_bp_no_diagnosis():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=32,
        systolic_bp=120, diastolic_bp=75, proteinuria_present=False,
    ))
    assert rep.classification == "no_preeclampsia"


def test_gestational_htn_only():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=32,
        systolic_bp=145, diastolic_bp=92, proteinuria_present=False,
    ))
    assert rep.classification == "gestational_hypertension"


def test_preeclampsia_without_severe():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=32,
        systolic_bp=145, diastolic_bp=92, proteinuria_present=True,
    ))
    assert rep.classification == "preeclampsia_without_severe_features"
    assert rep.delivery_recommended is False  # GA <37


def test_preeclampsia_with_severe_features_severe_bp():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=32,
        systolic_bp=165, diastolic_bp=112, proteinuria_present=True,
    ))
    assert rep.classification == "preeclampsia_with_severe_features"
    assert rep.magnesium_sulfate_indicated is True
    assert rep.antihypertensive_indicated is True


def test_preeclampsia_with_severe_visual_disturbance():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=32,
        systolic_bp=145, diastolic_bp=92, proteinuria_present=True,
        clinical_factors={"visual_disturbances": True,
                           "severe_headache_persistent": True},
    ))
    assert rep.classification == "preeclampsia_with_severe_features"


def test_eclampsia_seizure():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=33,
        systolic_bp=160, diastolic_bp=95, proteinuria_present=True,
        seizures_present=True,
    ))
    assert rep.classification == "eclampsia"
    assert rep.recommended_disposition == "icu_obstetric_anesthesia_consult"
    assert rep.delivery_recommended is True


def test_hellp_syndrome():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=33,
        systolic_bp=145, diastolic_bp=92, proteinuria_present=True,
        clinical_factors={"platelets_per_ul": 80_000, "ast_ul": 120,
                           "ldh_ul": 800, "haptoglobin_low": True},
    ))
    assert rep.classification == "hellp_syndrome"
    assert rep.delivery_recommended is True


def test_severe_at_term_delivers():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=37,
        systolic_bp=148, diastolic_bp=95, proteinuria_present=True,
    ))
    assert rep.delivery_recommended is True
    assert rep.recommended_disposition == "labor_and_delivery_for_delivery"


def test_severe_at_34weeks_delivers():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=34,
        systolic_bp=165, diastolic_bp=112, proteinuria_present=True,
    ))
    assert rep.delivery_recommended is True


def test_pre_20_weeks_abstains():
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=18,
        systolic_bp=145, diastolic_bp=95, proteinuria_present=True,
    ))
    assert rep.abstain_recommended is True
    assert "20w" in (rep.abstain_reason or "")
