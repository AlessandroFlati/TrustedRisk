"""Unit tests for compute_heart_score + compute_acs_disposition_decision."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.heart_score import compute_heart_score
from mcp_server.tools.acs_disposition_decision import compute_acs_disposition_decision


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── HEART score ───────────────────────

def test_low_heart_under_45_no_rf_normal_trop():
    rep = _run(compute_heart_score(
        history_descriptor="slightly_suspicious",
        ecg_descriptor="normal",
        age=35, risk_factors_count=0,
        troponin_times_uln=0.0,
    ))
    assert rep.total_score == 0
    assert rep.risk_band == "low"


def test_high_heart_typical_chest_pain():
    rep = _run(compute_heart_score(
        history_descriptor="highly_suspicious",
        ecg_descriptor="significant_st_depression",
        age=70, risk_factors_count=4,
        troponin_times_uln=4.0,
    ))
    assert rep.total_score == 10
    assert rep.risk_band == "high"
    assert rep.estimated_30d_mace_risk_pct > 40


def test_known_cad_overrides_rf_count():
    rep = _run(compute_heart_score(
        history_descriptor="moderately_suspicious",
        ecg_descriptor="normal", age=58,
        risk_factors_count=0, known_atherosclerotic_disease=True,
        troponin_times_uln=0.0,
    ))
    # Risk_factors_points should be 2 even with rf_count=0, due to known CAD
    assert rep.risk_factors_points == 2


def test_age_bands():
    young = _run(compute_heart_score(history_descriptor="non_suspicious",
                                          ecg_descriptor="normal", age=30,
                                          troponin_times_uln=0.0,
                                          risk_factors_count=0))
    middle = _run(compute_heart_score(history_descriptor="non_suspicious",
                                           ecg_descriptor="normal", age=50,
                                           troponin_times_uln=0.0,
                                           risk_factors_count=0))
    old = _run(compute_heart_score(history_descriptor="non_suspicious",
                                        ecg_descriptor="normal", age=72,
                                        troponin_times_uln=0.0,
                                        risk_factors_count=0))
    assert young.age_points == 0
    assert middle.age_points == 1
    assert old.age_points == 2


def test_troponin_bands():
    a = _run(compute_heart_score(history_descriptor="non_suspicious",
                                       ecg_descriptor="normal",
                                       troponin_times_uln=0.5, age=40,
                                       risk_factors_count=0))
    b = _run(compute_heart_score(history_descriptor="non_suspicious",
                                       ecg_descriptor="normal",
                                       troponin_times_uln=2.0, age=40,
                                       risk_factors_count=0))
    c = _run(compute_heart_score(history_descriptor="non_suspicious",
                                       ecg_descriptor="normal",
                                       troponin_times_uln=5.0, age=40,
                                       risk_factors_count=0))
    assert a.troponin_points == 0
    assert b.troponin_points == 1
    assert c.troponin_points == 2


def test_moderate_heart_typical_workup_pt():
    rep = _run(compute_heart_score(
        history_descriptor="moderately_suspicious",
        ecg_descriptor="non_specific_repolarization",
        age=58, risk_factors_count=2,
        troponin_times_uln=0.5,
    ))
    assert rep.risk_band == "moderate"
    assert 4 <= rep.total_score <= 6


# ─────────────────────── ACS disposition ───────────────────────

def test_acs_stemi_immediate_cath():
    rep = _run(compute_acs_disposition_decision(
        heart_score_total=8, has_stemi=True,
    ))
    assert rep.disposition == "cath_lab_activation_immediate"


def test_acs_dynamic_troponin_admits():
    rep = _run(compute_acs_disposition_decision(
        heart_score_total=5, has_dynamic_troponin=True,
    ))
    assert rep.disposition == "admit_telemetry_for_workup"


def test_acs_low_heart_discharges():
    rep = _run(compute_acs_disposition_decision(heart_score_total=2))
    assert rep.disposition == "discharge_with_outpatient_followup"
    assert rep.recommended_followup_hours <= 96


def test_acs_moderate_heart_observation():
    rep = _run(compute_acs_disposition_decision(heart_score_total=5))
    assert rep.disposition == "ed_observation_serial_troponin"


def test_acs_hemodynamic_instability_immediate_cath():
    rep = _run(compute_acs_disposition_decision(
        heart_score_total=4, hemodynamic_instability=True,
    ))
    assert rep.disposition == "cath_lab_activation_immediate"


def test_acs_high_heart_admits():
    rep = _run(compute_acs_disposition_decision(heart_score_total=8))
    assert rep.disposition == "admit_telemetry_for_workup"


def test_acs_ongoing_pain_in_low_heart_admits():
    """High-risk feature in a numerically low HEART pulls toward admission."""
    rep = _run(compute_acs_disposition_decision(
        heart_score_total=3, ongoing_chest_pain=True,
    ))
    assert rep.disposition == "admit_telemetry_for_workup"
