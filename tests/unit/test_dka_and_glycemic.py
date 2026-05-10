"""Unit tests for dka_severity + inpatient_glycemic_control."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.dka_severity import compute_dka_severity
from mcp_server.tools.inpatient_glycemic_control import compute_inpatient_glycemic_control


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── DKA severity ───────────────────────

def test_severe_dka_with_low_ph():
    rep = _run(compute_dka_severity(
        ph=6.95, bicarbonate_meq_l=8, glucose_mg_dl=520, ketones_present=True,
        mental_status="stupor", anion_gap=32, potassium_meq_l=4.0,
    ))
    assert rep.severity == "severe"
    assert rep.icu_admission_indicated is True
    # ADA: bicarbonate only at pH < 6.9; 6.95 is just above -> not indicated
    assert rep.bicarbonate_indicated is False


def test_moderate_dka():
    rep = _run(compute_dka_severity(
        ph=7.10, bicarbonate_meq_l=12, glucose_mg_dl=380, ketones_present=True,
        mental_status="alert", potassium_meq_l=4.0,
    ))
    assert rep.severity == "moderate"


def test_mild_dka_outpatient_eligible():
    rep = _run(compute_dka_severity(
        ph=7.28, bicarbonate_meq_l=16, glucose_mg_dl=320, ketones_present=True,
        mental_status="alert", potassium_meq_l=4.0,
    ))
    assert rep.severity == "mild"
    assert rep.icu_admission_indicated is False


def test_not_dka_when_ph_normal():
    rep = _run(compute_dka_severity(
        ph=7.35, bicarbonate_meq_l=22, glucose_mg_dl=180, ketones_present=False,
    ))
    assert rep.severity == "not_dka"


def test_low_potassium_holds_insulin():
    rep = _run(compute_dka_severity(
        ph=6.95, bicarbonate_meq_l=8, glucose_mg_dl=520, ketones_present=True,
        mental_status="stupor", potassium_meq_l=3.0,
    ))
    assert rep.potassium_replacement_at_initiation is True
    assert "HOLD insulin" in rep.insulin_protocol


def test_acidosis_without_ketones_abstains():
    rep = _run(compute_dka_severity(
        ph=7.10, bicarbonate_meq_l=10, glucose_mg_dl=200, ketones_present=False,
        mental_status="alert",
    ))
    assert rep.abstain_recommended is True
    assert "non-DKA" in (rep.abstain_reason or "") or "lactic" in (rep.abstain_reason or "").lower()


# ─────────────────────── Inpatient glycemic ───────────────────────

def test_target_icu_vs_ward():
    icu = _run(compute_inpatient_glycemic_control(is_icu=True))
    ward = _run(compute_inpatient_glycemic_control(is_icu=False))
    assert icu.target_range_mg_dl == (140, 180)
    assert ward.target_range_mg_dl == (100, 180)


def test_hypoglycemia_decreases_basal():
    rep = _run(compute_inpatient_glycemic_control(
        is_icu=False, average_glucose_24h=130,
        n_hypoglycemic_episodes_24h=1,
        current_basal_total_units=20,
    ))
    assert rep.basal_dose_change_pct < 0


def test_severe_hyperglycemia_increases_basal():
    rep = _run(compute_inpatient_glycemic_control(
        is_icu=False, average_glucose_24h=265,
        n_severe_hyperglycemic_episodes_24h=2,
        current_basal_total_units=20,
    ))
    assert rep.basal_dose_change_pct > 0
    assert rep.correctional_scale_change is not None


def test_high_hypoglycemia_risk_blocks_increase():
    rep = _run(compute_inpatient_glycemic_control(
        is_icu=False, average_glucose_24h=200,
        current_basal_total_units=20,
        risk_factors={
            "age": 80, "egfr_ml_min": 40,
            "recent_npo_or_diet_change": True, "heart_failure": True,
            "sulfonylurea_use": True,
        },
    ))
    assert rep.hypoglycemia_risk == "high"
    assert rep.abstain_recommended is True


def test_sliding_scale_only_recommendation_includes_basal_bolus():
    rep = _run(compute_inpatient_glycemic_control(
        is_icu=False, average_glucose_24h=210,
        current_regimen="sliding_scale_only", current_basal_total_units=0,
    ))
    assert "basal-bolus" in rep.recommended_regimen.lower() or "basal" in rep.recommended_regimen.lower()


def test_recurrent_hypoglycemia_aggressive_decrease():
    rep = _run(compute_inpatient_glycemic_control(
        is_icu=False, average_glucose_24h=130,
        n_hypoglycemic_episodes_24h=3,
        current_basal_total_units=30,
    ))
    assert rep.basal_dose_change_pct <= -20.0
