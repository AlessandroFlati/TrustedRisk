"""Phase 13.3 H2 -- Critical care + hepatology severity bundle tests."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.critical_care import (
    compute_apache_ii_score,
    compute_meld_score,
    compute_rifle_aki_classification,
    compute_sofa_score,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# APACHE II
# ─────────────────────────────────────────────────────────────────────

def test_apache_ii_healthy_young_returns_low_score():
    out = _run(compute_apache_ii_score(
        age=30, temperature_c=37.0,
        mean_arterial_pressure_mmHg=85, heart_rate=75,
        respiratory_rate=16, fio2=0.21, pao2=95,
        arterial_ph=7.40, serum_sodium_mmol_l=140,
        serum_potassium_mmol_l=4.0, serum_creatinine_mg_dl=1.0,
        hematocrit_pct=42, wbc_thousands_per_uL=7.0,
        glasgow_coma_scale=15,
    ))
    assert out.apache_ii_total <= 4
    assert out.severity_tier == "mild"
    assert not out.icu_admission_recommended


def test_apache_ii_critical_patient_recommends_icu():
    out = _run(compute_apache_ii_score(
        age=78, temperature_c=39.5,
        mean_arterial_pressure_mmHg=55, heart_rate=145,
        respiratory_rate=32, fio2=0.6, pao2=58,
        arterial_ph=7.20, serum_sodium_mmol_l=152,
        serum_potassium_mmol_l=5.8, serum_creatinine_mg_dl=2.5,
        hematocrit_pct=28, wbc_thousands_per_uL=22,
        glasgow_coma_scale=8,
        acute_renal_failure=True, chronic_health_severe=True,
    ))
    assert out.severity_tier in ("severe", "critical")
    assert out.icu_admission_recommended is True
    assert out.predicted_mortality_pct > 30


def test_apache_ii_age_points_climb_with_age():
    base = dict(
        temperature_c=37, mean_arterial_pressure_mmHg=85, heart_rate=75,
        respiratory_rate=16, fio2=0.21, pao2=95, arterial_ph=7.4,
        serum_sodium_mmol_l=140, serum_potassium_mmol_l=4.0,
        serum_creatinine_mg_dl=1.0, hematocrit_pct=42,
        wbc_thousands_per_uL=7.0,
    )
    young = _run(compute_apache_ii_score(age=30, **base))
    old = _run(compute_apache_ii_score(age=80, **base))
    assert old.age_points > young.age_points


# ─────────────────────────────────────────────────────────────────────
# SOFA
# ─────────────────────────────────────────────────────────────────────

def test_sofa_healthy_returns_zero_total():
    out = _run(compute_sofa_score(
        pao2_fio2_ratio=420, mechanical_ventilation=False,
        platelets_thousands_per_uL=200, bilirubin_mg_dl=0.8,
        mean_arterial_pressure_mmHg=85, glasgow_coma_scale=15,
        creatinine_mg_dl=0.9,
    ))
    assert out.sofa_total == 0
    assert out.estimated_mortality_pct == 0.0


def test_sofa_critical_organ_failure():
    out = _run(compute_sofa_score(
        pao2_fio2_ratio=80, mechanical_ventilation=True,
        platelets_thousands_per_uL=18, bilirubin_mg_dl=14.0,
        mean_arterial_pressure_mmHg=55,
        pressors_doses={"norepinephrine_mcg_kg_min": 0.5},
        glasgow_coma_scale=5, creatinine_mg_dl=5.5,
    ))
    assert out.sofa_total >= 18
    assert out.estimated_mortality_pct > 50


def test_sofa_sepsis3_dysfunction_when_delta_at_least_2():
    out = _run(compute_sofa_score(
        pao2_fio2_ratio=250, mechanical_ventilation=False,
        platelets_thousands_per_uL=120, bilirubin_mg_dl=2.5,
        mean_arterial_pressure_mmHg=85, glasgow_coma_scale=14,
        creatinine_mg_dl=1.4,
        baseline_sofa_total=0, suspected_infection=True,
    ))
    assert out.sepsis_3_dysfunction is True


def test_sofa_no_sepsis3_without_infection():
    out = _run(compute_sofa_score(
        pao2_fio2_ratio=250, mechanical_ventilation=False,
        platelets_thousands_per_uL=120, bilirubin_mg_dl=2.5,
        mean_arterial_pressure_mmHg=85, glasgow_coma_scale=14,
        creatinine_mg_dl=1.4,
        baseline_sofa_total=0, suspected_infection=False,
    ))
    assert out.sepsis_3_dysfunction is False


# ─────────────────────────────────────────────────────────────────────
# MELD
# ─────────────────────────────────────────────────────────────────────

def test_meld_low_score_for_compensated_cirrhosis():
    out = _run(compute_meld_score(
        bilirubin_mg_dl=1.0, creatinine_mg_dl=0.9, inr=1.1,
        sodium_mmol_l=140,
    ))
    assert out.meld_classic <= 12
    assert out.severity_tier in ("low", "moderate")
    assert out.transplant_eligibility_threshold_met is False


def test_meld_high_score_unlocks_transplant_listing():
    out = _run(compute_meld_score(
        bilirubin_mg_dl=8.0, creatinine_mg_dl=2.5, inr=2.5,
        sodium_mmol_l=128,
    ))
    assert out.meld_classic >= 25
    assert out.transplant_eligibility_threshold_met is True
    assert out.estimated_3mo_mortality_pct > 30


def test_meld_dialysis_caps_creatinine_at_4():
    """Per Kamath 2001: patients on dialysis or with creatinine > 4 are
    capped at 4 mg/dL."""
    on = _run(compute_meld_score(
        bilirubin_mg_dl=2.0, creatinine_mg_dl=8.0, inr=1.2,
        on_dialysis=True,
    ))
    capped = _run(compute_meld_score(
        bilirubin_mg_dl=2.0, creatinine_mg_dl=4.0, inr=1.2,
    ))
    assert on.meld_classic == capped.meld_classic


def test_meld_na_diverges_from_classic_when_hyponatraemia():
    out = _run(compute_meld_score(
        bilirubin_mg_dl=3.0, creatinine_mg_dl=1.5, inr=1.6,
        sodium_mmol_l=128,
    ))
    assert out.meld_na is not None
    assert out.meld_na > out.meld_classic


# ─────────────────────────────────────────────────────────────────────
# RIFLE
# ─────────────────────────────────────────────────────────────────────

def test_rifle_no_aki_for_stable_creatinine():
    out = _run(compute_rifle_aki_classification(
        serum_creatinine_baseline_mg_dl=1.0,
        serum_creatinine_current_mg_dl=1.05,
    ))
    assert out.rifle_class == "no_aki"
    assert out.nephrology_consult_recommended is False


def test_rifle_risk_at_1_5x_creatinine_increase():
    out = _run(compute_rifle_aki_classification(
        serum_creatinine_baseline_mg_dl=1.0,
        serum_creatinine_current_mg_dl=1.6,
    ))
    assert out.rifle_class == "risk"


def test_rifle_injury_at_2x_creatinine_increase():
    out = _run(compute_rifle_aki_classification(
        serum_creatinine_baseline_mg_dl=1.0,
        serum_creatinine_current_mg_dl=2.2,
    ))
    assert out.rifle_class == "injury"
    assert out.nephrology_consult_recommended is True


def test_rifle_failure_at_3x_or_creatinine_4():
    out = _run(compute_rifle_aki_classification(
        serum_creatinine_baseline_mg_dl=1.0,
        serum_creatinine_current_mg_dl=4.5,
    ))
    assert out.rifle_class == "failure"


def test_rifle_etiology_clue_from_history_keywords():
    out = _run(compute_rifle_aki_classification(
        serum_creatinine_baseline_mg_dl=1.0,
        serum_creatinine_current_mg_dl=2.0,
        history_keywords="Patient received iodinated contrast 24h ago.",
    ))
    assert "Contrast" in out.aki_etiology_clue


def test_rifle_rejects_zero_baseline():
    with pytest.raises(ValueError):
        _run(compute_rifle_aki_classification(
            serum_creatinine_baseline_mg_dl=0,
            serum_creatinine_current_mg_dl=2.0,
        ))


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_critical_care_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "critical_care" in BUNDLES
    assert set(BUNDLES["critical_care"]) == {
        "compute_apache_ii_score", "compute_sofa_score",
        "compute_meld_score", "compute_rifle_aki_classification",
    }
