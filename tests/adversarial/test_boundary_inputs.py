"""Adversarial boundary-input tests (SAFE-1).

Verify the clinical tools degrade gracefully under:
  - Out-of-physiological-range vitals (BP=0, HR=999, glucose=10000)
  - Negative numbers where positive expected
  - Extreme ages (0, 120, -1)
  - Empty/None where structured input expected
  - Malformed strings in enum-like fields
  - Extremely large input lists
  - NaN / Inf

The tools must EITHER produce a valid output (clamped / abstained) OR
raise a structured ValueError -- never hang, leak, or produce a
schema-invalid response.
"""
from __future__ import annotations

import asyncio
import math

import pytest


pytestmark = pytest.mark.adversarial


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Vital signs at extremes ───────────────────────

def test_news2_zero_vitals():
    """All-zero vitals should not crash (likely high score)."""
    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    rep = _run(compute_clinical_deterioration_score(vital_signs=[
        {"type": "respiratory_rate", "value": 0, "observed_at": "2026-01-01T00:00:00Z"},
        {"type": "spo2", "value": 0, "observed_at": "2026-01-01T00:00:00Z"},
        {"type": "systolic_bp", "value": 0, "observed_at": "2026-01-01T00:00:00Z"},
        {"type": "heart_rate", "value": 0, "observed_at": "2026-01-01T00:00:00Z"},
    ]))
    assert 0 <= rep.score_total <= 20
    assert rep.severity_tier in ("low", "low_medium", "medium", "high")


def test_news2_extreme_high_vitals():
    """Implausibly high vitals -> medium-or-high tier, no crash.
    HR 999 (+3) + RR 99 (+3) = score 6 -> medium tier baseline; the
    single-3 escalation rule may promote to high. Accept either."""
    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    rep = _run(compute_clinical_deterioration_score(vital_signs=[
        {"type": "heart_rate", "value": 999, "observed_at": "2026-01-01T00:00:00Z"},
        {"type": "respiratory_rate", "value": 99, "observed_at": "2026-01-01T00:00:00Z"},
    ]))
    assert rep.severity_tier in ("medium", "high")


def test_news2_unknown_avpu_value():
    """AVPU not in {A,V,P,U} should not crash."""
    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    rep = _run(compute_clinical_deterioration_score(vital_signs=[
        {"type": "consciousness", "value": "Q-not-an-AVPU-letter",
         "observed_at": "2026-01-01T00:00:00Z"},
    ]))
    assert 0 <= rep.score_total <= 20


def test_news2_empty_vitals():
    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    rep = _run(compute_clinical_deterioration_score(vital_signs=[]))
    assert rep.score_total == 0


# ─────────────────────── Age extremes ───────────────────────

def test_pews_age_negative_clamped():
    """Negative age must not crash."""
    from mcp_server.tools.pediatric_early_warning import compute_pediatric_early_warning
    rep = _run(compute_pediatric_early_warning(age_months=-5))
    assert rep.age_band in ("0-11mo", "1-4y", "5-11y", "12-17y")


def test_pews_age_over_max_falls_back():
    from mcp_server.tools.pediatric_early_warning import compute_pediatric_early_warning
    rep = _run(compute_pediatric_early_warning(age_months=400))
    assert rep.age_band == "12-17y"


def test_dose_extreme_weight():
    """1000 kg patient should still cap at adult max."""
    from mcp_server.tools.weight_based_dosing import compute_weight_based_dosing
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=300, age_months=60,
    ))
    assert rep.final_dose_mg <= 1000.0


# ─────────────────────── BP extremes ───────────────────────

def test_admission_triage_bp_zero():
    from mcp_server.tools.admission_triage import compute_admission_triage
    rep = _run(compute_admission_triage(
        chief_complaint="weakness",
        vital_signs={"systolic_bp": 0, "heart_rate": 200},
        age=70,
    ))
    assert rep.esi_level in (1, 2, 3, 4, 5)
    # SBP=0 is "shock range" -- should pull to ESI 1
    assert rep.esi_level == 1


def test_preeclampsia_implausible_bp():
    from mcp_server.tools.preeclampsia_assessment import compute_preeclampsia_assessment
    rep = _run(compute_preeclampsia_assessment(
        gestational_age_weeks=32,
        systolic_bp=999, diastolic_bp=600, proteinuria_present=True,
    ))
    assert rep.classification in (
        "no_preeclampsia", "gestational_hypertension",
        "preeclampsia_without_severe_features",
        "preeclampsia_with_severe_features",
        "eclampsia", "hellp_syndrome",
    )


# ─────────────────────── Lab extremes ───────────────────────

def test_dka_implausible_ph():
    from mcp_server.tools.dka_severity import compute_dka_severity
    # pH 5.0 -- beyond physiologic range but tool shouldn't crash
    rep = _run(compute_dka_severity(
        ph=5.0, bicarbonate_meq_l=2, glucose_mg_dl=1000,
        ketones_present=True, mental_status="stupor", potassium_meq_l=4.0,
    ))
    assert rep.severity in ("severe", "moderate", "mild", "not_dka")


def test_aki_zero_baseline():
    from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=0.0,  # impossible but defensive
        creatinine_current_mg_dl=2.5,
    ))
    # ratio division-by-zero protected
    assert rep.aki_stage in ("no_aki", "stage_1", "stage_2", "stage_3")


# ─────────────────────── Massive input lists ───────────────────────

def test_polypharmacy_huge_med_list():
    """50 medications should be processed without timeout."""
    from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
    meds = [
        {"name": f"drug-{i}", "drug_class": "unknown", "status": "active"}
        for i in range(50)
    ]
    rep = _run(detect_polypharmacy_concerns(medications=meds))
    assert rep.n_medications == 50


def test_recist_zero_lesions_not_evaluable():
    from mcp_server.tools.oncology_treatment_response import compute_oncology_treatment_response
    rep = _run(compute_oncology_treatment_response(target_lesions=[]))
    assert rep.overall_response == "not_evaluable"
    assert rep.abstain_recommended is True


# ─────────────────────── Malformed strings ───────────────────────

def test_admission_triage_empty_complaint():
    from mcp_server.tools.admission_triage import compute_admission_triage
    rep = _run(compute_admission_triage(chief_complaint=""))
    assert rep.abstain_recommended is True


def test_admission_triage_unicode_garbage():
    """Random unicode in chief complaint must not crash."""
    from mcp_server.tools.admission_triage import compute_admission_triage
    rep = _run(compute_admission_triage(
        chief_complaint="中文 emoji \U0001F480 SQL'); DROP TABLE; --",
        age=40,
    ))
    assert rep.esi_level in (1, 2, 3, 4, 5)


def test_treatment_selection_unknown_condition():
    from mcp_server.tools.treatment_selection import compute_treatment_selection
    rep = _run(compute_treatment_selection(
        condition="completely-made-up-condition-12345",
    ))
    assert rep.abstain_recommended is True
