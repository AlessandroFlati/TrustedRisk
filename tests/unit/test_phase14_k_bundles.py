"""Phase 14 K1-K6 -- clinical depth bundles."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.rheumatology import (
    compute_acr_eular_ra_classification, compute_asdas_axspa,
    compute_das28_rheumatoid_arthritis,
)
from mcp_server.tools.peri_op_risk import (
    compute_ariscat_pulmonary_risk, compute_caprini_vte_risk,
    compute_rcri_cardiac_risk,
)
from mcp_server.tools.infectious_disease import (
    compute_hiv_management_tier, compute_lactate_clearance,
    compute_qsofa_score, compute_tb_risk_screen,
)
from mcp_server.tools.gi_hepatology_depth import (
    compute_fib4_liver_fibrosis,
    compute_glasgow_blatchford_ugib,
    compute_maddrey_alcoholic_hepatitis, compute_rome_iv_ibs,
)
from mcp_server.tools.neurology_depth import (
    compute_hauser_ambulation_index, compute_hunt_hess_sah,
    compute_ich_score, compute_modified_rankin,
)
from mcp_server.tools.ob_peds_advanced import (
    compute_apgar_score, compute_bell_nec_stage,
    compute_bilirubin_nomogram, compute_bishop_induction_score,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# K6 Rheumatology
# ─────────────────────────────────────────────────────────────────────

def test_das28_remission_at_low_inputs():
    out = _run(compute_das28_rheumatoid_arthritis(
        tender_joint_count_28=0, swollen_joint_count_28=0,
        esr_mm_per_hour=10, patient_global_assessment_vas_100=10,
    ))
    assert out.activity_tier == "remission"


def test_das28_high_at_severe_disease():
    out = _run(compute_das28_rheumatoid_arthritis(
        tender_joint_count_28=20, swollen_joint_count_28=15,
        esr_mm_per_hour=80, patient_global_assessment_vas_100=80,
    ))
    assert out.activity_tier == "high"
    assert out.biologic_eligibility_threshold_met is True


def test_asdas_inactive_at_low_disease():
    out = _run(compute_asdas_axspa(
        back_pain_vas_0_10=1, duration_morning_stiffness_vas_0_10=1,
        patient_global_vas_0_10=1, peripheral_pain_swelling_vas_0_10=0,
        crp_mg_l=2,
    ))
    assert out.activity_tier == "inactive"


def test_acr_eular_ra_classification_met_at_score_6():
    out = _run(compute_acr_eular_ra_classification(
        n_small_joints_involved=4, n_large_joints_involved=2,
        rf_or_acpa_positive=True, rf_or_acpa_high_titre=False,
        elevated_crp_or_esr=True, symptom_duration_weeks_ge_6=True,
    ))
    assert out.classification_met is True
    assert out.total_score >= 6


# ─────────────────────────────────────────────────────────────────────
# K4 Peri-op risk
# ─────────────────────────────────────────────────────────────────────

def test_rcri_low_for_clean_profile():
    out = _run(compute_rcri_cardiac_risk())
    assert out.risk_tier == "low"
    assert out.cardiac_risk_pct < 1.0


def test_rcri_high_at_3_plus_factors():
    out = _run(compute_rcri_cardiac_risk(
        high_risk_surgery=True,
        history_ischemic_heart_disease=True,
        history_congestive_heart_failure=True,
    ))
    assert out.risk_tier == "high"


def test_ariscat_high_at_emergency_thoracic():
    out = _run(compute_ariscat_pulmonary_risk(
        age=82, preop_spo2_pct=90,
        surgical_incision_site="intrathoracic",
        surgical_duration_hours=4,
        respiratory_infection_last_month=True,
        emergency_surgery=True,
    ))
    assert out.risk_tier == "high"


def test_caprini_highest_for_oncology_with_history():
    out = _run(compute_caprini_vte_risk(
        age=80, bmi_gt_25=True,
        major_surgery_planned=True,
        active_malignancy=True, history_vte=True,
    ))
    assert out.vte_risk_tier == "highest"


# ─────────────────────────────────────────────────────────────────────
# K1 ID
# ─────────────────────────────────────────────────────────────────────

def test_qsofa_likely_at_score_2():
    out = _run(compute_qsofa_score(
        altered_mentation_gcs_lt_15=True, respiratory_rate_ge_22=True,
    ))
    assert out.score == 2
    assert out.sepsis_likely is True


def test_lactate_clearance_target_met_at_15_pct():
    out = _run(compute_lactate_clearance(
        initial_lactate_mmol_l=4.0, repeat_lactate_mmol_l=3.4,
    ))
    assert out.target_met is True
    assert out.clearance_pct == pytest.approx(15.0, abs=0.5)


def test_hiv_aids_tier_at_low_cd4_high_vl_no_art():
    out = _run(compute_hiv_management_tier(
        cd4_count=80, viral_load_copies_ml=50_000, on_art=False,
    ))
    assert out.tier == "AIDS_defining_immune_failure"
    assert "TMP-SMX" in " ".join(out.opportunistic_prophylaxis_indicated)


def test_tb_high_risk_with_active_symptoms():
    out = _run(compute_tb_risk_screen(
        cough_gt_3_weeks_with_constitutional_symptoms=True,
    ))
    assert out.risk_tier == "high"
    assert out.recommended_test == "imaging+sputum"


# ─────────────────────────────────────────────────────────────────────
# K2 GI
# ─────────────────────────────────────────────────────────────────────

def test_maddrey_severe_indicates_steroid():
    out = _run(compute_maddrey_alcoholic_hepatitis(
        patient_pt_seconds=20, control_pt_seconds=12,
        serum_bilirubin_mg_dl=10,
    ))
    assert out.severity_tier == "severe"
    assert out.corticosteroid_indicated is True


def test_fib4_minimal_at_low_index():
    out = _run(compute_fib4_liver_fibrosis(
        age=40, ast_iu_l=20, alt_iu_l=25,
        platelets_thousands_per_uL=250,
    ))
    assert out.fibrosis_stage_estimate == "F0_F1_minimal"


def test_fib4_advanced_at_high_index():
    out = _run(compute_fib4_liver_fibrosis(
        age=70, ast_iu_l=120, alt_iu_l=80,
        platelets_thousands_per_uL=110,
    ))
    assert out.fibrosis_stage_estimate == "F3_F4_advanced"


def test_gbs_zero_for_dischargeable():
    out = _run(compute_glasgow_blatchford_ugib(
        blood_urea_mmol_l=4.0, hemoglobin_g_dl=14, sex="male",
        systolic_bp_mmHg=120,
    ))
    assert out.score == 0
    assert out.risk_tier == "very_low"


def test_gbs_high_with_shock():
    out = _run(compute_glasgow_blatchford_ugib(
        blood_urea_mmol_l=30, hemoglobin_g_dl=8,
        sex="male", systolic_bp_mmHg=85,
        pulse_ge_100=True, melena=True, syncope=True,
        hepatic_disease=True,
    ))
    assert out.risk_tier == "high"
    assert out.endoscopy_within_24h is True


def test_rome_iv_ibs_met_with_full_criteria():
    out = _run(compute_rome_iv_ibs(
        abdominal_pain_days_per_week=2,
        related_to_defecation=True,
        associated_with_change_in_stool_frequency=True,
        associated_with_change_in_stool_form=True,
        symptom_duration_months=8,
        bowel_pattern_predominant="mixed",
    ))
    assert out.criteria_met is True
    assert out.subtype == "IBS_M"


# ─────────────────────────────────────────────────────────────────────
# K3 Neurology
# ─────────────────────────────────────────────────────────────────────

def test_hunt_hess_grade_5_poor_outcome():
    out = _run(compute_hunt_hess_sah(grade=5))
    assert out.surgical_eligibility == "poor_candidate"
    assert out.estimated_mortality_pct >= 95


def test_ich_high_score_high_mortality():
    out = _run(compute_ich_score(
        glasgow_coma_scale=4, ich_volume_ml=40,
        intraventricular_hemorrhage=True,
        infratentorial_origin=True, age_ge_80=True,
    ))
    assert out.score == 6
    assert out.estimated_30d_mortality_pct == 100.0


def test_modified_rankin_3_loses_independence():
    out = _run(compute_modified_rankin(grade=3))
    assert out.independent_living is False


def test_hauser_grade_0_normal():
    out = _run(compute_hauser_ambulation_index(grade=0))
    assert out.grade == 0


# ─────────────────────────────────────────────────────────────────────
# K5 OB / peds
# ─────────────────────────────────────────────────────────────────────

def test_bishop_favourable_at_dilated_anterior_soft():
    out = _run(compute_bishop_induction_score(
        cervical_dilation_cm=4, cervical_effacement_pct=80,
        fetal_station=0, cervical_consistency="soft",
        cervical_position="anterior",
    ))
    assert out.favourable_for_induction is True


def test_apgar_severely_depressed_at_low_score():
    out = _run(compute_apgar_score(
        one_min_appearance=0, one_min_pulse=1,
        one_min_grimace=0, one_min_activity=0, one_min_respiration=1,
        five_min_appearance=1, five_min_pulse=1,
        five_min_grimace=0, five_min_activity=0, five_min_respiration=1,
    ))
    assert out.severity == "severely_depressed"
    assert out.nicu_evaluation_indicated is True


def test_bell_iiib_for_pneumoperitoneum_with_dic():
    out = _run(compute_bell_nec_stage(
        pneumoperitoneum=True, septic_shock_or_dic=True,
    ))
    assert out.stage == "IIIB"
    assert out.surgical_consult_required is True


def test_bilirubin_high_zone_triggers_phototherapy():
    out = _run(compute_bilirubin_nomogram(
        age_hours=48, total_bilirubin_mg_dl=18,
    ))
    assert out.risk_zone == "high"
    assert out.phototherapy_indicated is True


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_all_six_k_bundles_registered():
    from mcp_server.tools import BUNDLES
    for b in (
        "rheumatology", "peri_op_risk", "infectious_disease",
        "gi_hepatology_depth", "neurology_depth", "ob_peds_advanced",
    ):
        assert b in BUNDLES, f"missing {b!r}"
        assert len(BUNDLES[b]) >= 3
