"""Phase 13.6/13.7/13.8/13.9 -- heme_onc + endocrinology + sleep_pain + transplant."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.heme_onc_depth import (
    compute_ecog_performance_status, compute_ipss_r_mds_score,
    compute_iss_myeloma_staging, compute_karnofsky_performance,
)
from mcp_server.tools.endocrinology_advanced import (
    compute_adrenal_insufficiency_workup,
    compute_hypocalcemia_severity, compute_thyroid_management,
)
from mcp_server.tools.sleep_pain import (
    compute_dn4_neuropathic_pain, compute_epworth_sleepiness_scale,
    compute_stop_bang_osa_screen,
)
from mcp_server.tools.transplant import (
    compute_epts_recipient_score,
    compute_immunosuppression_dose_check,
    compute_kdpi_kidney_donor,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Heme/Onc
# ─────────────────────────────────────────────────────────────────────

def test_iss_stage_i_for_normal_b2m_albumin():
    out = _run(compute_iss_myeloma_staging(
        serum_beta2_microglobulin_mg_l=2.5, serum_albumin_g_dl=4.0,
    ))
    assert out.iss_stage == "I"


def test_iss_stage_iii_at_high_b2m():
    out = _run(compute_iss_myeloma_staging(
        serum_beta2_microglobulin_mg_l=6.0, serum_albumin_g_dl=3.2,
    ))
    assert out.iss_stage == "III"


def test_ipss_r_very_low_for_clean_profile():
    out = _run(compute_ipss_r_mds_score(
        cytogenetic_category="very_good", bm_blast_pct=1,
        hemoglobin_g_dl=12, platelets_thousands_per_uL=200,
        anc_thousands_per_uL=2.5,
    ))
    assert out.ipss_r_category == "very_low"


def test_ipss_r_high_risk_with_poor_cyto_high_blasts():
    out = _run(compute_ipss_r_mds_score(
        cytogenetic_category="poor", bm_blast_pct=15,
        hemoglobin_g_dl=8.5, platelets_thousands_per_uL=40,
        anc_thousands_per_uL=0.6,
    ))
    assert out.ipss_r_category in ("high", "very_high")


def test_ecog_zero_eligible_for_chemo():
    out = _run(compute_ecog_performance_status(grade=0))
    assert out.chemotherapy_eligibility is True


def test_ecog_grade_3_blocks_chemo():
    out = _run(compute_ecog_performance_status(grade=3))
    assert out.chemotherapy_eligibility is False


def test_karnofsky_100_maps_to_ecog_0():
    out = _run(compute_karnofsky_performance(score=100))
    assert out.ecog_equivalent == 0
    assert out.care_setting == "fully_active"


def test_karnofsky_low_score_maps_to_palliative():
    out = _run(compute_karnofsky_performance(score=20))
    assert out.care_setting == "hospice_or_palliative"


# ─────────────────────────────────────────────────────────────────────
# Endocrinology
# ─────────────────────────────────────────────────────────────────────

def test_thyroid_overt_hypothyroid_at_high_tsh_low_t4():
    out = _run(compute_thyroid_management(
        tsh_mU_L=18.0, free_t4_ng_dl=0.6,
    ))
    assert out.pattern == "overt_hypothyroid"
    assert out.levothyroxine_dose_change_mcg > 0


def test_thyroid_subclinical_hyperthyroid_at_low_tsh_normal_t4():
    out = _run(compute_thyroid_management(
        tsh_mU_L=0.2, free_t4_ng_dl=1.4, on_levothyroxine=True,
    ))
    assert out.pattern == "subclinical_hyperthyroid"
    assert out.levothyroxine_dose_change_mcg < 0


def test_thyroid_euthyroid_no_change():
    out = _run(compute_thyroid_management(tsh_mU_L=2.0, free_t4_ng_dl=1.2))
    assert out.pattern == "euthyroid"
    assert out.levothyroxine_dose_change_mcg == 0


def test_adrenal_rule_out_at_high_morning_cortisol():
    out = _run(compute_adrenal_insufficiency_workup(
        morning_cortisol_ug_dl=22.0,
    ))
    assert out.diagnosis_tier == "rule_out"


def test_adrenal_primary_AI_at_low_cortisol_high_acth():
    out = _run(compute_adrenal_insufficiency_workup(
        morning_cortisol_ug_dl=2.0, acth_pg_ml=300,
    ))
    assert out.diagnosis_tier == "primary_AI"
    assert out.fludrocortisone_recommended is True


def test_hypocalcemia_critical_at_corrected_below_6():
    out = _run(compute_hypocalcemia_severity(
        serum_calcium_mg_dl=5.5, serum_albumin_g_dl=4.0,
    ))
    assert out.severity == "critical"
    assert out.iv_calcium_indicated is True


def test_hypocalcemia_mild_at_albumin_corrected_normal():
    out = _run(compute_hypocalcemia_severity(
        serum_calcium_mg_dl=8.5, serum_albumin_g_dl=4.0,
    ))
    assert out.severity == "mild"


# ─────────────────────────────────────────────────────────────────────
# Sleep + pain
# ─────────────────────────────────────────────────────────────────────

def test_epworth_normal_at_low_score():
    out = _run(compute_epworth_sleepiness_scale())
    assert out.severity == "normal"
    assert out.sleep_study_recommended is False


def test_epworth_severe_at_high_score():
    out = _run(compute_epworth_sleepiness_scale(
        sitting_reading=3, watching_tv=3,
        sitting_inactive_in_public=3, passenger_in_car_one_hour=3,
        lying_down_to_rest_afternoon=3,
        sitting_and_talking_to_someone=2,
        sitting_quietly_after_lunch=3,
        in_car_stopped_in_traffic=2,
    ))
    assert out.severity == "severe_excessive"
    assert out.sleep_study_recommended is True


def test_stop_bang_high_risk_at_score_5():
    out = _run(compute_stop_bang_osa_screen(
        snoring_loudly=True, tired_during_day=True,
        observed_apnea=True, high_blood_pressure=True,
        bmi_gt_35=True,
    ))
    assert out.score == 5
    assert out.risk_tier == "high"


def test_dn4_neuropathic_likely_at_score_4():
    out = _run(compute_dn4_neuropathic_pain(
        burning=True, tingling=True,
        numbness=True, hypoesthesia_to_pinprick=True,
    ))
    assert out.score == 4
    assert out.neuropathic_pain_likely is True
    assert out.suggested_first_line


# ─────────────────────────────────────────────────────────────────────
# Transplant
# ─────────────────────────────────────────────────────────────────────

def test_kdpi_better_than_bottom_quartile_for_young_clean_donor():
    out = _run(compute_kdpi_kidney_donor(age=25))
    assert out.quality_tier in ("top_quartile", "middle_50")
    assert out.kdpi_pct <= 80


def test_kdpi_bottom_quartile_for_old_diabetic_donor():
    out = _run(compute_kdpi_kidney_donor(
        age=68, history_of_hypertension=True,
        history_of_diabetes=True,
        serum_creatinine_mg_dl=2.0,
        donation_after_circulatory_death=True,
    ))
    assert out.quality_tier == "bottom_quartile"


def test_epts_top_20_pct_for_young_recipient():
    out = _run(compute_epts_recipient_score(
        age=22, time_on_dialysis_years=0.5,
    ))
    assert out.top_20_pct_eligible is True


def test_immunosuppression_tacrolimus_increase_for_cyp3a5_intermediate():
    out = _run(compute_immunosuppression_dose_check(
        drug="tacrolimus", weight_kg=70,
        cyp3a5_phenotype="intermediate_metabolizer",
    ))
    assert out.adjusted_dose_mg_per_kg_per_day > out.standard_dose_mg_per_kg_per_day
    assert out.cpic_aware is True


def test_immunosuppression_azathioprine_drop_for_tpmt_poor():
    out = _run(compute_immunosuppression_dose_check(
        drug="azathioprine", weight_kg=70,
        tpmt_phenotype="poor_metabolizer",
    ))
    assert out.adjusted_dose_mg_per_kg_per_day < (
        0.2 * out.standard_dose_mg_per_kg_per_day
    )


def test_immunosuppression_unknown_drug_raises():
    with pytest.raises(ValueError):
        _run(compute_immunosuppression_dose_check(
            drug="aspirin", weight_kg=70,
        ))


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_all_four_phase13_h_bundles_registered():
    from mcp_server.tools import BUNDLES
    for b in (
        "heme_onc_depth", "endocrinology_advanced",
        "sleep_pain", "transplant",
    ):
        assert b in BUNDLES, f"missing bundle {b!r}"
        assert len(BUNDLES[b]) >= 3, (
            f"bundle {b}: only {len(BUNDLES[b])} tool(s)"
        )
