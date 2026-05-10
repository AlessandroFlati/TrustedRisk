"""Unit tests for imaging_appropriateness + contrast_safety_check."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.imaging_appropriateness import compute_imaging_appropriateness
from mcp_server.tools.contrast_safety_check import compute_contrast_safety_check


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Imaging appropriateness ───────────────────────

def test_high_risk_pe_picks_ctpa():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="suspected_pe_high_risk", patient_age=58,
    ))
    assert rep.top_pick_modality is not None
    assert "pulmonary angiography" in rep.top_pick_modality.lower()


def test_low_risk_pe_picks_ddimer():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="suspected_pe_low_risk", patient_age=40,
    ))
    # D-dimer is the rated-9 first step
    assert rep.top_pick_modality is not None
    assert "d-dimer" in rep.top_pick_modality.lower()


def test_acute_lbp_no_imaging_recommended():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="acute_lbp_no_red_flag", patient_age=45,
    ))
    assert rep.top_pick_modality is not None
    assert "no imaging" in rep.top_pick_modality.lower()


def test_pediatric_appendicitis_picks_us():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="pediatric_appendicitis", patient_age=10,
    ))
    assert rep.top_pick_modality is not None
    assert "ultrasound" in rep.top_pick_modality.lower()
    assert rep.pediatric_alara_caution is True


def test_pediatric_alara_downgrades_ct():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="acute_abdominal_pain", patient_age=8,
    ))
    # CT abd+pelvis (14 mSv) should be downgraded for pediatric
    ct = next((o for o in rep.options if "CT abdomen" in o.modality), None)
    assert ct is not None
    assert ct.appropriateness_rating <= 7  # pediatric ALARA pulled it down


def test_pregnancy_downgrades_iodinated_contrast():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="acute_abdominal_pain", patient_age=28,
        patient_pregnant=True,
    ))
    ct = next((o for o in rep.options if "CT abdomen" in o.modality), None)
    assert ct is not None
    assert ct.appropriateness_rating < 8   # pregnancy pulled it down


def test_unknown_scenario_abstains():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="something_weird",
    ))
    assert rep.abstain_recommended is True


def test_cumulative_radiation_caution():
    rep = _run(compute_imaging_appropriateness(
        clinical_scenario="acute_headache_thunderclap",
        cumulative_radiation_msv_last_12mo=60,
    ))
    assert rep.cumulative_radiation_caution is True


# ─────────────────────── Contrast safety ───────────────────────

def test_iodinated_low_risk_proceeds():
    rep = _run(compute_contrast_safety_check(
        contrast_type="iodinated_iv", egfr_ml_min=80,
    ))
    assert rep.proceed_with_contrast is True
    assert rep.contrast_induced_nephropathy_risk == "low"


def test_iodinated_severe_renal_blocked():
    rep = _run(compute_contrast_safety_check(
        contrast_type="iodinated_iv", egfr_ml_min=20,
    ))
    assert rep.proceed_with_contrast is False
    assert rep.contrast_induced_nephropathy_risk == "contraindicated"


def test_metformin_held_low_egfr():
    rep = _run(compute_contrast_safety_check(
        contrast_type="iodinated_iv", egfr_ml_min=45, on_metformin=True,
    ))
    assert rep.metformin_hold_recommended is True
    assert rep.metformin_hold_duration_hours == 48


def test_metformin_no_hold_normal_egfr():
    rep = _run(compute_contrast_safety_check(
        contrast_type="iodinated_iv", egfr_ml_min=80, on_metformin=True,
    ))
    assert rep.metformin_hold_recommended is False


def test_severe_iodine_reaction_premedicates():
    rep = _run(compute_contrast_safety_check(
        contrast_type="iodinated_iv", egfr_ml_min=80,
        iodine_contrast_prior_severe_reaction=True,
    ))
    assert rep.premedication_recommended is True


def test_gadolinium_low_egfr_blocks():
    rep = _run(compute_contrast_safety_check(
        contrast_type="gadolinium_iv", egfr_ml_min=12,
    ))
    assert rep.proceed_with_contrast is False
    assert rep.nsf_risk_for_gadolinium == "contraindicated"


def test_gadolinium_pregnant_blocks():
    rep = _run(compute_contrast_safety_check(
        contrast_type="gadolinium_iv", egfr_ml_min=80, pregnant=True,
    ))
    assert rep.proceed_with_contrast is False


def test_pre_hydration_flagged():
    rep = _run(compute_contrast_safety_check(
        contrast_type="iodinated_iv", egfr_ml_min=40,
    ))
    assert rep.pre_hydration_recommended is True


def test_no_contrast_always_proceeds():
    rep = _run(compute_contrast_safety_check(contrast_type="no_contrast"))
    assert rep.proceed_with_contrast is True
