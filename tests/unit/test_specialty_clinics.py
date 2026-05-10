"""Phase 13.4 H1 -- specialty_clinics bundle tests."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.specialty_clinics import (
    compute_diabetic_retinopathy_severity,
    compute_lesion_triage,
    compute_pft_interpretation,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Lesion triage (ABCDE + 7-point)
# ─────────────────────────────────────────────────────────────────────

def test_lesion_benign_for_clean_lesion():
    out = _run(compute_lesion_triage(diameter_mm=3.0))
    assert out.suspicion_tier == "benign"
    assert out.biopsy_recommended is False


def test_lesion_urgent_referral_when_all_abcde_flagged():
    out = _run(compute_lesion_triage(
        asymmetry=True, border_irregularity=True,
        color_variegated=True, diameter_mm=8.0,
        evolving_or_changing=True,
        seven_point_atypical_features=2,
        family_history_melanoma=True,
    ))
    assert out.suspicion_tier == "urgent_referral"
    assert out.biopsy_recommended is True


def test_lesion_biopsy_at_three_abcde():
    out = _run(compute_lesion_triage(
        asymmetry=True, border_irregularity=True, diameter_mm=7.0,
    ))
    assert out.suspicion_tier == "biopsy_recommended"


def test_lesion_seven_point_3_or_more_triggers_urgent():
    out = _run(compute_lesion_triage(
        seven_point_atypical_features=3,
    ))
    assert out.suspicion_tier == "urgent_referral"


# ─────────────────────────────────────────────────────────────────────
# DR severity
# ─────────────────────────────────────────────────────────────────────

def test_dr_no_dr_when_clean_exam():
    out = _run(compute_diabetic_retinopathy_severity())
    assert out.severity_class == "no_dr"
    assert out.referral_urgency == "routine"


def test_dr_pdr_when_neovascularization():
    out = _run(compute_diabetic_retinopathy_severity(
        neovascularization=True,
    ))
    assert out.severity_class == "pdr"
    assert out.referral_urgency == "urgent"


def test_dr_severe_npdr_when_imatypia_or_venous_beading():
    out = _run(compute_diabetic_retinopathy_severity(
        venous_beading=True,
    ))
    assert out.severity_class == "severe_npdr"


def test_dr_macular_oedema_flagged():
    out = _run(compute_diabetic_retinopathy_severity(
        microaneurysms=True, macular_thickening=True,
    ))
    assert out.macular_oedema_present is True


# ─────────────────────────────────────────────────────────────────────
# PFT interpretation
# ─────────────────────────────────────────────────────────────────────

def test_pft_normal_pattern_at_high_ratio_and_normal_fev1():
    out = _run(compute_pft_interpretation(
        fev1_l=3.5, fvc_l=4.5, fev1_pct_predicted=95.0,
    ))
    assert out.pattern == "normal"
    assert out.gold_stage == "none"


def test_pft_obstructive_at_low_ratio():
    out = _run(compute_pft_interpretation(
        fev1_l=1.8, fvc_l=3.0, fev1_pct_predicted=55.0,
    ))
    assert out.pattern == "obstructive"
    assert out.gold_stage == "GOLD_2"


def test_pft_gold_4_at_severe_obstruction():
    out = _run(compute_pft_interpretation(
        fev1_l=0.6, fvc_l=2.0, fev1_pct_predicted=25.0,
    ))
    assert out.gold_stage == "GOLD_4"


def test_pft_restrictive_at_normal_ratio_low_fev1():
    out = _run(compute_pft_interpretation(
        fev1_l=2.0, fvc_l=2.5, fev1_pct_predicted=68.0,
    ))
    assert out.pattern == "restrictive"


def test_pft_act_well_controlled_at_score_22():
    out = _run(compute_pft_interpretation(
        fev1_l=2.5, fvc_l=3.0, fev1_pct_predicted=85.0,
        asthma_control_test_score=22,
    ))
    assert out.asthma_control_tier == "well_controlled"


def test_pft_act_very_poorly_at_score_12():
    out = _run(compute_pft_interpretation(
        fev1_l=2.0, fvc_l=3.0, fev1_pct_predicted=72.0,
        asthma_control_test_score=12,
    ))
    assert out.asthma_control_tier == "very_poorly_controlled"


def test_pft_rejects_zero_fvc():
    with pytest.raises(ValueError):
        _run(compute_pft_interpretation(
            fev1_l=2.0, fvc_l=0, fev1_pct_predicted=85,
        ))


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_specialty_clinics_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "specialty_clinics" in BUNDLES
    assert set(BUNDLES["specialty_clinics"]) == {
        "compute_lesion_triage",
        "compute_diabetic_retinopathy_severity",
        "compute_pft_interpretation",
    }
