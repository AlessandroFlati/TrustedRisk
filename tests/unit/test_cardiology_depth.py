"""Phase 13.5 H4 -- cardiology_depth bundle tests."""

from __future__ import annotations

import asyncio

from mcp_server.tools.cardiology_depth import (
    compute_cha2ds2_vasc, compute_grace_acs_score,
    compute_has_bled, compute_timi_acs_score,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# CHA2DS2-VASc
# ─────────────────────────────────────────────────────────────────────

def test_cha2ds2_zero_risk_for_young_male_no_comorbidities():
    out = _run(compute_cha2ds2_vasc(age=45))
    assert out.score == 0
    assert out.anticoagulation_recommendation == "no_anticoagulation"


def test_cha2ds2_anticoagulation_recommended_score_2():
    out = _run(compute_cha2ds2_vasc(
        age=70, hypertension=True, diabetes_mellitus=True,
    ))
    assert out.score >= 3
    assert out.anticoagulation_recommendation == "anticoagulation_recommended"


def test_cha2ds2_age_75_adds_two_points():
    young = _run(compute_cha2ds2_vasc(age=64))
    old = _run(compute_cha2ds2_vasc(age=80))
    assert old.score - young.score == 2


def test_cha2ds2_score_capped_at_9():
    out = _run(compute_cha2ds2_vasc(
        age=80, sex_female=True,
        congestive_heart_failure=True, hypertension=True,
        diabetes_mellitus=True,
        stroke_tia_thromboembolism_history=True,
        vascular_disease=True,
    ))
    assert out.score == 9


# ─────────────────────────────────────────────────────────────────────
# HAS-BLED
# ─────────────────────────────────────────────────────────────────────

def test_has_bled_low_for_clean_profile():
    out = _run(compute_has_bled())
    assert out.score == 0
    assert out.bleeding_tier == "low"
    assert out.modifiable_factors == []


def test_has_bled_high_for_score_3_plus():
    out = _run(compute_has_bled(
        hypertension_uncontrolled_sbp_gt_160=True,
        abnormal_renal_function=True,
        labile_inr=True,
        drugs_concomitant_antiplatelet_or_nsaid=True,
    ))
    assert out.score == 4
    assert out.bleeding_tier == "high"
    assert "Tighten BP control" in " ".join(out.modifiable_factors)


# ─────────────────────────────────────────────────────────────────────
# TIMI ACS
# ─────────────────────────────────────────────────────────────────────

def test_timi_low_risk_for_clean_profile():
    out = _run(compute_timi_acs_score())
    assert out.score == 0
    assert out.risk_tier == "low"


def test_timi_high_risk_with_5_plus_features():
    out = _run(compute_timi_acs_score(
        age_ge_65=True, three_or_more_cad_risk_factors=True,
        st_deviation_ge_0_5_mm=True,
        elevated_cardiac_markers=True,
        severe_anginal_episodes_in_last_24h=True,
    ))
    assert out.score == 5
    assert out.risk_tier == "high"


# ─────────────────────────────────────────────────────────────────────
# GRACE ACS
# ─────────────────────────────────────────────────────────────────────

def test_grace_low_risk_for_young_stable():
    out = _run(compute_grace_acs_score(
        age=50, heart_rate=72, systolic_bp=130,
        creatinine_mg_dl=0.9, killip_class=1,
    ))
    assert out.risk_tier == "low"
    assert out.estimated_in_hospital_mortality_pct < 5.0


def test_grace_high_risk_for_elderly_unstable():
    out = _run(compute_grace_acs_score(
        age=85, heart_rate=130, systolic_bp=80,
        creatinine_mg_dl=2.5, killip_class=4,
        cardiac_arrest_at_admission=True,
        st_segment_deviation=True,
        elevated_cardiac_enzymes=True,
    ))
    assert out.risk_tier == "high"
    assert out.estimated_in_hospital_mortality_pct >= 11.0


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_cardiology_depth_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "cardiology_depth" in BUNDLES
    assert set(BUNDLES["cardiology_depth"]) == {
        "compute_cha2ds2_vasc", "compute_has_bled",
        "compute_timi_acs_score", "compute_grace_acs_score",
    }
