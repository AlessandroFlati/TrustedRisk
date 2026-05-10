"""Unit tests for SAFE-2 bundle composition orchestrator."""
from __future__ import annotations

import pytest

from a2a_agent.bundle_orchestrator import suggest_bundles


# ─────────────────────── Single-bundle primary selection ───────────────────────

def test_chest_pain_picks_stroke_acs():
    res = suggest_bundles(chief_complaint="58yo M with crushing chest pain")
    assert res.primary_bundle == "stroke_acs"


def test_stroke_signs_picks_stroke_acs():
    res = suggest_bundles(
        chief_complaint="sudden facial droop and slurred speech, hemiparesis",
    )
    assert res.primary_bundle == "stroke_acs"


def test_polytrauma_picks_trauma():
    res = suggest_bundles(
        free_text_summary="32yo M MVC, FAST positive, hemorrhage from splenic laceration",
    )
    assert res.primary_bundle == "trauma_critical"


def test_pediatric_age_picks_pediatric():
    res = suggest_bundles(
        chief_complaint="lethargic infant with fever",
        structured_features={"age": 4},  # months OR years <18
    )
    assert res.primary_bundle == "pediatric"


def test_preeclampsia_picks_obstetric():
    res = suggest_bundles(
        chief_complaint="severe preeclampsia, BP 165/115, G1P0 32 weeks",
    )
    assert res.primary_bundle == "obstetric_geriatric"


def test_dka_picks_endocrine_acute():
    res = suggest_bundles(
        chief_complaint="DKA with glucose 540 and ketoacidosis",
    )
    assert res.primary_bundle == "endocrine_acute"


def test_aki_picks_nephrology():
    res = suggest_bundles(
        chief_complaint="AKI stage 3 requiring dialysis evaluation",
    )
    assert res.primary_bundle == "nephrology"


def test_chemo_picks_oncology():
    res = suggest_bundles(
        chief_complaint="NSCLC patient, cycle 4 carboplatin/pemetrexed",
        structured_features={"on_chemotherapy": True},
    )
    assert res.primary_bundle == "oncology"


def test_suicide_risk_picks_mental_health():
    res = suggest_bundles(
        chief_complaint="active suicidal ideation, prior attempt",
        structured_features={"suicide_ideation_present": True},
    )
    assert res.primary_bundle == "mental_health"


def test_delirium_picks_obstetric_geriatric():
    res = suggest_bundles(
        chief_complaint="85yo with fluctuating delirium, recent fall",
    )
    assert res.primary_bundle == "obstetric_geriatric"


def test_sepsis_picks_antimicrobial():
    res = suggest_bundles(
        chief_complaint="complicated UTI with sepsis, blood cultures pending",
    )
    assert res.primary_bundle == "antimicrobial"


def test_no_signal_falls_back_to_core_discharge():
    res = suggest_bundles(chief_complaint="discharge planning for stable patient")
    assert res.primary_bundle == "core_discharge"


# ─────────────────────── Cross-bundle add-on inference ───────────────────────

def test_trauma_adds_ed_acute():
    res = suggest_bundles(
        free_text_summary="polytrauma after MVC, FAST positive, GCS 10",
    )
    assert res.primary_bundle == "trauma_critical"
    # ed_acute should appear in add-ons or candidates due to cross-bundle bonus
    bundle_ids = [s.bundle_id for s in res.ranked]
    assert "ed_acute" in bundle_ids


def test_aki_with_contrast_planned_adds_imaging():
    res = suggest_bundles(
        chief_complaint="AKI patient with contrast CT planned",
        structured_features={"egfr_ml_min": 25,
                                "contrast_imaging_planned": True},
    )
    assert res.primary_bundle == "nephrology"
    add_ons = res.add_on_bundles
    assert "imaging" in add_ons or any(
        s.bundle_id == "imaging" and s.score >= 1.0 for s in res.ranked
    )


def test_stroke_adds_imaging():
    res = suggest_bundles(
        chief_complaint="acute stroke, NIHSS 12, LVO suspected",
    )
    assert res.primary_bundle == "stroke_acs"
    bundle_ids = [s.bundle_id for s in res.ranked]
    assert "imaging" in bundle_ids


# ─────────────────────── Ranking + score behavior ───────────────────────

def test_ranked_in_score_order():
    res = suggest_bundles(
        free_text_summary="stroke patient with AKI and contrast CT decision needed",
    )
    scores = [s.score for s in res.ranked]
    assert scores == sorted(scores, reverse=True)


def test_primary_has_role_primary():
    res = suggest_bundles(chief_complaint="DKA glucose 600")
    primary_entry = next(s for s in res.ranked if s.bundle_id == res.primary_bundle)
    assert primary_entry.role == "primary"


def test_add_ons_capped_at_three():
    """The add_on_bundles list should never exceed 3 entries."""
    res = suggest_bundles(
        free_text_summary=(
            "polytrauma MVC AKI dialysis preeclampsia chemo NSCLC stroke "
            "DKA suicide ideation chest pain"
        ),
    )
    assert len(res.add_on_bundles) <= 3


def test_keyword_signals_recorded():
    res = suggest_bundles(chief_complaint="acute stroke with NIHSS 12")
    assert "stroke_acs" in res.keyword_signals
    matched = res.keyword_signals["stroke_acs"]
    assert any("stroke" in m.lower() or "nihss" in m.lower() for m in matched)


def test_structured_features_only_no_text():
    """Pure structured features should still produce a sensible ranking."""
    res = suggest_bundles(
        structured_features={"age": 8, "egfr_ml_min": 90},
    )
    assert res.primary_bundle == "pediatric"


def test_pregnancy_flag_alone_picks_obstetric():
    res = suggest_bundles(
        structured_features={"gestational_age_weeks": 30, "pregnant": True},
    )
    assert res.primary_bundle == "obstetric_geriatric"


# ─────────────────────── Edge cases ───────────────────────

def test_empty_input_falls_back():
    res = suggest_bundles()
    assert res.primary_bundle == "core_discharge"


def test_unicode_in_summary_no_crash():
    res = suggest_bundles(
        chief_complaint="患者 chest pain 急 \U0001F480",
    )
    assert res.primary_bundle in ("stroke_acs", "core_discharge")


def test_summary_string_format():
    res = suggest_bundles(chief_complaint="DKA glucose 540")
    assert "Primary bundle" in res.summary
    assert "endocrine_acute" in res.summary
