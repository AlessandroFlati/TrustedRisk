"""Unit tests for compute_falls_risk_morse + compute_delirium_screening_cam."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.falls_risk_morse import compute_falls_risk_morse
from mcp_server.tools.delirium_screening_cam import compute_delirium_screening_cam


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Morse Falls ───────────────────────

def test_morse_low_risk():
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=False, secondary_diagnosis_present=False,
        ambulatory_aid="none", has_iv_or_heparin_lock=False,
        gait="normal", mental_status="oriented",
    ))
    assert rep.score_total == 0
    assert rep.risk_tier == "low"


def test_morse_high_risk_classic():
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=True,            # 25
        secondary_diagnosis_present=True,        # 15
        ambulatory_aid="walker",                  # 15
        has_iv_or_heparin_lock=True,            # 20
        gait="weak",                             # 10
        mental_status="forgets_limitations",    # 15
    ))
    assert rep.score_total == 100
    assert rep.risk_tier == "high"
    assert rep.recommended_intervention == "high_risk_falls_protocol"


def test_morse_furniture_aid_30():
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=False, secondary_diagnosis_present=False,
        ambulatory_aid="furniture", has_iv_or_heparin_lock=False,
        gait="normal", mental_status="oriented",
    ))
    assert rep.ambulatory_aid_points == 30


def test_morse_flags_falls_risk_meds():
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=False, secondary_diagnosis_present=False,
        ambulatory_aid="none", has_iv_or_heparin_lock=False,
        gait="normal", mental_status="oriented",
        current_medications=[
            "lorazepam 1 mg PO BID",
            "oxycodone 5 mg PO q4h",
            "diphenhydramine 25 mg PO qhs",
            "lisinopril 10 mg PO daily",
        ],
    ))
    assert len(rep.contributing_medications_flagged) == 3
    flags_text = " ".join(rep.contributing_medications_flagged).lower()
    assert "benzodiazepine" in flags_text
    assert "opioid" in flags_text


def test_morse_three_high_risk_meds_bumps_to_high():
    """Moderate Morse + 3 falls-risk meds -> bump to high tier."""
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=True,
        secondary_diagnosis_present=False,
        ambulatory_aid="none",  # baseline 25 -> moderate
        has_iv_or_heparin_lock=False,
        gait="normal", mental_status="oriented",
        current_medications=[
            "lorazepam 1 mg",
            "oxycodone 5 mg",
            "diphenhydramine 25 mg",
        ],
    ))
    assert rep.score_total == 25
    # Without meds, would be moderate; with 3 meds -> high
    assert rep.risk_tier == "high"


def test_morse_moderate_at_25():
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=True, secondary_diagnosis_present=False,
        ambulatory_aid="none", has_iv_or_heparin_lock=False,
        gait="normal", mental_status="oriented",
    ))
    assert rep.score_total == 25
    assert rep.risk_tier == "moderate"


def test_morse_abstains_when_axis_missing():
    """Missing any of the 6 Morse axes -> abstain."""
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=True, ambulatory_aid="none",
        # secondary_diagnosis_present, has_iv_or_heparin_lock,
        # gait, mental_status all missing -> abstain
    ))
    assert rep.abstain_recommended is True
    assert rep.abstain_reason is not None
    assert "missing_morse_axes" in rep.abstain_reason


# ─────────────────────── CAM ───────────────────────

def test_cam_negative_when_f1_or_f2_missing():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=False,
        feature3_disorganized_thinking=True,
    ))
    assert rep.cam_positive is False


def test_cam_positive_classic():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=True,
    ))
    assert rep.cam_positive is True


def test_cam_positive_with_only_f4():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=False,
        feature4_altered_consciousness=True,
    ))
    assert rep.cam_positive is True


def test_cam_negative_no_f3_no_f4():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=False,
        feature4_altered_consciousness=False,
    ))
    assert rep.cam_positive is False


def test_cam_subtype_default():
    """When CAM positive but no subtype declared -> mixed."""
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=True,
        motor_subtype="unknown",
    ))
    assert rep.delirium_subtype == "mixed"


def test_cam_flags_deliriogenic_meds():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=True,
        current_medications=[
            "lorazepam 2 mg q4h", "oxycodone 5 mg",
            "diphenhydramine 50 mg qhs", "prednisone 20 mg",
        ],
    ))
    contrib_text = " ".join(rep.contributing_factors).lower()
    assert "benzodiazepine" in contrib_text
    assert "opioid" in contrib_text
    assert "corticosteroid" in contrib_text


def test_cam_negative_next_steps_appropriate():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=False,
        feature2_inattention=False,
    ))
    assert rep.cam_positive is False
    assert any("repeat" in s.lower() or "monitor" in s.lower()
                for s in rep.next_steps)


def test_cam_positive_next_steps_include_workup():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=True,
    ))
    next_text = " ".join(rep.next_steps).lower()
    assert "workup" in next_text or "ua" in next_text


def test_cam_hyperactive_includes_antipsychotic_caution():
    rep = _run(compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=True,
        motor_subtype="hyperactive",
    ))
    next_text = " ".join(rep.next_steps).lower()
    assert "antipsychotic" in next_text or "haloperidol" in next_text
