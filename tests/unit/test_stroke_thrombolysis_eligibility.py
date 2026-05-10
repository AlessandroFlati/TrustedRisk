"""Unit tests for compute_stroke_thrombolysis_eligibility."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.stroke_thrombolysis_eligibility import (
    compute_stroke_thrombolysis_eligibility,
)


def _run(coro):
    return asyncio.run(coro)


def test_within_3h_clean_picks_tpa_plus_evt():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=90, nihss_total=9,
        clinical_factors={
            "anterior_circulation_lvo_on_imaging": True, "aspects_score": 8,
            "systolic_bp": 165, "diastolic_bp": 95,
            "inr": 1.0, "platelets_per_ul": 220_000, "glucose_mg_dl": 110,
        },
    ))
    assert rep.iv_tpa_eligible is True
    assert rep.evt_eligible is True
    assert rep.decision == "iv_tpa_plus_evt"


def test_high_bp_blocks_tpa():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=120, nihss_total=10,
        clinical_factors={"systolic_bp": 200, "diastolic_bp": 95,
                           "inr": 1.0, "platelets_per_ul": 220_000},
    ))
    assert rep.iv_tpa_eligible is False
    assert any("SBP" in e for e in rep.iv_tpa_exclusion_present)


def test_inr_high_blocks_tpa():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=90, nihss_total=8,
        clinical_factors={"systolic_bp": 150, "diastolic_bp": 85,
                           "inr": 2.5, "platelets_per_ul": 200_000},
    ))
    assert rep.iv_tpa_eligible is False
    assert any("INR" in e for e in rep.iv_tpa_exclusion_present)


def test_low_platelets_blocks_tpa():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=90, nihss_total=8,
        clinical_factors={"systolic_bp": 150, "diastolic_bp": 85,
                           "inr": 1.0, "platelets_per_ul": 80_000},
    ))
    assert rep.iv_tpa_eligible is False
    assert any("platelets" in e.lower() for e in rep.iv_tpa_exclusion_present)


def test_recent_ich_blocks_tpa():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=60, nihss_total=12,
        clinical_factors={"systolic_bp": 150, "diastolic_bp": 85,
                           "history_intracranial_hemorrhage": True,
                           "inr": 1.0, "platelets_per_ul": 200_000},
    ))
    assert rep.iv_tpa_eligible is False


def test_extended_window_age_over_80_blocks():
    """3-4.5h window has age >80 as relative exclusion."""
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=240, nihss_total=10,
        clinical_factors={"systolic_bp": 160, "diastolic_bp": 85,
                           "inr": 1.0, "platelets_per_ul": 200_000,
                           "age": 84},
    ))
    assert rep.iv_tpa_window == "3_4_5h"
    assert any("age" in e.lower() and ">80" in e for e in rep.iv_tpa_exclusion_present)


def test_outside_window_no_tpa():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=400, nihss_total=10,
    ))
    assert rep.iv_tpa_window == "outside_window"
    assert rep.iv_tpa_eligible is False


def test_late_window_evt_with_dawn_mismatch():
    """6-24h with imaging mismatch -> EVT eligible per DAWN/DEFUSE 3."""
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=12 * 60, nihss_total=14,
        clinical_factors={
            "anterior_circulation_lvo_on_imaging": True,
            "dawn_or_defuse_mismatch_present": True,
        },
    ))
    assert rep.evt_window == "6_24h_dawn_defuse"
    assert rep.evt_eligible is True
    assert rep.decision == "evt_only"


def test_late_window_evt_blocked_without_mismatch():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=10 * 60, nihss_total=10,
        clinical_factors={"anterior_circulation_lvo_on_imaging": True,
                           "dawn_or_defuse_mismatch_present": False},
    ))
    assert rep.evt_eligible is False
    assert any("mismatch" in c.lower() for c in rep.evt_criteria_unmet)


def test_low_nihss_outside_window_supportive():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=2000, nihss_total=2,
    ))
    assert rep.decision == "no_reperfusion_supportive_care"


def test_doac_within_48h_blocks_tpa():
    rep = _run(compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=90, nihss_total=8,
        clinical_factors={"systolic_bp": 150, "diastolic_bp": 85,
                           "inr": 1.0, "platelets_per_ul": 200_000,
                           "doac_use_within_48h": True},
    ))
    assert rep.iv_tpa_eligible is False
    assert any("DOAC" in e for e in rep.iv_tpa_exclusion_present)
