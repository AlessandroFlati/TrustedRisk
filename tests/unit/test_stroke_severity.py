"""Unit tests for compute_stroke_severity (NIHSS)."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.stroke_severity import (
    _classify_severity,
    compute_stroke_severity,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("total,expected", [
    (0, "minor"), (4, "minor"), (5, "moderate"),
    (15, "moderate"), (16, "moderate_severe"), (20, "moderate_severe"),
    (21, "severe"), (42, "severe"),
])
def test_severity_classification(total, expected):
    assert _classify_severity(total) == expected


def test_e2e_minor_stroke_no_lvo():
    rep = _run(compute_stroke_severity(item_scores={"facial_palsy": 1}))
    assert rep.score_total == 1
    assert rep.severity_tier == "minor"
    assert rep.lvo_suspected is False
    assert rep.recommended_response == "outpatient_workup"


def test_e2e_moderate_stroke_lvo_picks_evt():
    """NIHSS ≥6 + cortical signs (gaze + language + neglect) -> LVO suspected."""
    rep = _run(compute_stroke_severity(
        item_scores={
            "loc_responsiveness": 1,
            "best_gaze": 2,
            "motor_arm_right": 3,
            "best_language": 2,
            "extinction_inattention": 1,
        },
        last_known_well_minutes_ago=90,
    ))
    assert rep.score_total == 9
    assert rep.severity_tier == "moderate"
    assert rep.lvo_suspected is True
    assert rep.recommended_response == "endovascular_thrombectomy_evaluation"


def test_e2e_high_nihss_no_cortical_no_lvo_flag():
    """High NIHSS with motor-only deficits -> not LVO-flagged (still severe)."""
    rep = _run(compute_stroke_severity(item_scores={
        "motor_arm_left": 4, "motor_arm_right": 4,
        "motor_leg_left": 4, "motor_leg_right": 4, "facial_palsy": 3,
    }))
    assert rep.severity_tier in ("moderate_severe", "severe")
    assert rep.lvo_suspected is False  # no cortical sign


def test_e2e_outside_window_no_reperfusion_recommendation():
    rep = _run(compute_stroke_severity(
        item_scores={"loc_responsiveness": 1, "best_gaze": 2, "motor_arm_right": 3},
        last_known_well_minutes_ago=2000,
    ))
    assert rep.recommended_response == "stroke_unit_admission"


def test_e2e_thrombolysis_window_recommendation():
    rep = _run(compute_stroke_severity(
        item_scores={"facial_palsy": 2, "motor_arm_right": 2},
        last_known_well_minutes_ago=120,  # within 4.5h
    ))
    assert rep.recommended_response == "thrombolysis_evaluation"


def test_e2e_score_clamped_to_max_per_item():
    """NIHSS items have max points 2/3/4 -- overshoot is clipped."""
    rep = _run(compute_stroke_severity(item_scores={"facial_palsy": 99}))
    facial = next(it for it in rep.items if it.item == "facial_palsy")
    assert facial.points == 3  # max for facial_palsy


def test_e2e_unknown_item_ignored():
    rep = _run(compute_stroke_severity(
        item_scores={"facial_palsy": 1, "made_up_field": 5},
    ))
    assert rep.score_total == 1
    item_names = {it.item for it in rep.items}
    assert "made_up_field" not in item_names


def test_e2e_references_present():
    rep = _run(compute_stroke_severity(item_scores={"facial_palsy": 1}))
    assert any("AHA" in r or "Stroke" in r or "Brott" in r for r in rep.references)
