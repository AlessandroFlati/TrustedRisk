"""Unit tests for compute_oncology_treatment_response (RECIST 1.1)."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.oncology_treatment_response import (
    _classify_recist,
    compute_oncology_treatment_response,
)


def _run(coro):
    return asyncio.run(coro)


def _lesion(id_: str, baseline: float, current: float) -> dict:
    return {
        "lesion_id": id_, "location": "right_lung",
        "baseline_longest_diameter_mm": baseline,
        "current_longest_diameter_mm": current,
    }


# ─────────────────────── Classification ───────────────────────

def test_complete_response_when_all_disappear():
    cls = _classify_recist(baseline_sum=80, current_sum=0,
                            new_lesions=False, non_target_pd=False,
                            all_lesions_disappeared=True)
    assert cls == "complete_response"


def test_progressive_disease_with_new_lesion():
    cls = _classify_recist(baseline_sum=80, current_sum=70,
                            new_lesions=True, non_target_pd=False,
                            all_lesions_disappeared=False)
    assert cls == "progressive_disease"


def test_partial_response_at_30_percent():
    cls = _classify_recist(baseline_sum=100, current_sum=70,
                            new_lesions=False, non_target_pd=False,
                            all_lesions_disappeared=False)
    assert cls == "partial_response"


def test_progressive_at_20_percent_increase():
    cls = _classify_recist(baseline_sum=100, current_sum=125,
                            new_lesions=False, non_target_pd=False,
                            all_lesions_disappeared=False)
    assert cls == "progressive_disease"


def test_stable_disease_in_between():
    cls = _classify_recist(baseline_sum=100, current_sum=85,
                            new_lesions=False, non_target_pd=False,
                            all_lesions_disappeared=False)
    assert cls == "stable_disease"


def test_5mm_minimum_for_progression():
    """20% increase but absolute change <5mm should NOT count as progression."""
    cls = _classify_recist(baseline_sum=20, current_sum=24,  # +20% but only 4mm
                            new_lesions=False, non_target_pd=False,
                            all_lesions_disappeared=False)
    assert cls == "stable_disease"


def test_non_evaluable_when_baseline_zero():
    cls = _classify_recist(baseline_sum=0, current_sum=0,
                            new_lesions=False, non_target_pd=False,
                            all_lesions_disappeared=False)
    assert cls == "not_evaluable"


# ─────────────────────── End-to-end ───────────────────────

def test_e2e_partial_response_continues_therapy():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 25), _lesion("L2", 22, 16)],
    ))
    assert rep.overall_response == "partial_response"
    assert rep.decision_implication == "continue_current_therapy"


def test_e2e_progression_switches_therapy():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 50), _lesion("L2", 20, 30)],
    ))
    assert rep.overall_response == "progressive_disease"
    assert rep.decision_implication == "switch_therapy"


def test_e2e_consecutive_progression_palliative():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 60)],
        prior_response_categories=["progressive_disease"],
    ))
    assert rep.overall_response == "progressive_disease"
    assert rep.decision_implication == "transition_palliative"


def test_e2e_repeated_stable_disease_considers_escalation():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 38)],
        prior_response_categories=["stable_disease", "stable_disease"],
    ))
    assert rep.overall_response == "stable_disease"
    assert rep.decision_implication == "consider_dose_escalation"


def test_e2e_new_lesion_pd():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 30)],  # would be PR
        new_lesions_present=True,
    ))
    # New lesion overrides -- PD
    assert rep.overall_response == "progressive_disease"


def test_e2e_non_target_pd():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 32)],
        non_target_progression=True,
    ))
    assert rep.overall_response == "progressive_disease"


def test_e2e_no_lesions_not_evaluable():
    rep = _run(compute_oncology_treatment_response(target_lesions=[]))
    assert rep.overall_response == "not_evaluable"
    assert rep.abstain_recommended is True


def test_e2e_complete_response():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 0), _lesion("L2", 20, 0)],
    ))
    assert rep.overall_response == "complete_response"
    assert rep.decision_implication == "continue_current_therapy"


def test_e2e_pct_change_signed():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 100, 60)],
    ))
    assert rep.sum_change_pct == pytest.approx(-40.0)


def test_e2e_references_present():
    rep = _run(compute_oncology_treatment_response(
        target_lesions=[_lesion("L1", 40, 30)],
    ))
    assert any("RECIST" in r or "Eisenhauer" in r for r in rep.references)
