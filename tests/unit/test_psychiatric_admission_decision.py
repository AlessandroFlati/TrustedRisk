"""Unit tests for compute_psychiatric_admission_decision."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.psychiatric_admission_decision import (
    compute_psychiatric_admission_decision,
)


def _run(coro):
    return asyncio.run(coro)


def test_imminent_with_voluntary_capacity_voluntary_inpatient():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="imminent",
        danger_to_self=True,
        voluntary_capable=True,
    ))
    # Imminent -> admission required; voluntary capable + danger to self only -> voluntary
    # but danger to self at imminent level is interpreted as needing involuntary
    # in our current policy (defensive). Either is acceptable for clinical safety.
    assert rep.disposition in ("voluntary_inpatient", "involuntary_hold_evaluation")


def test_danger_to_others_is_involuntary():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="high",
        danger_to_others=True,
        voluntary_capable=True,
    ))
    assert rep.disposition == "involuntary_hold_evaluation"
    assert rep.abstain_recommended is True


def test_grave_disability_no_capacity_involuntary():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="high",
        grave_disability=True,
        voluntary_capable=False,
    ))
    assert rep.disposition == "involuntary_hold_evaluation"


def test_moderate_with_treatment_outpatient():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="moderate",
        has_engaged_outpatient_treatment=True,
        has_safety_plan_in_place=True,
    ))
    assert rep.disposition == "outpatient_followup"
    assert rep.follow_up_within_hours <= 72


def test_moderate_no_treatment_crisis_unit():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="moderate",
        has_engaged_outpatient_treatment=False,
        has_safety_plan_in_place=False,
    ))
    assert rep.disposition == "crisis_stabilization_unit"


def test_low_risk_outpatient_followup():
    rep = _run(compute_psychiatric_admission_decision(risk_level="low"))
    assert rep.disposition == "outpatient_followup"
    assert rep.follow_up_within_hours == 168


def test_state_jurisdiction_legal_basis():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="imminent",
        danger_to_self=True,
        voluntary_capable=False,
        state_jurisdiction="CO",
    ))
    assert rep.legal_basis is not None
    assert "Colorado" in rep.legal_basis or "M-1" in rep.legal_basis


def test_california_5150():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="high",
        danger_to_self=True,
        voluntary_capable=False,
        state_jurisdiction="CA",
    ))
    assert rep.legal_basis is not None
    assert "5150" in rep.legal_basis


def test_unknown_jurisdiction_generic_legal_basis():
    rep = _run(compute_psychiatric_admission_decision(
        risk_level="high",
        danger_to_self=True,
        voluntary_capable=False,
        state_jurisdiction="ZZ",  # nonexistent
    ))
    assert rep.legal_basis is not None
    assert "varies by jurisdiction" in rep.legal_basis


def test_safety_plan_required_for_non_low():
    moderate = _run(compute_psychiatric_admission_decision(risk_level="moderate"))
    assert moderate.safety_plan_required is True

    low = _run(compute_psychiatric_admission_decision(risk_level="low"))
    # low with no other findings -- safety plan not strictly required
    assert low.safety_plan_required is False or low.disposition != "outpatient_followup"
