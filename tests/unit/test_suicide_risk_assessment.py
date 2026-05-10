"""Unit tests for compute_suicide_risk_assessment."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.suicide_risk_assessment import (
    _classify_risk,
    compute_suicide_risk_assessment,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Risk classification ───────────────────────

def test_recent_attempt_is_imminent():
    risk = _classify_risk(ideation_recent=2, behavior_past_30d=True,
                           behavior_lifetime_attempts=1, warning_factors=2,
                           protective_factors=0)
    assert risk == "imminent"


def test_active_plan_intent_is_imminent():
    risk = _classify_risk(ideation_recent=5, behavior_past_30d=False,
                           behavior_lifetime_attempts=0, warning_factors=0,
                           protective_factors=0)
    assert risk == "imminent"


def test_active_plan_no_intent_is_high():
    risk = _classify_risk(ideation_recent=4, behavior_past_30d=False,
                           behavior_lifetime_attempts=0, warning_factors=0,
                           protective_factors=0)
    assert risk == "high"


def test_lifetime_attempt_plus_recent_ideation_is_high():
    risk = _classify_risk(ideation_recent=3, behavior_past_30d=False,
                           behavior_lifetime_attempts=1, warning_factors=0,
                           protective_factors=0)
    assert risk == "high"


def test_passive_ideation_alone_is_low_or_moderate():
    risk = _classify_risk(ideation_recent=1, behavior_past_30d=False,
                           behavior_lifetime_attempts=0, warning_factors=0,
                           protective_factors=0)
    assert risk == "low"


def test_moderate_ideation_with_protective_can_downgrade():
    risk = _classify_risk(ideation_recent=2, behavior_past_30d=False,
                           behavior_lifetime_attempts=0, warning_factors=0,
                           protective_factors=0)
    assert risk == "moderate"


# ─────────────────────── End-to-end ───────────────────────

def test_e2e_imminent_with_recent_attempt():
    rep = _run(compute_suicide_risk_assessment(
        ideation_past_30d_level=5,
        behavior_past_30d_any=True,
        behavior_lifetime_attempts=1,
        warning_factors_count=3,
    ))
    assert rep.risk_level == "imminent"
    assert rep.safety_plan_indicated is True


def test_e2e_bias_guard_triggers_abstain():
    """Demographic flag for under-predicted minoritized groups -> abstain."""
    rep = _run(compute_suicide_risk_assessment(
        ideation_past_30d_level=2,
        patient_demographics={"race": "black", "ses": "low_ses"},
    ))
    assert rep.abstain_recommended is True
    assert "demographic_bias_guard" in (rep.abstain_reason or "")


def test_e2e_sparse_input_abstains():
    """All zeros -> abstain (could reflect non-disclosure)."""
    rep = _run(compute_suicide_risk_assessment())
    assert rep.abstain_recommended is True
    assert "sparse_input" in (rep.abstain_reason or "")


def test_e2e_low_risk_no_abstain_with_complete_input():
    rep = _run(compute_suicide_risk_assessment(
        ideation_lifetime_level=2,  # at least one positive field
        ideation_past_30d_level=0,
        behavior_lifetime_attempts=0,
        protective_factors_count=4,
    ))
    assert rep.risk_level == "low"
    assert rep.abstain_recommended is False


def test_e2e_safety_plan_indicated_for_moderate_and_above():
    moderate = _run(compute_suicide_risk_assessment(
        ideation_past_30d_level=3, behavior_lifetime_attempts=0,
        protective_factors_count=0,
    ))
    assert moderate.risk_level in ("moderate", "high")
    assert moderate.safety_plan_indicated is True


def test_e2e_lgbtq_demographic_triggers_bias_guard():
    rep = _run(compute_suicide_risk_assessment(
        ideation_past_30d_level=3,
        patient_demographics={"sexual_orientation": "lgbtq"},
    ))
    assert rep.abstain_recommended is True


def test_e2e_references_present():
    rep = _run(compute_suicide_risk_assessment(ideation_past_30d_level=2))
    assert any("Posner" in r or "Columbia" in r for r in rep.references)
