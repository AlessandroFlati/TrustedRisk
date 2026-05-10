"""Phase 14.16 Q1 -- Multi-agent debate / round-table tests."""

from __future__ import annotations

import pytest

from a2a_agent.multi_agent_debate import (
    DebateInput, run_debate,
)


def _payload(**overrides) -> DebateInput:
    base = dict(
        recommended_action="discharge_home",
        risk_point_estimate=0.10,
        risk_ci_width=0.04,
        fairness_subgroup=None,
        fairness_audit_present=False,
    )
    base.update(overrides)
    return DebateInput(**base)


# ─────────────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────────────

def test_debate_returns_three_votes():
    out = run_debate(_payload())
    assert len(out.votes) == 3
    ids = [v.agent_id for v in out.votes]
    assert set(ids) == {
        "clinical_conservative", "evidence_aggressive", "fairness_guard",
    }


def test_each_vote_carries_rationale():
    out = run_debate(_payload())
    for v in out.votes:
        assert v.rationale.strip() != ""


# ─────────────────────────────────────────────────────────────────────
# Approve path
# ─────────────────────────────────────────────────────────────────────

def test_unanimous_approve_when_low_risk_tight_ci_no_flagged_subgroup():
    out = run_debate(_payload(
        risk_point_estimate=0.08, risk_ci_width=0.04,
    ))
    assert out.verdict == "approved"
    assert all(v.vote == "approve" for v in out.votes)


# ─────────────────────────────────────────────────────────────────────
# Conservative agent veto
# ─────────────────────────────────────────────────────────────────────

def test_conservative_abstain_high_risk_discharge_forces_abstain():
    out = run_debate(_payload(
        recommended_action="discharge_home",
        risk_point_estimate=0.45,
        risk_ci_width=0.05,
    ))
    assert out.verdict == "force_abstain"
    cc_vote = next(
        v for v in out.votes if v.agent_id == "clinical_conservative"
    )
    assert cc_vote.vote == "abstain"


def test_conservative_revise_when_wide_ci_on_discharge():
    out = run_debate(_payload(
        recommended_action="discharge_home",
        risk_point_estimate=0.10,
        risk_ci_width=0.18,   # > 0.10 moderate bar
    ))
    cc_vote = next(
        v for v in out.votes if v.agent_id == "clinical_conservative"
    )
    assert cc_vote.vote == "revise"


# ─────────────────────────────────────────────────────────────────────
# Evidence-aggressive agent
# ─────────────────────────────────────────────────────────────────────

def test_evidence_aggressive_approves_tight_ci():
    out = run_debate(_payload(
        risk_point_estimate=0.12, risk_ci_width=0.03,
    ))
    ea_vote = next(
        v for v in out.votes if v.agent_id == "evidence_aggressive"
    )
    assert ea_vote.vote == "approve"


def test_evidence_aggressive_revises_very_wide_ci():
    out = run_debate(_payload(
        risk_point_estimate=0.12, risk_ci_width=0.30,
    ))
    ea_vote = next(
        v for v in out.votes if v.agent_id == "evidence_aggressive"
    )
    assert ea_vote.vote == "revise"


# ─────────────────────────────────────────────────────────────────────
# Fairness-guard agent veto
# ─────────────────────────────────────────────────────────────────────

def test_fairness_guard_abstains_on_flagged_subgroup_no_audit():
    out = run_debate(_payload(
        fairness_subgroup="black",
        fairness_audit_present=False,
    ))
    fg_vote = next(
        v for v in out.votes if v.agent_id == "fairness_guard"
    )
    assert fg_vote.vote == "abstain"
    assert out.verdict == "force_abstain"


def test_fairness_guard_revises_on_flagged_subgroup_with_audit():
    out = run_debate(_payload(
        fairness_subgroup="indigenous",
        fairness_audit_present=True,
    ))
    fg_vote = next(
        v for v in out.votes if v.agent_id == "fairness_guard"
    )
    assert fg_vote.vote == "revise"


def test_fairness_guard_approves_unflagged_subgroup():
    out = run_debate(_payload(
        fairness_subgroup="caucasian",
        fairness_audit_present=False,
    ))
    fg_vote = next(
        v for v in out.votes if v.agent_id == "fairness_guard"
    )
    assert fg_vote.vote == "approve"


# ─────────────────────────────────────────────────────────────────────
# Aggregation logic
# ─────────────────────────────────────────────────────────────────────

def test_majority_revise_yields_revise():
    out = run_debate(_payload(
        recommended_action="discharge_home",
        risk_point_estimate=0.12,
        risk_ci_width=0.25,   # both CC and EA say revise
    ))
    assert out.verdict == "revise"


def test_safety_veto_overrides_two_approve_votes():
    """Even if 2 of 3 say approve, fairness_guard's abstain overrides."""
    out = run_debate(_payload(
        recommended_action="discharge_home",
        risk_point_estimate=0.05,
        risk_ci_width=0.03,
        fairness_subgroup="lgbtq+",
        fairness_audit_present=False,
    ))
    assert out.verdict == "force_abstain"


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────

def test_debate_is_deterministic():
    payload = _payload(
        risk_point_estimate=0.18, risk_ci_width=0.07,
        fairness_subgroup="medicaid", fairness_audit_present=True,
    )
    a = run_debate(payload)
    b = run_debate(payload)
    assert a.model_dump() == b.model_dump()


# ─────────────────────────────────────────────────────────────────────
# Schema invariants
# ─────────────────────────────────────────────────────────────────────

def test_payload_rejects_out_of_unit_risk():
    with pytest.raises(Exception):
        DebateInput(
            recommended_action="discharge_home",
            risk_point_estimate=1.5,
            risk_ci_width=0.05,
        )


def test_payload_rejects_negative_ci_width():
    with pytest.raises(Exception):
        DebateInput(
            recommended_action="discharge_home",
            risk_point_estimate=0.1,
            risk_ci_width=-0.05,
        )
