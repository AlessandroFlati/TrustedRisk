"""Unit tests for AMB-5.1 multi-critic ensemble vote + AMB-5.2 plan revision."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from a2a_agent.critique import (
    aggregate_critic_verdicts,
    apply_critique,
    collapse_ensemble_to_single_critique,
)
from a2a_agent.plan_revision import run_with_plan_revision
from shared.schemas import (
    AbstainTrigger,
    Action,
    AuditBlock,
    ClaimGrounding,
    CriticEnsembleVerdict,
    CritiqueDecision,
    DecisionCard,
    DecisionReasoning,
    DecisionValidation,
    Factor,
    PHIReport,
    ProbInterval,
    Recommendation,
    RiskEstimate,
    SubClaim,
    UtilityAnalysis,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Aggregation rules ───────────────────────

def test_aggregate_three_approved_is_approved():
    votes = [
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="clinical_safety"),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="evidence"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    assert ens.aggregated_verdict == "approved"
    assert ens.applied_verdict == "approved"
    assert ens.n_approved == 3


def test_aggregate_one_force_abstain_dominates():
    votes = [
        CritiqueDecision(verdict="approved", rationale="ok", critic_role="evidence"),
        CritiqueDecision(verdict="approved", rationale="ok", critic_role="fairness"),
        CritiqueDecision(verdict="force_abstain", rationale="anaphylaxis",
                          critic_role="clinical_safety",
                          abstain_trigger_to_add=AbstainTrigger(
                              type="evidence_insufficient",
                              detail="mock anaphylaxis path")),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    assert ens.aggregated_verdict == "force_abstain"
    assert ens.applied_verdict == "force_abstain"
    assert ens.n_force_abstain == 1


def test_aggregate_two_downgrade_is_downgrade():
    votes = [
        CritiqueDecision(verdict="downgrade_confidence", rationale="moderate drift",
                          critic_role="fairness", confidence_after="medium"),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="evidence"),
        CritiqueDecision(verdict="downgrade_confidence", rationale="borderline CI",
                          critic_role="clinical_safety", confidence_after="low"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    assert ens.aggregated_verdict == "downgrade_confidence"
    assert ens.applied_verdict == "downgrade_confidence"
    assert ens.n_downgrade == 2


def test_aggregate_one_replay_with_budget():
    votes = [
        CritiqueDecision(verdict="request_replay", rationale="missing fairness audit",
                          critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ok", critic_role="evidence"),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="clinical_safety"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    assert ens.aggregated_verdict == "request_replay"
    assert ens.applied_verdict == "request_replay"


def test_replay_collapses_to_force_abstain_when_budget_exhausted():
    votes = [
        CritiqueDecision(verdict="request_replay", rationale="missing fairness audit",
                          critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ok", critic_role="evidence"),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="clinical_safety"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=0)
    assert ens.aggregated_verdict == "request_replay"
    assert ens.applied_verdict == "force_abstain"


def test_aggregate_priority_force_over_replay():
    """force_abstain trumps request_replay even with retries available."""
    votes = [
        CritiqueDecision(verdict="force_abstain", rationale="hyperK + ACE-I",
                          critic_role="clinical_safety"),
        CritiqueDecision(verdict="request_replay", rationale="missing audit",
                          critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="evidence"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    assert ens.aggregated_verdict == "force_abstain"


def test_aggregate_empty_defaults_to_replay_or_abstain():
    ens = aggregate_critic_verdicts([], retries_remaining=2)
    assert ens.aggregated_verdict == "request_replay"
    assert ens.applied_verdict == "request_replay"

    ens0 = aggregate_critic_verdicts([], retries_remaining=0)
    assert ens0.applied_verdict == "force_abstain"


def test_aggregate_one_downgrade_one_replay_combines_to_downgrade():
    votes = [
        CritiqueDecision(verdict="downgrade_confidence", rationale="borderline",
                          critic_role="clinical_safety", confidence_after="medium"),
        CritiqueDecision(verdict="request_replay", rationale="missing tool",
                          critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ok", critic_role="evidence"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    # Replay wins -- request_replay has higher severity than downgrade
    assert ens.aggregated_verdict == "request_replay"


def test_collapse_ensemble_picks_lowest_confidence_after():
    votes = [
        CritiqueDecision(verdict="downgrade_confidence", rationale="a",
                          critic_role="fairness", confidence_after="medium"),
        CritiqueDecision(verdict="downgrade_confidence", rationale="b",
                          critic_role="clinical_safety", confidence_after="low"),
        CritiqueDecision(verdict="approved", rationale="c", critic_role="evidence"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    collapsed = collapse_ensemble_to_single_critique(ens)
    assert collapsed.confidence_after == "low"


def test_collapse_ensemble_picks_first_force_trigger():
    votes = [
        CritiqueDecision(verdict="force_abstain", rationale="hyperK",
                          critic_role="clinical_safety",
                          abstain_trigger_to_add=AbstainTrigger(
                              type="evidence_insufficient",
                              detail="hyperK refractory")),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ok",
                          critic_role="evidence"),
    ]
    ens = aggregate_critic_verdicts(votes, retries_remaining=2)
    collapsed = collapse_ensemble_to_single_critique(ens)
    assert collapsed.abstain_trigger_to_add is not None
    assert "hyperK" in collapsed.abstain_trigger_to_add.detail


# ─────────────────────── Plan revision integration ───────────────────────

def _make_card(action: str = "discharge_home", confidence: str = "high"
                 ) -> DecisionCard:
    """Minimal valid DecisionCard for testing."""
    risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="test-v1",
        outcome_id="readmission_30d",
        horizon_days=30, lace_raw_score=5,
        probability_mean=0.10,
        probability_ci95=(0.07, 0.13),
        probability_ci_width=0.06,
        contributing_factors=[Factor(name="LACE_length_of_stay",
                                        raw_value=2.0, lace_points=2,
                                        weight=0.4)],
        fhir_observations_used=[],
        computed_at=datetime.now(timezone.utc),
    )
    util = UtilityAnalysis(
        action_scores_qaly_weeks={Action.DISCHARGE_HOME: 0.8},
        action_scores_ci95={Action.DISCHARGE_HOME: (0.6, 1.0)},
        action_costs_usd={Action.DISCHARGE_HOME: 0.0},
        dominant_action=Action.DISCHARGE_HOME,
        dominance_confidence=0.95,
        reasoning_trace="discharge dominates",
    )
    grounding = ClaimGrounding(
        claim_text="patient stable",
        sub_claims=[SubClaim(text="vitals stable", atomic_kind="vital",
                              verdict="supported", confidence=0.9)],
        overall_verdict="supported",
        context_fingerprint="abc",
        grounded_at=datetime.now(timezone.utc),
    )
    phi = PHIReport(entities_found=[], entity_count_by_type={},
                     redaction_map={}, risk_level="none")
    audit = AuditBlock(
        request_id="req-test", context_fingerprint="abc",
        tool_trace=[], server_version="test", agent_version="test",
        model_coefficients_version="test",
        timestamp=datetime.now(timezone.utc),
    )
    return DecisionCard(
        recommendation=Recommendation(action=Action(action),
                                          confidence=confidence),  # type: ignore[arg-type]
        reasoning=DecisionReasoning(risk_estimate=risk, utility_analysis=util),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=None,
        audit=audit,
    )


def test_plan_revision_approved_terminates_in_one_attempt():
    async def compose(input_, caveats):
        return _make_card()

    async def critic(card):
        return aggregate_critic_verdicts([
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="clinical_safety"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="fairness"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="evidence"),
        ], retries_remaining=2)

    trace = _run(run_with_plan_revision(compose, critic, initial_input={}))
    assert len(trace.attempts) == 1
    assert trace.revisions_used == 0
    assert trace.terminated_reason == "approved"


def test_plan_revision_replay_then_approve():
    """First attempt: replay. Second attempt: approve."""
    state = {"call_count": 0}

    async def compose(input_, caveats):
        state["call_count"] += 1
        return _make_card()

    async def critic(card):
        # First call -> replay; subsequent calls -> approve
        if state["call_count"] == 1:
            verdict = "request_replay"
        else:
            verdict = "approved"
        votes = [
            CritiqueDecision(verdict=verdict, rationale="trying",
                              critic_role="clinical_safety"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="fairness"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="evidence"),
        ]
        return aggregate_critic_verdicts(votes, retries_remaining=2)

    trace = _run(run_with_plan_revision(compose, critic, initial_input={}))
    assert len(trace.attempts) == 2
    assert trace.revisions_used == 1
    assert trace.terminated_reason == "approved"


def test_plan_revision_budget_exhaustion_collapses_to_abstain():
    """All 3 attempts emit replay -> final attempt collapses to force_abstain."""
    async def compose(input_, caveats):
        return _make_card()

    async def critic(card):
        return aggregate_critic_verdicts([
            CritiqueDecision(verdict="request_replay", rationale="loop",
                              critic_role="fairness"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="evidence"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="clinical_safety"),
        ], retries_remaining=2)

    trace = _run(run_with_plan_revision(compose, critic, initial_input={},
                                            max_replays=2))
    assert len(trace.attempts) == 3
    assert trace.revisions_used == 2
    # Final attempt has applied_verdict force_abstain (budget exhausted)
    assert trace.attempts[-1].ensemble.applied_verdict == "force_abstain"
    assert trace.terminated_reason == "force_abstain"


def test_plan_revision_caveats_accumulate():
    """Each replay should pass the prior critic rationale as a caveat."""
    captured_caveats: list[list[str]] = []

    async def compose(input_, caveats):
        captured_caveats.append(list(caveats))
        return _make_card()

    state = {"call_count": 0}

    async def critic(card):
        state["call_count"] += 1
        if state["call_count"] <= 2:
            verdict = "request_replay"
        else:
            verdict = "approved"
        votes = [
            CritiqueDecision(verdict=verdict, rationale=f"missing_X_attempt_{state['call_count']}",
                              critic_role="fairness"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="evidence"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="clinical_safety"),
        ]
        return aggregate_critic_verdicts(votes, retries_remaining=2)

    trace = _run(run_with_plan_revision(compose, critic, initial_input={},
                                            max_replays=2))
    # 3 calls -> 3 caveat lists (0, 1, 2 caveats respectively)
    assert len(captured_caveats) == 3
    assert captured_caveats[0] == []
    assert len(captured_caveats[1]) == 1
    assert len(captured_caveats[2]) == 2


def test_plan_revision_force_abstain_terminates_immediately():
    """force_abstain on attempt 0 -> no replay attempted."""
    async def compose(input_, caveats):
        return _make_card()

    async def critic(card):
        return aggregate_critic_verdicts([
            CritiqueDecision(verdict="force_abstain", rationale="critical",
                              critic_role="clinical_safety"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="fairness"),
            CritiqueDecision(verdict="approved", rationale="ok",
                              critic_role="evidence"),
        ], retries_remaining=2)

    trace = _run(run_with_plan_revision(compose, critic, initial_input={}))
    assert len(trace.attempts) == 1
    assert trace.terminated_reason == "force_abstain"
    assert trace.final_card.recommendation is None
