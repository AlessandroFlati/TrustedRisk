"""Unit tests for apply_critique() -- deterministic critique application."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from a2a_agent.critique import CRITIC_INSTRUCTION, apply_critique
from shared.schemas import (
    AbstainTrigger,
    Action,
    AuditBlock,
    ClaimGrounding,
    CritiqueDecision,
    DecisionCard,
    DecisionReasoning,
    DecisionValidation,
    Factor,
    PHIReport,
    Recommendation,
    RiskEstimate,
    ToolInvocationTrace,
    UtilityAnalysis,
)


def _now():
    return datetime.now(timezone.utc)


def _full_card(*, recommendation_action=Action.HOME_WITH_CARE,
               confidence="high", abstain=None) -> DecisionCard:
    risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="test",
        horizon_days=30,
        lace_raw_score=10,
        probability_mean=0.18,
        probability_ci95=(0.12, 0.24),
        probability_ci_width=0.12,
        contributing_factors=[Factor(name="L", raw_value=4, lace_points=4, weight=0.4)],
        computed_at=_now(),
    )
    util = UtilityAnalysis(
        action_scores_qaly_weeks={Action.HOME_WITH_CARE: 4.5},
        action_scores_ci95={Action.HOME_WITH_CARE: (4.0, 5.0)},
        action_costs_usd={Action.HOME_WITH_CARE: 1500.0},
        dominant_action=Action.HOME_WITH_CARE,
        dominance_confidence=0.85,
        reasoning_trace="ok",
    )
    grounding = ClaimGrounding(
        claim_text="Discharge plan ok.",
        sub_claims=[], overall_verdict="supported",
        context_fingerprint="fp", grounded_at=_now(),
    )
    phi = PHIReport(entities_found=[], entity_count_by_type={}, redaction_map={}, risk_level="none")
    audit = AuditBlock(
        request_id="req-1", context_fingerprint="fp",
        tool_trace=[ToolInvocationTrace(tool="compute_readmission_risk",
                                         invoked_at=_now(), duration_ms=120, status="ok")],
        server_version="trustedrisk-0.1", agent_version="agent-0.1",
        model_coefficients_version="v1", timestamp=_now(),
    )
    return DecisionCard(
        recommendation=Recommendation(action=recommendation_action,
                                       confidence=confidence),  # type: ignore[arg-type]
        reasoning=DecisionReasoning(risk_estimate=risk, utility_analysis=util),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=abstain,
        audit=audit,
    )


# ─────────────────────── approved ───────────────────────

def test_apply_approved_passes_through():
    card = _full_card()
    critique = CritiqueDecision(verdict="approved", rationale="card coherent")
    out = apply_critique(card, critique)
    assert out.recommendation == card.recommendation
    assert out.reasoning == card.reasoning
    assert out.self_critique is critique


def test_apply_approved_does_not_mutate_input():
    card = _full_card()
    original_conf = card.recommendation.confidence
    critique = CritiqueDecision(verdict="approved", rationale="ok")
    apply_critique(card, critique)
    assert card.recommendation.confidence == original_conf
    assert card.self_critique is None  # original was never edited


# ─────────────────────── downgrade_confidence ───────────────────────

def test_apply_downgrade_high_to_medium():
    card = _full_card(confidence="high")
    critique = CritiqueDecision(
        verdict="downgrade_confidence",
        rationale="fairness_audit medium drift on race subgroup",
    )
    out = apply_critique(card, critique)
    assert out.recommendation.confidence == "medium"
    # RiskEstimate.confidence flipped to degraded
    assert out.reasoning.risk_estimate.confidence == "degraded"


def test_apply_downgrade_medium_to_low():
    card = _full_card(confidence="medium")
    critique = CritiqueDecision(verdict="downgrade_confidence", rationale="x")
    out = apply_critique(card, critique)
    assert out.recommendation.confidence == "low"


def test_apply_downgrade_low_to_none():
    card = _full_card(confidence="low")
    critique = CritiqueDecision(verdict="downgrade_confidence", rationale="x")
    out = apply_critique(card, critique)
    assert out.recommendation.confidence == "none"


def test_apply_downgrade_none_floor():
    card = _full_card(confidence="none")
    critique = CritiqueDecision(verdict="downgrade_confidence", rationale="x")
    out = apply_critique(card, critique)
    assert out.recommendation.confidence == "none"


def test_apply_downgrade_explicit_target():
    card = _full_card(confidence="high")
    critique = CritiqueDecision(
        verdict="downgrade_confidence", rationale="x", confidence_after="low",
    )
    out = apply_critique(card, critique)
    assert out.recommendation.confidence == "low"


def test_apply_downgrade_no_recommendation_safe():
    card = _full_card()
    card_no_rec = card.model_copy(update={"recommendation": None})
    critique = CritiqueDecision(verdict="downgrade_confidence", rationale="x")
    out = apply_critique(card_no_rec, critique)
    # Should not crash; remains None
    assert out.recommendation is None


# ─────────────────────── force_abstain ───────────────────────

def test_apply_force_abstain_drops_recommendation():
    card = _full_card()
    trigger = AbstainTrigger(
        type="evidence_insufficient", detail="creatinine stale",
    )
    critique = CritiqueDecision(
        verdict="force_abstain",
        rationale="Critic detected unsupported critical sub-claim",
        abstain_trigger_to_add=trigger,
    )
    out = apply_critique(card, critique)
    assert out.recommendation is None
    assert out.reasoning is None
    assert out.abstain is not None
    assert any(a.type == "evidence_insufficient" for a in out.abstain)


def test_apply_force_abstain_synthesizes_trigger_when_missing():
    """If critic asks for force_abstain without supplying a trigger, synthesize one."""
    card = _full_card()
    critique = CritiqueDecision(
        verdict="force_abstain",
        rationale="Card incoherent -- recommendation contradicts grounding",
    )
    out = apply_critique(card, critique)
    assert out.recommendation is None
    assert out.abstain and len(out.abstain) >= 1


def test_apply_force_abstain_appends_to_existing_list():
    existing = AbstainTrigger(type="confidence_interval_too_wide", detail="ci wide")
    card = _full_card(abstain=[existing])
    new_trigger = AbstainTrigger(type="evidence_insufficient", detail="creatinine stale")
    critique = CritiqueDecision(
        verdict="force_abstain", rationale="add evidence trigger",
        abstain_trigger_to_add=new_trigger,
    )
    out = apply_critique(card, critique)
    assert len(out.abstain) == 2
    types = {a.type for a in out.abstain}
    assert "confidence_interval_too_wide" in types
    assert "evidence_insufficient" in types


# ─────────────────────── request_replay ───────────────────────

def test_apply_request_replay_drops_recommendation():
    card = _full_card()
    critique = CritiqueDecision(
        verdict="request_replay",
        rationale="missing fairness_audit when demographics were available",
    )
    out = apply_critique(card, critique)
    assert out.recommendation is None
    assert out.reasoning is not None  # reasoning preserved for re-pass
    assert out.self_critique is critique


# ─────────────────────── critique attached to audit ───────────────────────

def test_critique_always_attached():
    card = _full_card()
    for verdict in ["approved", "downgrade_confidence", "force_abstain", "request_replay"]:
        critique = CritiqueDecision(verdict=verdict, rationale="x")  # type: ignore[arg-type]
        out = apply_critique(card, critique)
        assert out.self_critique is critique


def test_critic_instruction_has_four_verdicts():
    """Sanity check: the LLM instruction enumerates exactly the 4 verdicts
    the apply_critique() function knows how to handle."""
    for v in ["approved", "downgrade_confidence", "force_abstain", "request_replay"]:
        assert v in CRITIC_INSTRUCTION
