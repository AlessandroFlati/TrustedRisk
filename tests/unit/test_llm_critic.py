"""GENAI-3 unit tests for the LLM-driven critic."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from a2a_agent import llm_critic as mod
from a2a_agent.llm_critic import llm_judge_critic
from shared.schemas import (
    AbstainTrigger,
    Action,
    AuditBlock,
    ClaimGrounding,
    DecisionCard,
    DecisionReasoning,
    DecisionValidation,
    Factor,
    PHIReport,
    Recommendation,
    RiskEstimate,
    UtilityAnalysis,
)


def _risk_estimate(prob=0.20, lace=10):
    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="0.7.0",
        outcome_id="readmission_30d",
        horizon_days=30,
        lace_raw_score=lace,
        probability_mean=prob,
        probability_ci95=(max(0.0, prob - 0.05),
                              min(1.0, prob + 0.05)),
        probability_ci_width=0.10,
        contributing_factors=[
            Factor(name="LACE_length_of_stay", raw_value=4.0,
                     lace_points=2, weight=0.25),
            Factor(name="LACE_acuity", raw_value=1.0,
                     lace_points=3, weight=0.30),
            Factor(name="LACE_comorbidity", raw_value=4.0,
                     lace_points=3, weight=0.30),
            Factor(name="LACE_ed_visits_6mo", raw_value=2.0,
                     lace_points=2, weight=0.15),
        ],
        fhir_observations_used=[],
        computed_at=datetime.now(timezone.utc),
        confidence="preferred",
        valid_for_minutes=60,
        valid_until=datetime.now(timezone.utc),
    )


def _build_card(*, action=Action.HOME_WITH_CARE,
                  confidence="medium", risk=0.20,
                  abstain_triggers=None):
    util = UtilityAnalysis(
        action_scores_qaly_weeks={action: 0.8},
        action_scores_ci95={action: (0.6, 1.0)},
        action_costs_usd={action: 0.0},
        dominant_action=action,
        dominance_confidence=0.85,
        reasoning_trace="trace",
    )
    grounding = ClaimGrounding(
        claim_text="patient stable", sub_claims=[],
        overall_verdict="supported",
        context_fingerprint="fp", grounded_at=datetime.now(timezone.utc),
    )
    phi = PHIReport(entities_found=[], entity_count_by_type={},
                      redaction_map={}, risk_level="none")
    audit = AuditBlock(
        request_id="req-001", context_fingerprint="fp",
        tool_trace=[], server_version="x", agent_version="x",
        model_coefficients_version="seed-v1",
        timestamp=datetime.now(timezone.utc),
    )
    return DecisionCard(
        recommendation=Recommendation(action=action, confidence=confidence),
        reasoning=DecisionReasoning(
            risk_estimate=_risk_estimate(prob=risk),
            utility_analysis=util,
        ),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=abstain_triggers,
        audit=audit,
    )


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    """Default: deterministic floor (no LLM call). Tests opt in via override."""
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


# ─────────────────────── Deterministic floor ───────────────────────

def test_deterministic_floor_returns_neutral_approval():
    card = _build_card()
    verdict = llm_judge_critic(card)
    assert verdict.verdict == "approved"
    assert verdict.critic_role == "llm_judge"
    assert "fallback" in verdict.rationale.lower() or \
           "unavailable" in verdict.rationale.lower()


def test_deterministic_floor_never_raises():
    """Floor must hold even when card has unusual / edge fields."""
    card = _build_card(abstain_triggers=[
        AbstainTrigger(type="confidence_interval_too_wide",
                          detail="CI > 0.4")])
    v = llm_judge_critic(card)
    assert v.verdict in {"approved", "downgrade_confidence",
                            "force_abstain", "request_replay"}


# ─────────────────────── LLM-driven path with mocked Ollama ───────────────────────

def test_llm_returns_approved(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: '{"verdict": "approved", '
                                      '"rationale": "All consistent."}')
    v = llm_judge_critic(_build_card())
    assert v.verdict == "approved"
    assert "consistent" in v.rationale.lower()
    assert v.critic_role == "llm_judge"


def test_llm_returns_force_abstain_with_trigger(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: '{"verdict": "force_abstain", '
                                      '"rationale": "discharge_home but risk > 0.4"}')
    v = llm_judge_critic(_build_card(action=Action.DISCHARGE_HOME, risk=0.45))
    assert v.verdict == "force_abstain"
    assert v.abstain_trigger_to_add is not None
    assert v.abstain_trigger_to_add.type == "evidence_insufficient"
    assert "LLM judge" in v.abstain_trigger_to_add.detail


def test_llm_returns_downgrade_with_confidence_after(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(
        mod, "_call_ollama",
        lambda _: '{"verdict": "downgrade_confidence", '
                  '"rationale": "Borderline case", '
                  '"confidence_after": "low"}',
    )
    v = llm_judge_critic(_build_card())
    assert v.verdict == "downgrade_confidence"
    assert v.confidence_after == "low"


def test_llm_returns_request_replay(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: '{"verdict": "request_replay", '
                                      '"rationale": "Need to refetch."}')
    v = llm_judge_critic(_build_card())
    assert v.verdict == "request_replay"


# ─────────────────────── LLM hardening: malformed responses ───────────────────────

def test_invalid_verdict_falls_back_to_approved(monkeypatch):
    """An LLM hallucinating a non-canonical verdict falls back to neutral."""
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: '{"verdict": "definitely_yes", '
                                      '"rationale": "..."}')
    v = llm_judge_critic(_build_card())
    assert v.verdict == "approved"
    assert "fallback" in v.rationale.lower() or \
           "unavailable" in v.rationale.lower()


def test_non_json_response_falls_back_to_approved(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: "I think this is fine.")
    v = llm_judge_critic(_build_card())
    assert v.verdict == "approved"


def test_empty_response_falls_back(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", lambda _: "")
    v = llm_judge_critic(_build_card())
    assert v.verdict == "approved"


def test_response_with_extra_text_around_json_parses(monkeypatch):
    """LLMs often wrap JSON in prose -- the parser must extract."""
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(
        mod, "_call_ollama",
        lambda _: 'Here is my analysis:\n\n'
                    '{"verdict": "approved", "rationale": "ok"}\n\n'
                    'Hope this helps!',
    )
    v = llm_judge_critic(_build_card())
    assert v.verdict == "approved"


def test_truncated_response_falls_back(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: '{"verdict": "approved",')
    v = llm_judge_critic(_build_card())
    assert v.verdict == "approved"
    # Truncated JSON -> parse failure -> fallback path
    assert ("fallback" in v.rationale.lower()
            or "unavailable" in v.rationale.lower())


# ─────────────────────── PHI sanity: prompt cannot exfiltrate raw FHIR ───────────────────────

def test_card_summary_contains_no_raw_phi():
    """The serialized summary used in the prompt must contain only the
    structured features -- never raw patient names, dates, or addresses."""
    card = _build_card()
    summary = mod._summarize_card_for_prompt(card)
    blob = repr(summary).lower()
    # Any of these would be a leak
    for forbidden in ("name", "given", "family", "birthdate",
                          "dob", "ssn", "mrn", "address", "phone", "email"):
        # `name` appears in factor names like "LACE_length_of_stay" -- allow it
        if forbidden == "name":
            continue
        assert forbidden not in blob, \
            f"PHI-adjacent token {forbidden!r} leaked into prompt: {blob}"


def test_summary_contains_lace_breakdown():
    card = _build_card()
    summary = mod._summarize_card_for_prompt(card)
    risk = summary.get("risk_estimate") or {}
    factors = risk.get("contributing_factors") or []
    factor_names = {f["name"] for f in factors}
    assert "LACE_length_of_stay" in factor_names


def test_summary_includes_action_and_confidence():
    card = _build_card(action=Action.HOME_WITH_CARE, confidence="medium")
    summary = mod._summarize_card_for_prompt(card)
    assert summary["action"] == "home_with_care"
    assert summary["confidence"] == "medium"


def test_summary_includes_abstain_triggers_when_present():
    card = _build_card(abstain_triggers=[
        AbstainTrigger(type="out_of_distribution",
                          detail="LACE bucket empty")
    ])
    summary = mod._summarize_card_for_prompt(card)
    triggers = summary.get("abstain_triggers")
    assert triggers and triggers[0]["type"] == "out_of_distribution"


# ─────────────────────── Integration with the critic ensemble ───────────────────────

def test_llm_critic_role_recognized_by_aggregator():
    """The aggregator must accept llm_judge as a valid critic role."""
    from a2a_agent.critique import aggregate_critic_verdicts
    from shared.schemas import CritiqueDecision

    verdicts = [
        CritiqueDecision(verdict="approved", rationale="cs",
                            critic_role="clinical_safety"),
        CritiqueDecision(verdict="approved", rationale="fa",
                            critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ev",
                            critic_role="evidence"),
        CritiqueDecision(verdict="approved", rationale="lj",
                            critic_role="llm_judge"),
    ]
    ensemble = aggregate_critic_verdicts(verdicts, retries_remaining=1)
    assert ensemble.applied_verdict == "approved"
    assert ensemble.n_approved == 4


def test_llm_critic_force_abstain_blocks_ensemble():
    """Even a single LLM force_abstain must cascade to ensemble force_abstain."""
    from a2a_agent.critique import aggregate_critic_verdicts
    from shared.schemas import CritiqueDecision

    verdicts = [
        CritiqueDecision(verdict="approved", rationale="cs",
                            critic_role="clinical_safety"),
        CritiqueDecision(verdict="approved", rationale="fa",
                            critic_role="fairness"),
        CritiqueDecision(verdict="approved", rationale="ev",
                            critic_role="evidence"),
        CritiqueDecision(
            verdict="force_abstain",
            rationale="LLM caught contradiction",
            critic_role="llm_judge",
            abstain_trigger_to_add=AbstainTrigger(
                type="evidence_insufficient",
                detail="LLM judge: discharge_home with risk 0.45"),
        ),
    ]
    ensemble = aggregate_critic_verdicts(verdicts, retries_remaining=1)
    assert ensemble.aggregated_verdict == "force_abstain"
    assert ensemble.applied_verdict == "force_abstain"
