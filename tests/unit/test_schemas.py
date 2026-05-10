"""Unit tests for shared.schemas -- Pydantic contract surface."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.schemas import (
    SUPPORTED_MODEL_NAMES,
    AbstainTrigger,
    Action,
    AuditBlock,
    ClaimGrounding,
    DecisionCard,
    DecisionReasoning,
    DecisionValidation,
    EvidenceSource,
    Factor,
    PHIEntity,
    PHIReport,
    ProbInterval,
    Recommendation,
    RiskEstimate,
    SubClaim,
    ToolInvocationTrace,
    UtilityAnalysis,
)


def _now():
    return datetime.now(timezone.utc)


# ─────────────────────── ProbInterval ───────────────────────

def test_prob_interval_valid():
    p = ProbInterval(mean=0.18, ci95=(0.12, 0.24))
    assert p.mean == 0.18
    assert p.ci95 == (0.12, 0.24)


def test_prob_interval_rejects_lo_above_hi():
    with pytest.raises(ValidationError):
        ProbInterval(mean=0.18, ci95=(0.30, 0.20))


def test_prob_interval_rejects_out_of_unit():
    with pytest.raises(ValidationError):
        ProbInterval(mean=0.18, ci95=(-0.1, 0.5))
    with pytest.raises(ValidationError):
        ProbInterval(mean=0.5, ci95=(0.0, 1.5))


def test_prob_interval_mean_at_bounds():
    ProbInterval(mean=0.0, ci95=(0.0, 0.5))
    ProbInterval(mean=1.0, ci95=(0.5, 1.0))


# ─────────────────────── Factor / RiskEstimate ───────────────────────

def test_factor_valid():
    f = Factor(name="LACE_L", raw_value=5.0, lace_points=4, weight=0.3)
    assert f.lace_points == 4


def test_factor_rejects_lace_above_7():
    with pytest.raises(ValidationError):
        Factor(name="x", raw_value=1.0, lace_points=8, weight=0.5)


def test_factor_rejects_weight_above_1():
    with pytest.raises(ValidationError):
        Factor(name="x", raw_value=1.0, lace_points=3, weight=1.2)


def test_risk_estimate_valid():
    r = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="readmission-test-v1",
        horizon_days=30,
        lace_raw_score=14,
        probability_mean=0.282,
        probability_ci95=(0.255, 0.309),
        probability_ci_width=0.054,
        contributing_factors=[Factor(name="LACE_L", raw_value=5, lace_points=4, weight=0.3)],
        fhir_observations_used=["obs-1", "obs-2"],
        computed_at=_now(),
    )
    assert r.confidence == "preferred"
    assert r.lace_raw_score == 14


def test_risk_estimate_rejects_lace_above_19():
    with pytest.raises(ValidationError):
        RiskEstimate(
            model_name="lace-plus-bayesian-v1",
            model_version="v",
            horizon_days=30,
            lace_raw_score=20,
            probability_mean=0.5,
            probability_ci95=(0.4, 0.6),
            probability_ci_width=0.2,
            contributing_factors=[],
            computed_at=_now(),
        )


def test_supported_model_names_contains_v1():
    assert "lace-plus-bayesian-v1" in SUPPORTED_MODEL_NAMES


# ─────────────────────── UtilityAnalysis ───────────────────────

def test_utility_analysis_valid():
    u = UtilityAnalysis(
        action_scores_qaly_weeks={Action.HOME_WITH_CARE: 4.5, Action.SNF: 4.2},
        action_scores_ci95={Action.HOME_WITH_CARE: (4.0, 5.0), Action.SNF: (3.8, 4.6)},
        action_costs_usd={Action.HOME_WITH_CARE: 1500.0, Action.SNF: 18000.0},
        dominant_action=Action.HOME_WITH_CARE,
        dominance_confidence=0.82,
        reasoning_trace="MC samples favored home_with_care 82% of draws",
    )
    assert u.dominant_action == Action.HOME_WITH_CARE


def test_utility_analysis_inconclusive():
    u = UtilityAnalysis(
        action_scores_qaly_weeks={Action.HOME_WITH_CARE: 4.5},
        action_scores_ci95={Action.HOME_WITH_CARE: (4.0, 5.0)},
        action_costs_usd={Action.HOME_WITH_CARE: 1500.0},
        dominant_action="INCONCLUSIVE",
        dominance_confidence=0.45,
        reasoning_trace="No action dominates above the 60% threshold",
    )
    assert u.dominant_action == "INCONCLUSIVE"


# ─────────────────────── ClaimGrounding ───────────────────────

def test_claim_grounding_valid():
    c = ClaimGrounding(
        claim_text="patient is stable for discharge",
        sub_claims=[
            SubClaim(
                text="vital signs stable",
                atomic_kind="vital",
                verdict="supported",
                confidence=0.9,
                evidence_sources=[
                    EvidenceSource(
                        source_type="fhir_observation",
                        source_id="obs-bp-recent",
                        excerpt="BP 120/80 within last 6h",
                        relevance_score=0.85,
                        recency_days=0,
                    )
                ],
            )
        ],
        overall_verdict="supported",
        context_fingerprint="abc123",
        grounded_at=_now(),
    )
    assert len(c.sub_claims) == 1


def test_subclaim_unsupported_with_reason():
    sc = SubClaim(
        text="renal function adequate",
        atomic_kind="general",
        verdict="unsupported",
        confidence=0.4,
        reason_if_unsupported="No creatinine in last 48h",
    )
    assert sc.reason_if_unsupported


# ─────────────────────── PHIReport ───────────────────────

def test_phi_report_valid():
    r = PHIReport(
        entities_found=[
            PHIEntity(type="MRN", start=10, end=20, confidence=0.95, suggested_replacement="[MRN]")
        ],
        entity_count_by_type={"MRN": 1},
        redaction_map={"MRN-123456": "[MRN]"},
        risk_level="high",
    )
    assert r.risk_level == "high"
    assert r.entities_found[0].type == "MRN"


def test_phi_report_empty_text():
    r = PHIReport(entities_found=[], entity_count_by_type={}, redaction_map={}, risk_level="none")
    assert r.risk_level == "none"


def test_phi_entity_rejects_negative_offset():
    with pytest.raises(ValidationError):
        PHIEntity(type="EMAIL", start=-1, end=10, confidence=0.7, suggested_replacement="[EMAIL]")


# ─────────────────────── AbstainTrigger ───────────────────────

def test_abstain_trigger_ci_width():
    a = AbstainTrigger(
        type="confidence_interval_too_wide",
        detail="CI width 0.18 exceeds threshold 0.15",
        threshold_exceeded={"observed_width": 0.18, "threshold": 0.15},
    )
    assert a.type == "confidence_interval_too_wide"


def test_abstain_trigger_rejects_unknown_type():
    with pytest.raises(ValidationError):
        AbstainTrigger(type="some_invalid_type", detail="x")


# ─────────────────────── DecisionCard composition ───────────────────────

def test_decision_card_composes():
    risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="v1",
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
        dominance_confidence=0.8,
        reasoning_trace="ok",
    )
    grounding = ClaimGrounding(
        claim_text="discharge plan ok",
        sub_claims=[],
        overall_verdict="supported",
        context_fingerprint="fp",
        grounded_at=_now(),
    )
    phi = PHIReport(entities_found=[], entity_count_by_type={}, redaction_map={}, risk_level="none")
    audit = AuditBlock(
        request_id="req-1",
        context_fingerprint="fp",
        tool_trace=[
            ToolInvocationTrace(tool="compute_readmission_risk", invoked_at=_now(), duration_ms=120, status="ok")
        ],
        server_version="trustedrisk-0.1",
        agent_version="agent-0.1",
        model_coefficients_version="v1",
        timestamp=_now(),
    )
    card = DecisionCard(
        recommendation=Recommendation(action=Action.HOME_WITH_CARE, confidence="high"),
        reasoning=DecisionReasoning(risk_estimate=risk, utility_analysis=util),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=None,
        audit=audit,
    )
    assert card.recommendation.action == Action.HOME_WITH_CARE
    assert card.audit.clinical_disclaimer.startswith("TrustedRisk is a clinical decision support")
