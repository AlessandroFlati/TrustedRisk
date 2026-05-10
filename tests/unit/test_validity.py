"""Unit tests for shared.validity -- temporal-validity windows."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from shared.schemas import (
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
    ToolInvocationTrace,
    UtilityAnalysis,
)
from shared.validity import (
    card_validity_remaining_minutes,
    compute_valid_until,
    compute_validity_window_minutes,
    is_card_valid,
    is_risk_estimate_valid,
)


def _now():
    return datetime.now(timezone.utc)


# ─────────────────────── compute_validity_window_minutes ───────────────────────

@pytest.mark.parametrize("lace,expected_minutes", [
    (0, 24 * 60),
    (3, 24 * 60),
    (5, 24 * 60),
    (6, 12 * 60),
    (8, 12 * 60),
    (11, 12 * 60),
    (12, 4 * 60),
    (15, 4 * 60),
    (19, 4 * 60),
])
def test_validity_window_by_lace(lace, expected_minutes):
    assert compute_validity_window_minutes(lace) == expected_minutes


def test_validity_window_clamps_negative():
    assert compute_validity_window_minutes(-5) == 24 * 60


def test_validity_window_clamps_above_19():
    assert compute_validity_window_minutes(50) == 4 * 60


# ─────────────────────── compute_valid_until ───────────────────────

def test_valid_until_low_risk():
    base = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    expected = base + timedelta(hours=24)
    assert compute_valid_until(base, lace_total=3) == expected


def test_valid_until_high_risk():
    base = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    expected = base + timedelta(hours=4)
    assert compute_valid_until(base, lace_total=15) == expected


def test_valid_until_naive_datetime_treated_as_utc():
    base_naive = datetime(2026, 4, 27, 12, 0)
    result = compute_valid_until(base_naive, lace_total=8)
    assert result.tzinfo is not None
    assert result == base_naive.replace(tzinfo=timezone.utc) + timedelta(hours=12)


# ─────────────────────── is_risk_estimate_valid ───────────────────────

def _risk(*, lace=10, valid_for_min=720, ts=None) -> RiskEstimate:
    ts = ts or _now()
    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="test",
        horizon_days=30,
        lace_raw_score=lace,
        probability_mean=0.18,
        probability_ci95=(0.12, 0.24),
        probability_ci_width=0.12,
        contributing_factors=[Factor(name="L", raw_value=4, lace_points=4, weight=0.4)],
        computed_at=ts,
        valid_for_minutes=valid_for_min,
        valid_until=ts + timedelta(minutes=valid_for_min),
    )


def test_risk_estimate_valid_within_window():
    risk = _risk()
    assert is_risk_estimate_valid(risk) is True


def test_risk_estimate_invalid_after_window():
    past = _now() - timedelta(hours=2)
    risk = _risk(ts=past, valid_for_min=60)  # 1 hour window, computed 2 hours ago
    assert is_risk_estimate_valid(risk) is False


def test_risk_estimate_no_valid_until_legacy_compat():
    """Older artifacts without valid_until are treated as still valid."""
    risk = _risk()
    risk = risk.model_copy(update={"valid_until": None})
    assert is_risk_estimate_valid(risk) is True


def test_risk_estimate_explicit_now():
    base = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    risk = _risk(lace=10, valid_for_min=120, ts=base)
    # 1 hour later -> still valid
    assert is_risk_estimate_valid(risk, now=base + timedelta(hours=1)) is True
    # 3 hours later -> expired
    assert is_risk_estimate_valid(risk, now=base + timedelta(hours=3)) is False


# ─────────────────────── DecisionCard validity ───────────────────────

def _card_with_risk(risk: RiskEstimate) -> DecisionCard:
    util = UtilityAnalysis(
        action_scores_qaly_weeks={Action.HOME_WITH_CARE: 4.5},
        action_scores_ci95={Action.HOME_WITH_CARE: (4.0, 5.0)},
        action_costs_usd={Action.HOME_WITH_CARE: 1500.0},
        dominant_action=Action.HOME_WITH_CARE,
        dominance_confidence=0.8,
        reasoning_trace="ok",
    )
    grounding = ClaimGrounding(
        claim_text="ok", sub_claims=[], overall_verdict="supported",
        context_fingerprint="fp", grounded_at=_now(),
    )
    phi = PHIReport(entities_found=[], entity_count_by_type={}, redaction_map={}, risk_level="none")
    audit = AuditBlock(
        request_id="req-1", context_fingerprint="fp",
        tool_trace=[ToolInvocationTrace(tool="x", invoked_at=_now(),
                                         duration_ms=10, status="ok")],
        server_version="0.1", agent_version="0.1",
        model_coefficients_version="v1", timestamp=_now(),
    )
    return DecisionCard(
        recommendation=Recommendation(action=Action.HOME_WITH_CARE, confidence="high"),
        reasoning=DecisionReasoning(risk_estimate=risk, utility_analysis=util),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=None, audit=audit,
    )


def test_card_valid_when_risk_valid():
    card = _card_with_risk(_risk())
    assert is_card_valid(card) is True


def test_card_invalid_when_risk_expired():
    past = _now() - timedelta(hours=10)
    card = _card_with_risk(_risk(ts=past, valid_for_min=60))
    assert is_card_valid(card) is False


def test_card_invalid_when_no_reasoning():
    risk = _risk()
    card = _card_with_risk(risk)
    card_no_reasoning = card.model_copy(update={"reasoning": None})
    assert is_card_valid(card_no_reasoning) is False


# ─────────────────────── card_validity_remaining_minutes ───────────────────────

def test_remaining_minutes_full_window():
    base = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    risk = _risk(lace=10, valid_for_min=720, ts=base)
    card = _card_with_risk(risk)
    # Right at issue time, full window remaining
    rem = card_validity_remaining_minutes(card, now=base)
    assert 719 <= rem <= 720


def test_remaining_minutes_half_consumed():
    base = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    risk = _risk(lace=10, valid_for_min=120, ts=base)  # 2-hour window
    card = _card_with_risk(risk)
    rem = card_validity_remaining_minutes(card, now=base + timedelta(hours=1))
    assert 59 <= rem <= 60


def test_remaining_minutes_zero_when_expired():
    base = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    risk = _risk(lace=10, valid_for_min=60, ts=base)
    card = _card_with_risk(risk)
    rem = card_validity_remaining_minutes(card, now=base + timedelta(hours=5))
    assert rem == 0


def test_remaining_minutes_no_valid_until():
    """Legacy artifact: returns valid_for_minutes as the answer."""
    risk = _risk()
    risk = risk.model_copy(update={"valid_until": None})
    card = _card_with_risk(risk)
    rem = card_validity_remaining_minutes(card)
    assert rem == 720


# ─────────────────────── Integration: high-LACE patient gets short window ───────────────────────

def test_high_risk_patient_has_4h_window():
    """A LACE 14 patient gets a 4-hour validity window -- sicker patients
    require more frequent re-evaluation."""
    risk = _risk(lace=14, valid_for_min=compute_validity_window_minutes(14))
    assert risk.valid_for_minutes == 240  # 4 hours


def test_low_risk_patient_has_24h_window():
    """A LACE 4 patient gets a 24-hour validity window."""
    minutes = compute_validity_window_minutes(4)
    assert minutes == 1440
