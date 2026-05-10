"""IMPACT-2 unit tests for the cohort impact aggregator + playground endpoint."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from a2a_agent.impact_aggregator import aggregate_impact_kpis


def _card(*, action: str | None = "discharge_home",
            confidence: str = "high",
            risk: float = 0.20,
            critique_verdict: str | None = None) -> dict:
    """Build a minimal DecisionCard-shaped dict."""
    rec: dict | None
    if action is None:
        rec = None
    else:
        rec = {"action": action, "confidence": confidence}
    return {
        "recommendation": rec,
        "reasoning": {
            "risk_estimate": {"probability_mean": risk},
        },
        "validation": {},
        "audit": {"tools": []},
        "abstain": [] if action is not None else [{"trigger": "abstain"}],
        "self_critique": ({"verdict": critique_verdict}
                            if critique_verdict else None),
    }


# ─────────────────────── Empty / singleton ───────────────────────

def test_empty_input_returns_zeros():
    k = aggregate_impact_kpis(decisions=[])
    assert k.n_decisions == 0
    assert k.n_with_recommendation == 0
    assert k.n_abstained == 0
    assert k.estimated_events_avoided == 0.0
    assert k.estimated_cost_avoided_usd == 0.0
    assert k.action_counts == {}


def test_single_discharge_home_no_intervention():
    k = aggregate_impact_kpis(decisions=[_card(action="discharge_home", risk=0.10)])
    assert k.n_decisions == 1
    assert k.n_with_recommendation == 1
    assert k.action_counts == {"discharge_home": 1}
    # No intervention -> zero events avoided
    assert k.estimated_events_avoided == 0.0


def test_single_intervention_yields_arr():
    k = aggregate_impact_kpis(
        decisions=[_card(action="home_with_care", risk=0.40)],
        intervention_relative_risk_reduction=0.25,
        avoided_event_cost_usd=10_000.0,
    )
    # ARR per patient = 0.40 * 0.25 = 0.10
    assert k.estimated_events_avoided == pytest.approx(0.10)
    assert k.estimated_cost_avoided_usd == pytest.approx(1_000.0)


# ─────────────────────── Action distribution ───────────────────────

def test_action_distribution_counted():
    cards = [
        _card(action="discharge_home"),
        _card(action="discharge_home"),
        _card(action="home_with_care"),
        _card(action="snf"),
        _card(action="continued_admission"),
    ]
    k = aggregate_impact_kpis(decisions=cards)
    assert k.action_counts == {
        "discharge_home": 2,
        "home_with_care": 1,
        "snf": 1,
        "continued_admission": 1,
    }


def test_abstain_counted_separately():
    cards = [_card(action="discharge_home"), _card(action=None)]
    k = aggregate_impact_kpis(decisions=cards)
    assert k.n_with_recommendation == 1
    assert k.n_abstained == 1


def test_confidence_distribution():
    cards = [
        _card(action="home_with_care", confidence="high"),
        _card(action="home_with_care", confidence="medium"),
        _card(action="snf", confidence="medium"),
    ]
    k = aggregate_impact_kpis(decisions=cards)
    assert k.confidence_counts == {"high": 1, "medium": 2}


# ─────────────────────── Critic flags ───────────────────────

def test_critic_force_abstain_counted():
    cards = [
        _card(critique_verdict="force_abstain"),
        _card(critique_verdict="force_abstain"),
        _card(critique_verdict="downgrade_confidence"),
        _card(critique_verdict="approved"),
    ]
    k = aggregate_impact_kpis(decisions=cards)
    assert k.n_critic_force_abstain == 2
    assert k.n_critic_downgrade == 1


# ─────────────────────── Risk averages ───────────────────────

def test_avg_baseline_risk_correct():
    cards = [_card(risk=0.20), _card(risk=0.30), _card(risk=0.10)]
    k = aggregate_impact_kpis(decisions=cards)
    assert k.avg_baseline_risk == pytest.approx(0.20)


def test_post_intervention_risk_lower_than_baseline():
    cards = [
        _card(action="home_with_care", risk=0.40),
        _card(action="home_with_care", risk=0.50),
    ]
    k = aggregate_impact_kpis(
        decisions=cards,
        intervention_relative_risk_reduction=0.30,
    )
    # baseline = 0.45, post = 0.45 * 0.7 = 0.315
    assert k.avg_baseline_risk == pytest.approx(0.45)
    assert k.avg_post_intervention_risk == pytest.approx(0.315, rel=1e-3)


def test_post_intervention_risk_unchanged_for_discharge_home():
    cards = [_card(action="discharge_home", risk=0.20),
              _card(action="discharge_home", risk=0.30)]
    k = aggregate_impact_kpis(decisions=cards,
                                intervention_relative_risk_reduction=0.25)
    # No intervention applied -> post == baseline
    assert k.avg_post_intervention_risk == pytest.approx(k.avg_baseline_risk)


# ─────────────────────── Risk extraction fallback ───────────────────────

def test_falls_back_to_audit_tool_outputs():
    """If `reasoning.risk_estimate` is missing, the aggregator still finds
    the risk in `audit.tools[*].outputs.probability_mean`."""
    card = {
        "recommendation": {"action": "home_with_care", "confidence": "medium"},
        "reasoning": {},
        "audit": {
            "tools": [
                {"tool": "compute_readmission_risk",
                 "outputs": {"probability_mean": 0.55}}
            ]
        },
        "abstain": [], "validation": {},
    }
    k = aggregate_impact_kpis(
        decisions=[card],
        intervention_relative_risk_reduction=0.30,
    )
    assert k.avg_baseline_risk == pytest.approx(0.55)
    assert k.estimated_events_avoided == pytest.approx(0.55 * 0.30)


# ─────────────────────── Validation ───────────────────────

def test_invalid_rrr_raises():
    with pytest.raises(ValueError, match="intervention_relative_risk_reduction"):
        aggregate_impact_kpis(
            decisions=[],
            intervention_relative_risk_reduction=1.5,
        )


def test_negative_event_cost_raises():
    with pytest.raises(ValueError, match="avoided_event_cost"):
        aggregate_impact_kpis(
            decisions=[],
            avoided_event_cost_usd=-1.0,
        )


# ─────────────────────── Schema invariants ───────────────────────

def test_kpis_serialize_to_json():
    cards = [_card(action="home_with_care", risk=0.40),
              _card(action="discharge_home", risk=0.10)]
    k = aggregate_impact_kpis(decisions=cards)
    payload = k.model_dump(mode="json")
    assert "n_decisions" in payload
    assert "estimated_cost_avoided_usd" in payload
    assert payload["generated_at_iso"]
    # ISO timestamp parses cleanly
    datetime.fromisoformat(payload["generated_at_iso"])


def test_caveat_present():
    k = aggregate_impact_kpis(decisions=[_card()])
    assert "point estimates" in k.confidence_caveat.lower()
