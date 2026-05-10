"""IMPACT-1 tests for compute_expected_value_of_intervention."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mcp_server.tools.expected_value_of_intervention import (
    compute_expected_value_of_intervention,
)
from shared.schemas import (
    Factor,
    InterventionEvidence,
    RiskEstimate,
)


def _run(coro):
    return asyncio.run(coro)


def _evidence(**overrides):
    base = {
        "name": "Pharmacist-led discharge counseling",
        "relative_risk_reduction": 0.30,
        "cost_per_patient_usd": 75.0,
        "qaly_gained_per_avoided_event": 0.05,
        "horizon_days": 30,
        "evidence_grade": "A",
        "citation": "Schnipper JL, et al. Arch Intern Med 2006;166:565.",
    }
    base.update(overrides)
    return InterventionEvidence(**base)


def _risk_estimate(prob: float = 0.20):
    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="0.7.0",
        outcome_id="readmission_30d",
        horizon_days=30,
        lace_raw_score=10,
        probability_mean=prob,
        probability_ci95=(max(0.0, prob - 0.05), min(1.0, prob + 0.05)),
        probability_ci_width=0.10,
        contributing_factors=[
            Factor(name="LACE_length_of_stay", raw_value=6.0,
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


# ─────────────────────── Inputs / baseline resolution ───────────────────────

def test_baseline_from_explicit_probability():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(),
        baseline_event_probability=0.20,
        cohort_size=100,
    ))
    assert result.baseline_event_probability == pytest.approx(0.20)


def test_baseline_from_risk_estimate():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(),
        risk_estimate=_risk_estimate(prob=0.18),
        cohort_size=100,
    ))
    assert result.baseline_event_probability == pytest.approx(0.18)


def test_baseline_missing_raises():
    with pytest.raises(ValueError, match="baseline_event_probability"):
        _run(compute_expected_value_of_intervention(
            intervention=_evidence(),
            cohort_size=100,
        ))


def test_baseline_out_of_range_raises():
    with pytest.raises(ValueError, match="baseline_event_probability"):
        _run(compute_expected_value_of_intervention(
            intervention=_evidence(),
            baseline_event_probability=1.5,
            cohort_size=100,
        ))


def test_dict_inputs_are_accepted():
    """MCP transports inputs as dicts -- the tool must coerce them."""
    result = _run(compute_expected_value_of_intervention(
        intervention={
            "name": "test", "relative_risk_reduction": 0.25,
            "cost_per_patient_usd": 50.0, "evidence_grade": "B",
        },
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.intervention_name == "test"


# ─────────────────────── ARR / NNT math ───────────────────────

def test_arr_from_relative_risk_reduction():
    """ARR = baseline * RRR."""
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.25),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.absolute_risk_reduction == pytest.approx(0.05)
    assert result.number_needed_to_treat == pytest.approx(20.0)


def test_arr_explicit_overrides_rrr():
    """When both provided, ARR wins (less assumption-laden)."""
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(absolute_risk_reduction=0.08,
                                  relative_risk_reduction=0.50),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.absolute_risk_reduction == pytest.approx(0.08)
    assert result.number_needed_to_treat == pytest.approx(12.5)


def test_arr_clamped_to_baseline():
    """ARR can never exceed baseline probability."""
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(absolute_risk_reduction=0.50),
        baseline_event_probability=0.10, cohort_size=100,
    ))
    assert result.absolute_risk_reduction == pytest.approx(0.10)


def test_zero_arr_yields_no_nnt():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.0,
                                  absolute_risk_reduction=None),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.absolute_risk_reduction == 0.0
    assert result.number_needed_to_treat is None


def test_intervention_without_any_effect_estimate_raises():
    with pytest.raises(ValueError, match="relative_risk_reduction"):
        _run(compute_expected_value_of_intervention(
            intervention=InterventionEvidence(
                name="empty", cost_per_patient_usd=10.0,
            ),
            baseline_event_probability=0.20, cohort_size=100,
        ))


# ─────────────────────── Cohort scaling + totals ───────────────────────

def test_cohort_scaling():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.30,
                                  cost_per_patient_usd=100.0),
        baseline_event_probability=0.20, cohort_size=500,
        avoided_event_cost_usd=10_000.0,
    ))
    # ARR = 0.20 * 0.30 = 0.06; events avoided = 500 * 0.06 = 30
    assert result.expected_events_avoided == pytest.approx(30.0)
    assert result.intervention_cost_total_usd == pytest.approx(50_000.0)
    assert result.avoided_event_cost_total_usd == pytest.approx(300_000.0)
    assert result.net_cost_total_usd == pytest.approx(-250_000.0)


# ─────────────────────── Decision ladder ───────────────────────

def test_cost_saving_when_avoided_cost_exceeds_intervention_cost():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.30,
                                  cost_per_patient_usd=100.0,
                                  evidence_grade="A"),
        baseline_event_probability=0.20, cohort_size=100,
        avoided_event_cost_usd=14_000.0,
    ))
    # net = 100 * 100 - 6 * 14000 = 10000 - 84000 = -74000 -> cost_saving
    assert result.decision == "cost_saving"
    assert result.net_cost_total_usd < 0


def test_cost_effective_below_wtp_threshold():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.05,
                                  cost_per_patient_usd=200.0,
                                  qaly_gained_per_avoided_event=0.02,
                                  evidence_grade="A"),
        baseline_event_probability=0.10, cohort_size=100,
        avoided_event_cost_usd=2_000.0,
        wtp_threshold_per_qaly_usd=100_000.0,
    ))
    # ARR=0.005, events=0.5, intv_cost=20000, avoided=1000, net=19000
    # qaly = 0.5 * 0.02 = 0.01, ICER = 19000/0.01 = 1.9M -> not_cost_effective
    assert result.decision == "not_cost_effective"


def test_cost_effective_with_qaly_below_threshold():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.40,
                                  cost_per_patient_usd=300.0,
                                  qaly_gained_per_avoided_event=0.5,
                                  evidence_grade="A"),
        baseline_event_probability=0.20, cohort_size=100,
        avoided_event_cost_usd=2_000.0,
        wtp_threshold_per_qaly_usd=100_000.0,
    ))
    # ARR=0.08, events=8, intv_cost=30000, avoided=16000, net=14000
    # qaly = 8 * 0.5 = 4, ICER = 14000/4 = 3500/QALY -> cost_effective
    assert result.decision == "cost_effective"


def test_uncertain_evidence_blocks_decision():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.30,
                                  evidence_grade="C"),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.decision == "uncertain_evidence"


def test_expert_opinion_grade_recommends_abstain():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.30,
                                  evidence_grade="expert_opinion"),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.decision == "uncertain_evidence"
    assert result.abstain_recommended is True


def test_dominated_when_zero_effect():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.0,
                                  absolute_risk_reduction=0.0,
                                  evidence_grade="A"),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.decision == "dominated"


# ─────────────────────── ICER / cost-per-event ───────────────────────

def test_cost_per_qaly_computed_when_qaly_provided():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(qaly_gained_per_avoided_event=0.05),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.cost_per_qaly_usd is not None


def test_cost_per_qaly_none_when_qaly_missing():
    ev = _evidence()
    ev = ev.model_copy(update={"qaly_gained_per_avoided_event": None})
    result = _run(compute_expected_value_of_intervention(
        intervention=ev,
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert result.cost_per_qaly_usd is None


def test_cost_per_event_avoided():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.30,
                                  cost_per_patient_usd=100.0),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    # 100 * 100 / (100 * 0.06) = 10000 / 6 = ~1666.67
    assert result.cost_per_event_avoided_usd == pytest.approx(1_666.67, rel=1e-3)


# ─────────────────────── Validation guards ───────────────────────

def test_negative_cohort_raises():
    with pytest.raises(ValueError, match="cohort_size"):
        _run(compute_expected_value_of_intervention(
            intervention=_evidence(),
            baseline_event_probability=0.20, cohort_size=0,
        ))


def test_negative_event_cost_raises():
    with pytest.raises(ValueError, match="avoided_event_cost"):
        _run(compute_expected_value_of_intervention(
            intervention=_evidence(),
            baseline_event_probability=0.20, cohort_size=100,
            avoided_event_cost_usd=-1.0,
        ))


def test_references_are_populated():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert any("Sanders" in r for r in result.references)
    assert any("Neumann" in r for r in result.references)


def test_rationale_contains_key_numbers():
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.30,
                                  cost_per_patient_usd=100.0,
                                  evidence_grade="A"),
        baseline_event_probability=0.20, cohort_size=100,
    ))
    assert "0.20" in result.rationale or "0.200" in result.rationale
    assert "NNT" in result.rationale or "ARR" in result.rationale


# ─────────────────────── Integration: chains with RiskEstimate ───────────────────────

def test_chains_after_risk_estimate():
    """End-to-end: RiskEstimate -> CEA -- the chained workflow the agent runs."""
    risk = _risk_estimate(prob=0.25)
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.20,
                                  cost_per_patient_usd=80.0,
                                  evidence_grade="A"),
        risk_estimate=risk,
        cohort_size=200,
        avoided_event_cost_usd=14_000.0,
    ))
    assert result.baseline_event_probability == pytest.approx(0.25)
    # ARR = 0.05, events = 10, intv = 16000, avoided = 140000 -> cost_saving
    assert result.decision == "cost_saving"


def test_chains_with_dict_risk_estimate():
    """The tool must accept a dict-coerced RiskEstimate too (MCP transport)."""
    risk = _risk_estimate(prob=0.25).model_dump(mode="json")
    result = _run(compute_expected_value_of_intervention(
        intervention=_evidence(relative_risk_reduction=0.20),
        risk_estimate=risk,
        cohort_size=200,
    ))
    assert result.baseline_event_probability == pytest.approx(0.25)
