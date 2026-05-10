"""Functional tests for the EVAL pipeline: HEDIS + Hospital Compare +
Schwartz quality + comparative-effectiveness simulation.

These chain together to answer "is this agent actually delivering quality
care?" -- the Impact-criterion answer for the challenge submission.
"""
from __future__ import annotations

import pytest

from a2a_agent.eval_quality import (
    compute_comparative_effectiveness,
    compute_hedis_score,
    compute_hospital_compare_benchmark,
    compute_schwartz_quality_score,
)


# ─────────────────────── HEDIS on canonical eligible cohort ───────────────────────

def test_hedis_full_cohort_yields_eligible_measures(hedis_eligible_cohort_bundle):
    r = compute_hedis_score(hedis_eligible_cohort_bundle)
    eligible_ids = {m.measure_id for m in r.measures if m.eligible}
    # 60yo female with DM + HTN should be eligible for CDC, CBP, COL, BCS
    assert "CDC-HM2" in eligible_ids
    assert "CBP" in eligible_ids
    assert "COL" in eligible_ids
    assert "BCS" in eligible_ids


def test_hedis_compliance_for_cohort_with_full_screening_history(
    hedis_eligible_cohort_bundle,
):
    r = compute_hedis_score(hedis_eligible_cohort_bundle)
    # The fixture has a recent HbA1c (7.4) + recent BP + recent mammo ->
    # those measures should be in compliance.
    cdc = next(m for m in r.measures if m.measure_id == "CDC-HM2")
    cbp = next(m for m in r.measures if m.measure_id == "CBP")
    bcs = next(m for m in r.measures if m.measure_id == "BCS")
    assert cdc.in_compliance is True
    assert cbp.in_compliance is True
    assert bcs.in_compliance is True


def test_hedis_composite_score_realistic(hedis_eligible_cohort_bundle):
    r = compute_hedis_score(hedis_eligible_cohort_bundle)
    assert r.n_measures_evaluated >= 4
    assert r.composite_score_pct >= 50.0


def test_hedis_male_patient_reroutes_eligibility(chf_bundle):
    """The CHF fixture is a 75yo female with DM. Verify the eligibility
    routing differs from the all-screened cohort."""
    # Manually flip the patient gender to male in a copy
    import copy
    bundle = copy.deepcopy(chf_bundle)
    for entry in bundle["entry"]:
        r = entry["resource"]
        if r.get("resourceType") == "Patient":
            r["gender"] = "male"
            break
    report = compute_hedis_score(bundle)
    # BCS (female-only) and CCS (female-only) should NOT be eligible
    bcs = next(m for m in report.measures if m.measure_id == "BCS")
    ccs = next(m for m in report.measures if m.measure_id == "CCS")
    assert bcs.eligible is False
    assert ccs.eligible is False


# ─────────────────────── Hospital Compare benchmark ───────────────────────

def test_hospital_compare_excellent_kpis_high_composite():
    r = compute_hospital_compare_benchmark(
        institution_label="Hospital A (best-in-class academic)",
        institution_kpis={
            "readmission_rate_30d_pct": 11.0,
            "ed_revisit_rate_72h_pct": 2.5,
            "hcahps_overall_pct": 86,
            "mortality_rate_30d_pct": 9.0,
            "abstention_rate_pct": 2.0,
        },
        peer_tier="academic_major",
    )
    assert r.composite_quality_score > 60
    assert r.percentile_rank["readmission_rate_30d_pct"] >= 60


def test_hospital_compare_below_median_low_composite():
    r = compute_hospital_compare_benchmark(
        institution_label="Hospital B (below median)",
        institution_kpis={
            "readmission_rate_30d_pct": 22.0,
            "ed_revisit_rate_72h_pct": 7.5,
            "hcahps_overall_pct": 58,
        },
        peer_tier="community_large",
    )
    assert r.composite_quality_score < 30


def test_hospital_compare_handles_partial_kpis():
    """A subset of KPIs should still produce a valid composite."""
    r = compute_hospital_compare_benchmark(
        institution_label="X",
        institution_kpis={"readmission_rate_30d_pct": 14.0},
        peer_tier="rural",
    )
    assert "readmission_rate_30d_pct" in r.deltas_vs_median
    assert 0.0 <= r.composite_quality_score <= 100.0


# ─────────────────────── Schwartz JAMIA 2017 ───────────────────────

def test_schwartz_full_decision_card_grades_high(chf_decision_card):
    r = compute_schwartz_quality_score(chf_decision_card)
    assert r.grade in ("A", "B")
    assert r.composite_score > 0.65


def test_schwartz_decision_card_with_abstain_keeps_actionable_score(
    chf_decision_card,
):
    """A card with abstain triggers should still score ≥ moderately on
    actionable (the abstain trigger IS the actionable signal -- defer)."""
    import copy
    card = copy.deepcopy(chf_decision_card)
    card["abstain"] = [{"type": "evidence_insufficient",
                            "detail": "insufficient grounding"}]
    r = compute_schwartz_quality_score(card)
    assert r.actionable_score >= 0.5


def test_schwartz_force_abstain_critic_flags_trustworthy(chf_decision_card):
    import copy
    card = copy.deepcopy(chf_decision_card)
    card["self_critique"] = {"verdict": "force_abstain",
                                  "rationale": "uncertain", "critic_role": "structural"}
    r = compute_schwartz_quality_score(card)
    # Trustworthy must drop relative to the approved-version baseline
    baseline = compute_schwartz_quality_score(chf_decision_card)
    assert r.trustworthy_score <= baseline.trustworthy_score


# ─────────────────────── Comparative-effectiveness simulator ───────────────────────

def test_comparative_recovers_known_effect_at_n_5000():
    """With n=5000 + true RRR=0.30, ARR should be close to baseline*RRR."""
    r = compute_comparative_effectiveness(
        n_patients=5000,
        baseline_event_rate=0.20,
        intervention_relative_risk_reduction=0.30,
        seed=42,
    )
    # Expected ARR ≈ 0.20 * 0.30 = 0.06
    assert 0.04 <= r.arr_30d <= 0.08
    assert r.nnt is not None and 12 <= r.nnt <= 25


def test_comparative_p_value_significant_for_strong_effect():
    r = compute_comparative_effectiveness(
        n_patients=2000, baseline_event_rate=0.30,
        intervention_relative_risk_reduction=0.50, seed=7,
    )
    assert r.p_value < 0.001


def test_comparative_p_value_non_significant_for_zero_effect():
    r = compute_comparative_effectiveness(
        n_patients=2000, baseline_event_rate=0.20,
        intervention_relative_risk_reduction=0.0, seed=42,
    )
    assert r.p_value > 0.05


# ─────────────────────── End-to-end EVAL chain ───────────────────────

def test_full_eval_chain_for_a_realistic_institution(
    hedis_eligible_cohort_bundle,
    chf_decision_card,
):
    """One realistic institution: a HEDIS-eligible patient -> HEDIS score ->
    institution KPIs -> Hospital Compare -> DecisionCard -> Schwartz score ->
    comparative-effectiveness simulation. All four eval surfaces should
    produce well-formed reports without errors."""
    hedis = compute_hedis_score(hedis_eligible_cohort_bundle)
    hc = compute_hospital_compare_benchmark(
        institution_label="Demo Hospital",
        institution_kpis={
            "readmission_rate_30d_pct": 14.5,
            "ed_revisit_rate_72h_pct": 4.2,
            "hcahps_overall_pct": 73,
        },
        peer_tier="community_large",
    )
    schwartz = compute_schwartz_quality_score(chf_decision_card)
    comp = compute_comparative_effectiveness(
        n_patients=1000,
        baseline_event_rate=hc.institution_kpis["readmission_rate_30d_pct"]
        / 100.0,
        intervention_relative_risk_reduction=0.25,
        seed=42,
    )
    # All four reports produced + composite scores in [0, 100] / [0, 1]
    assert hedis.composite_score_pct >= 0
    assert 0 <= hc.composite_quality_score <= 100
    assert schwartz.grade in ("A", "B", "C", "D", "F")
    assert comp.arr_30d >= 0
