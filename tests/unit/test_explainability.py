"""Phase 7.4 -- explainability depth tests."""

from __future__ import annotations

import pytest

from a2a_agent.explainability import (
    compute_ale_plot,
    compute_concept_bottleneck_rationale,
    compute_constitutional_critic_check,
    compute_lace_shap_attribution,
)


# ─────────────────────── SHAP ───────────────────────


def test_shap_attributions_sum_to_prediction_minus_baseline():
    out = compute_lace_shap_attribution(
        lace_components={"L": 5, "A": 3, "C": 4, "E": 2},
    )
    # Sum of attributions equals predicted - baseline (within rounding)
    diff = out.predicted_probability - out.cohort_baseline
    assert abs(out.sum_of_attributions - diff) < 1e-3


def test_shap_baseline_lace_yields_zero_attribution():
    out = compute_lace_shap_attribution(
        lace_components={"L": 3.5, "A": 1.8, "C": 2.3, "E": 0.9},
    )
    # Roughly zero (each component at baseline -> ~ no contribution)
    for a in out.attributions:
        assert abs(a.attribution) < 0.05


def test_shap_high_lace_yields_positive_attributions():
    out = compute_lace_shap_attribution(
        lace_components={"L": 7, "A": 3, "C": 5, "E": 4},
    )
    # All components above baseline -> all attributions positive
    for a in out.attributions:
        assert a.attribution > 0


def test_shap_features_named_lace_components():
    out = compute_lace_shap_attribution(
        lace_components={"L": 5, "A": 3, "C": 4, "E": 2},
    )
    feats = {a.feature for a in out.attributions}
    assert feats == {"LACE.L", "LACE.A", "LACE.C", "LACE.E"}


def test_shap_uses_supplied_predicted_probability():
    out = compute_lace_shap_attribution(
        lace_components={"L": 5, "A": 3, "C": 4, "E": 2},
        predicted_probability=0.42,
    )
    assert out.predicted_probability == 0.42


# ─────────────────────── ALE ───────────────────────


def test_ale_plot_returns_monotonic_for_l_component():
    plot = compute_ale_plot("L")
    vals = [p.ale_value for p in plot.points]
    assert vals == sorted(vals)   # monotonic increasing


def test_ale_plot_handles_lower_case_and_lace_prefix():
    a = compute_ale_plot("l")
    b = compute_ale_plot("LACE.L")
    assert a.feature == b.feature == "LACE.L"


def test_ale_plot_unknown_feature_raises():
    with pytest.raises(ValueError, match="unknown LACE component"):
        compute_ale_plot("X")


def test_ale_plot_default_range_per_component():
    plot = compute_ale_plot("E")
    feature_values = [p.feature_value for p in plot.points]
    assert feature_values == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_ale_plot_custom_range():
    plot = compute_ale_plot("L", value_range=(2, 5))
    assert len(plot.points) == 4


# ─────────────────────── Concept-bottleneck ───────────────────────


def test_concept_bottleneck_extracts_high_acuity():
    card = {
        "recommendation": {"action": "snf", "confidence": "preferred"},
        "reasoning": {
            "risk_estimate": {
                "lace_raw_score": 14, "probability_mean": 0.34,
            },
        },
        "abstain": [], "validation": {}, "self_critique": {},
    }
    out = compute_concept_bottleneck_rationale(card)
    labels = {c.label for c in out.concepts}
    assert "high_acuity" in labels


def test_concept_bottleneck_extracts_polypharmacy():
    card = {
        "recommendation": {"action": "discharge_home"},
        "reasoning": {"risk_estimate": {"lace_raw_score": 5,
                                                 "probability_mean": 0.10}},
        "medications": [{"name": f"med-{i}"} for i in range(8)],
        "abstain": [], "validation": {}, "self_critique": {},
    }
    out = compute_concept_bottleneck_rationale(card)
    labels = {c.label for c in out.concepts}
    assert "polypharmacy_present" in labels


def test_concept_bottleneck_extracts_subgroup_drift():
    card = {
        "recommendation": {"action": "discharge_home"},
        "reasoning": {"risk_estimate": {"lace_raw_score": 8,
                                                 "probability_mean": 0.16}},
        "validation": {"fairness": {"max_subgroup_drift": 0.22}},
        "abstain": [], "self_critique": {},
    }
    out = compute_concept_bottleneck_rationale(card)
    labels = {c.label for c in out.concepts}
    assert "subgroup_calibration_drift_detected" in labels


def test_concept_bottleneck_default_when_nothing_special():
    card = {
        "recommendation": {"action": "discharge_home"},
        "reasoning": {}, "abstain": [], "validation": {},
        "self_critique": {},
    }
    out = compute_concept_bottleneck_rationale(card)
    assert out.n_concepts >= 1


# ─────────────────────── Constitutional critic ───────────────────────


def test_constitutional_pass_on_clean_card():
    card = {
        "recommendation": {"action": "discharge_home"},
        "reasoning": {
            "risk_estimate": {
                "probability_ci95": [0.10, 0.18],
                "valid_until": "2026-04-30T12:00:00Z",
            },
        },
        "validation": {
            "phi_check": {"risk_level": "low"},
            "grounding": {"overall_verdict": "grounded"},
            "fairness": {"max_subgroup_drift": 0.05},
        },
        "abstain": [], "audit": {},
    }
    out = compute_constitutional_critic_check(card)
    assert out.overall_verdict == "pass"
    assert out.n_violations == 0


def test_constitutional_blocks_on_high_phi_risk():
    card = {
        "recommendation": {"action": "discharge_home"},
        "validation": {"phi_check": {"risk_level": "high"}},
        "reasoning": {}, "abstain": [],
    }
    out = compute_constitutional_critic_check(card)
    assert out.overall_verdict == "block"
    assert out.n_violations >= 1


def test_constitutional_blocks_on_subgroup_drift_above_30pct():
    card = {
        "recommendation": {"action": "discharge_home"},
        "validation": {"fairness": {"max_subgroup_drift": 0.42}},
        "reasoning": {}, "abstain": [],
    }
    out = compute_constitutional_critic_check(card)
    assert out.overall_verdict == "block"
    discrim = next(o for o in out.outcomes
                       if o.principle_id == "no_discrimination")
    assert discrim.verdict == "violation"


def test_constitutional_concern_on_subgroup_drift_15_to_30pct():
    card = {
        "recommendation": {"action": "discharge_home"},
        "validation": {"fairness": {"max_subgroup_drift": 0.20}},
        "reasoning": {}, "abstain": [],
    }
    out = compute_constitutional_critic_check(card)
    assert out.overall_verdict in ("concern", "pass")
    discrim = next(o for o in out.outcomes
                       if o.principle_id == "no_discrimination")
    assert discrim.verdict == "concern"


def test_constitutional_blocks_on_wide_ci_without_abstain():
    card = {
        "recommendation": {"action": "discharge_home"},
        "reasoning": {"risk_estimate": {
            "probability_ci95": [0.05, 0.50],
        }},
        "abstain": [], "validation": {},
    }
    out = compute_constitutional_critic_check(card)
    abstain_check = next(o for o in out.outcomes
                              if o.principle_id == "abstain_on_uncertainty")
    assert abstain_check.verdict == "violation"


def test_constitutional_blocks_on_pricing_gatekeeping():
    card = {
        "recommendation": {"action": "premium_increase"},
        "reasoning": {}, "abstain": [], "validation": {},
    }
    out = compute_constitutional_critic_check(card)
    assert out.overall_verdict == "block"
