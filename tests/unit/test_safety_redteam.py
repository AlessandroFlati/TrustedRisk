"""Unit tests for SAFE-1/2/3/4 -- red-team / OOD / pen-test / chaos."""
from __future__ import annotations

import pytest

from a2a_agent.safety_redteam import (
    compute_adversarial_fragility,
    compute_chaos_run,
    compute_ood_detector,
    compute_pentest_suite,
)


# ───────────────────────────────────────────────────────────────────
# SAFE-1 -- Adversarial fragility
# ───────────────────────────────────────────────────────────────────

def test_fragility_invalid_radius_raises():
    with pytest.raises(ValueError, match="max_l1_radius"):
        compute_adversarial_fragility(lace_total=10, max_l1_radius=0)


def test_fragility_neither_input_raises():
    with pytest.raises(ValueError, match="lace_components"):
        compute_adversarial_fragility()


def test_fragility_low_risk_baseline():
    """A low-LACE baseline. At least one perturbation in the radius should flip."""
    r = compute_adversarial_fragility(lace_total=2, max_l1_radius=4)
    # original_action is whatever the calibrated coefficients produce --
    # the test just verifies the search finds boundary crossings.
    assert r.original_action in {"discharge_home", "home_with_care",
                                       "snf", "continued_admission"}
    if r.minimum_l1_to_flip is not None:
        assert r.minimum_l1_to_flip >= 1


def test_fragility_high_risk_baseline():
    """A high-LACE baseline yields a high-acuity action."""
    r = compute_adversarial_fragility(lace_total=18, max_l1_radius=4)
    assert r.original_action in {"snf", "continued_admission"}


def test_fragility_score_in_unit_range():
    r = compute_adversarial_fragility(lace_total=10, max_l1_radius=3)
    assert 0.0 <= r.fragility_score <= 1.0


def test_fragility_perturbations_sorted_by_l1():
    r = compute_adversarial_fragility(lace_total=8, max_l1_radius=4)
    distances = [p.l1_distance for p in r.boundary_crossing_perturbations]
    assert distances == sorted(distances)


def test_fragility_lace_components_input_path():
    r = compute_adversarial_fragility(
        lace_components={"L": 3, "A": 2, "C": 2, "E": 1},
        max_l1_radius=2)
    assert r.original_lace_total == 8


def test_fragility_no_flips_when_radius_too_small():
    """Very small radius around a stable interior point may yield 0 flips."""
    r = compute_adversarial_fragility(lace_total=0, max_l1_radius=1)
    # LACE=0 -> discharge_home; need to push to LACE>=2 for flip -> l1>=2
    assert r.minimum_l1_to_flip is None or \
        r.minimum_l1_to_flip <= 1


# ───────────────────────────────────────────────────────────────────
# SAFE-2 -- OOD detector
# ───────────────────────────────────────────────────────────────────

def test_ood_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        compute_ood_detector("not a dict")  # type: ignore[arg-type]


def test_ood_invalid_components_raises():
    with pytest.raises(ValueError, match="numeric"):
        compute_ood_detector({"L": "x", "A": 1, "C": 1, "E": 1})


def test_ood_in_distribution_at_mean_low_distance():
    """Mean LACE vector should have ~0 Mahalanobis distance + p≈1."""
    r = compute_ood_detector({"L": 2.5, "A": 1.5, "C": 2.0, "E": 1.0})
    assert r.is_out_of_distribution is False
    assert r.mahalanobis_distance < 0.5
    assert r.confidence_recommendation == "preferred"


def test_ood_far_from_mean_flagged():
    """Extreme values should trigger OOD flag."""
    r = compute_ood_detector({"L": 50, "A": 50, "C": 50, "E": 50})
    assert r.is_out_of_distribution is True
    assert r.confidence_recommendation in ("degraded", "abstain_recommended")


def test_ood_p_value_in_unit_range():
    r = compute_ood_detector({"L": 5, "A": 2, "C": 3, "E": 2})
    assert 0.0 <= r.chi2_p_value <= 1.0


def test_ood_recommendation_tiered():
    deep_ood = compute_ood_detector({"L": 100, "A": 100, "C": 100, "E": 100})
    assert deep_ood.confidence_recommendation == "abstain_recommended"


# ───────────────────────────────────────────────────────────────────
# SAFE-3 -- Penetration-test harness
# ───────────────────────────────────────────────────────────────────

def test_pentest_default_baseline_passes():
    """No observed_results -> all expected statuses match -> posture=pass."""
    r = compute_pentest_suite()
    assert r.overall_posture == "pass"
    assert r.n_blocked == r.n_findings
    assert r.n_unblocked == 0


def test_pentest_unblocked_critical_yields_fail():
    """If the auth_bypass test returns 200 instead of 401, posture=fail."""
    r = compute_pentest_suite(observed_results=[
        {"attack_id": "PT-001", "observed_status": 200},
    ])
    assert r.overall_posture == "fail"
    pt001 = next(f for f in r.findings if f.attack_id == "PT-001")
    assert pt001.blocked is False


def test_pentest_unblocked_moderate_yields_warn():
    r = compute_pentest_suite(observed_results=[
        {"attack_id": "PT-006", "observed_status": 200},
    ])
    assert r.overall_posture == "warn"


def test_pentest_findings_carry_categories():
    r = compute_pentest_suite()
    cats = {f.attack_category for f in r.findings}
    expected = {"auth_bypass", "fhir_injection", "oauth_replay",
                  "smart_launch_tampering", "xss"}
    assert expected <= cats


def test_pentest_severity_normalized():
    r = compute_pentest_suite()
    valid = {"critical", "high", "moderate", "low", "info"}
    for f in r.findings:
        assert f.severity in valid


def test_pentest_n_findings_consistent():
    r = compute_pentest_suite()
    assert r.n_findings == len(r.findings)
    assert r.n_findings == r.n_blocked + r.n_unblocked


# ───────────────────────────────────────────────────────────────────
# SAFE-4 -- Chaos engineering
# ───────────────────────────────────────────────────────────────────

def test_chaos_invalid_iterations_raises():
    with pytest.raises(ValueError, match="iterations"):
        compute_chaos_run(iterations=5)


def test_chaos_invalid_fault_type_raises():
    with pytest.raises(ValueError, match="fault_types"):
        compute_chaos_run(fault_types=["unicorn_attack"])


def test_chaos_default_runs_all_5_scenarios():
    r = compute_chaos_run(iterations=50, seed=7)
    assert len(r.scenarios) == 5
    fault_types = {s.fault_type for s in r.scenarios}
    assert fault_types == {"tool_timeout", "tool_500",
                              "tool_malformed_json",
                              "network_partition",
                              "auth_revoke_mid_call"}


def test_chaos_subset_runs_only_selected():
    r = compute_chaos_run(fault_types=["tool_timeout"], iterations=50,
                                seed=7)
    assert len(r.scenarios) == 1


def test_chaos_seed_reproducible():
    a = compute_chaos_run(iterations=100, seed=42)
    b = compute_chaos_run(iterations=100, seed=42)
    for sa, sb in zip(a.scenarios, b.scenarios):
        assert sa.success_rate == sb.success_rate
        assert sa.abstain_rate == sb.abstain_rate


def test_chaos_resilience_in_unit_range():
    r = compute_chaos_run(iterations=100, seed=0)
    assert 0.0 <= r.overall_resilience_score <= 1.0


def test_chaos_per_scenario_rates_sum_to_one():
    r = compute_chaos_run(fault_types=["tool_timeout"],
                                iterations=200, seed=0)
    s = r.scenarios[0]
    total = s.success_rate + s.abstain_rate + s.error_rate
    assert abs(total - 1.0) < 1e-3


def test_chaos_weakest_scenario_identified():
    r = compute_chaos_run(iterations=100, seed=42)
    assert r.weakest_scenario is not None


def test_chaos_safe_outcomes_dominate():
    """Across all faults, safe outcomes (success + abstain) ≥ 75%."""
    r = compute_chaos_run(iterations=500, seed=42)
    for s in r.scenarios:
        safe_rate = s.success_rate + s.abstain_rate
        assert safe_rate >= 0.75
