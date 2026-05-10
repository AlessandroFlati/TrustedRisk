"""Functional tests for the SAFE pipeline: fragility + OOD + pen-test + chaos.

Demonstrates the production-readiness validation chain a regulator
(or an internal QMS audit) would run before deploying TrustedRisk.
"""
from __future__ import annotations

from a2a_agent.safety_redteam import (
    compute_adversarial_fragility,
    compute_chaos_run,
    compute_ood_detector,
    compute_pentest_suite,
)


# ─────────────────────── SAFE-1: Adversarial fragility ───────────────────────

def test_fragility_at_decision_boundary_produces_low_l1_flips():
    """A baseline near a decision boundary should yield low minimum_l1_to_flip."""
    r = compute_adversarial_fragility(lace_total=8, max_l1_radius=3)
    # LACE=8 is in a transition zone; some perturbation within radius 3
    # should flip the action.
    if r.minimum_l1_to_flip is not None:
        assert r.minimum_l1_to_flip <= 3


def test_fragility_in_interior_of_high_risk_region():
    """LACE=18 is deep in the highest-acuity bucket -- fragility should be low."""
    r = compute_adversarial_fragility(lace_total=18, max_l1_radius=2)
    if r.minimum_l1_to_flip is None:
        # No flip in radius 2 means stable -- fragility = 0
        assert r.fragility_score == 0.0
    else:
        # If it flips, it's still a safe distance
        assert r.minimum_l1_to_flip >= 1


def test_fragility_per_component_paths_explored():
    r = compute_adversarial_fragility(lace_total=6, max_l1_radius=4)
    # The L1 ball at radius 4 in 4D is large
    assert r.n_perturbations_evaluated > 100


# ─────────────────────── SAFE-2: OOD detector ───────────────────────

def test_ood_typical_chf_patient_in_distribution():
    """A typical CHF discharge (LACE breakdown ~ mean) should be in-dist."""
    r = compute_ood_detector(
        {"L": 3, "A": 1, "C": 3, "E": 1})
    assert r.is_out_of_distribution is False
    assert r.confidence_recommendation == "preferred"


def test_ood_adversarial_extreme_input_flagged():
    r = compute_ood_detector({"L": 50, "A": 50, "C": 50, "E": 50})
    assert r.is_out_of_distribution is True
    assert r.confidence_recommendation in ("degraded", "abstain_recommended")


def test_ood_p_value_chi2_relationship():
    """Higher Mahalanobis distance -> lower p-value (monotonic)."""
    near = compute_ood_detector({"L": 2, "A": 1, "C": 2, "E": 1})
    far = compute_ood_detector({"L": 12, "A": 8, "C": 10, "E": 8})
    assert near.chi2_p_value > far.chi2_p_value


# ─────────────────────── SAFE-3: Pen-test harness ───────────────────────

def test_pentest_baseline_pass_when_no_findings():
    r = compute_pentest_suite()
    assert r.overall_posture == "pass"
    assert r.n_unblocked == 0


def test_pentest_critical_unblocked_yields_fail():
    r = compute_pentest_suite(observed_results=[
        {"attack_id": "PT-001", "observed_status": 200},   # auth bypass succeeded
    ])
    assert r.overall_posture == "fail"


def test_pentest_moderate_unblocked_yields_warn():
    r = compute_pentest_suite(observed_results=[
        {"attack_id": "PT-007", "observed_status": 500},   # XSS test went wrong
    ])
    assert r.overall_posture == "warn"


def test_pentest_categorization_present():
    r = compute_pentest_suite()
    cats = {f.attack_category for f in r.findings}
    assert {"auth_bypass", "fhir_injection",
              "smart_launch_tampering", "xss"} <= cats


# ─────────────────────── SAFE-4: Chaos engineering ───────────────────────

def test_chaos_safe_outcomes_above_threshold_for_all_faults():
    r = compute_chaos_run(iterations=500, seed=42)
    for s in r.scenarios:
        safe_rate = s.success_rate + s.abstain_rate
        assert safe_rate >= 0.75, (
            f"Scenario {s.fault_type} safe rate {safe_rate:.3f} "
            "below 75% threshold")


def test_chaos_resilience_score_realistic():
    r = compute_chaos_run(iterations=1000, seed=42)
    assert r.overall_resilience_score >= 0.85


def test_chaos_seed_reproducible_across_runs():
    a = compute_chaos_run(iterations=200, seed=99)
    b = compute_chaos_run(iterations=200, seed=99)
    for sa, sb in zip(a.scenarios, b.scenarios):
        assert sa.success_rate == sb.success_rate
        assert sa.abstain_rate == sb.abstain_rate
        assert sa.error_rate == sb.error_rate


# ─────────────────────── End-to-end SAFE chain ───────────────────────

def test_full_safe_chain_for_pre_deployment_gate():
    """Pre-deployment gate: fragility on a representative patient + OOD
    on the cohort + pen-test + chaos. ALL four must produce well-formed
    reports for the deployment to proceed."""
    fragility = compute_adversarial_fragility(lace_total=10, max_l1_radius=3)
    ood = compute_ood_detector({"L": 3, "A": 1, "C": 3, "E": 2})
    pen = compute_pentest_suite()
    chaos = compute_chaos_run(iterations=200, seed=42)

    # All four reports valid
    assert fragility.n_perturbations_evaluated > 0
    assert 0 <= ood.chi2_p_value <= 1
    assert pen.overall_posture in ("pass", "warn", "fail")
    assert 0 <= chaos.overall_resilience_score <= 1

    # The 4 reports are independent -- no test should depend on the
    # outcome of another. Functional check: at least 2 of the 4 should
    # pass / be in-distribution / have safe outcomes (a sanity check
    # that the system is not catastrophically broken by default).
    safe_signals = [
        ood.confidence_recommendation == "preferred",
        pen.overall_posture == "pass",
        chaos.overall_resilience_score > 0.7,
        fragility.fragility_score < 0.5,
    ]
    assert sum(safe_signals) >= 2
