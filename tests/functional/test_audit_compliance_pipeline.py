"""Functional tests for the AUDIT compliance trail.

Chains: DecisionCard -> Merkle audit chain -> DP equity dashboard ->
right-to-explanation -> patient-facing audit summary. Demonstrates the
full forensic / privacy / regulatory disclosure stack.
"""
from __future__ import annotations

import copy

from a2a_agent.dp_equity import compute_dp_equity_dashboard
from a2a_agent.equity_dashboard import compute_equity_dashboard
from a2a_agent.merkle_audit import (
    build_inclusion_proof,
    compute_merkle_audit_root,
    verify_inclusion,
)
from a2a_agent.patient_audit_summary import compute_patient_audit_summary
from a2a_agent.right_to_explanation import compute_right_to_explanation


def _audit_event(event_id: str, action: str, prob: float) -> dict:
    return {
        "event_id": event_id,
        "tool": "compute_readmission_risk",
        "outcome_hash": f"sha256:{prob:.3f}",
        "action": action,
        "timestamp": "2026-04-29T00:00:00+00:00",
    }


# ─────────────────────── AUDIT-1: Merkle chain ───────────────────────

def test_merkle_chain_over_realistic_audit_trail():
    events = [
        _audit_event("evt-001", "home_with_care", 0.30),
        _audit_event("evt-002", "discharge_home", 0.10),
        _audit_event("evt-003", "snf", 0.55),
        _audit_event("evt-004", "continued_admission", 0.70),
        _audit_event("evt-005", "home_with_care", 0.25),
    ]
    r = compute_merkle_audit_root(events)
    assert r.n_events == 5
    assert len(r.merkle_root) == 64

    # External verifier: build inclusion proof + verify
    proof = build_inclusion_proof(events, target_event_id="evt-003")
    assert proof.verified is True
    assert verify_inclusion(proof.leaf_hash, proof.proof_steps,
                                  proof.merkle_root) is True


def test_merkle_chain_detects_tamper():
    events = [_audit_event(f"e{i}", "home_with_care", 0.20)
                for i in range(10)]
    base_root = compute_merkle_audit_root(events).merkle_root

    # Tamper: change one event's outcome hash
    events_tampered = copy.deepcopy(events)
    events_tampered[5]["outcome_hash"] = "sha256:tampered"
    new_root = compute_merkle_audit_root(events_tampered).merkle_root
    assert new_root != base_root


def test_merkle_inclusion_proof_for_each_event():
    events = [_audit_event(f"e{i}", "home_with_care", 0.20)
                for i in range(7)]
    root_obj = compute_merkle_audit_root(events)
    for ev in events:
        proof = build_inclusion_proof(events, target_event_id=ev["event_id"])
        assert proof.merkle_root == root_obj.merkle_root
        assert verify_inclusion(proof.leaf_hash, proof.proof_steps,
                                      proof.merkle_root) is True


# ─────────────────────── AUDIT-2: Differential privacy ───────────────────────

def _equity_dashboard_for_dp():
    """Build a real EquityDashboard via compute_equity_dashboard so DP
    has a structured input."""
    cohort = []
    for i in range(20):
        cohort.append({
            "decision_card": {
                "recommendation": {"action": "home_with_care",
                                       "confidence": "high"},
                "reasoning": {"risk_estimate": {"probability_mean": 0.30}},
                "audit": {}, "validation": {}, "abstain": [],
            },
            "demographics": {"age": 70, "race": "white" if i % 2
                                else "black", "sex": "female",
                                "insurance_type": "medicare"},
        })
    return compute_equity_dashboard(cohort)


def test_dp_equity_suppresses_small_segments():
    """Build a dashboard with a small subgroup and confirm DP suppresses it."""
    cohort = (
        [{"decision_card": {"recommendation":
                                  {"action": "home_with_care",
                                   "confidence": "high"},
                                "reasoning": {"risk_estimate":
                                                  {"probability_mean": 0.20}},
                                "audit": {}, "validation": {}, "abstain": []},
            "demographics": {"age": 70, "race": "white", "sex": "female",
                                "insurance_type": "medicare"}}
         for _ in range(50)]
        + [{"decision_card": {"recommendation":
                                  {"action": "home_with_care",
                                   "confidence": "high"},
                                "reasoning": {"risk_estimate":
                                                  {"probability_mean": 0.30}},
                                "audit": {}, "validation": {}, "abstain": []},
            "demographics": {"age": 70, "race": "asian", "sex": "female",
                                "insurance_type": "medicare"}}
            for _ in range(2)]    # tiny subgroup -> must be suppressed
    )
    dash = compute_equity_dashboard(cohort)
    dp = compute_dp_equity_dashboard(dash, epsilon=1.0,
                                            suppression_threshold=5, seed=42)
    suppressed_blob = " ".join(dp.suppressed_segments)
    assert "asian" in suppressed_blob


def test_dp_equity_high_epsilon_low_distortion():
    dash = _equity_dashboard_for_dp()
    dp = compute_dp_equity_dashboard(dash, epsilon=100.0,
                                            suppression_threshold=5, seed=0)
    # ε very large -> noise -> 0; n_total close to raw (within tolerance)
    raw_total = dash.n_total_decisions
    assert abs(dp.n_total_decisions_noised - raw_total) <= 2


def test_dp_equity_seed_reproducibility():
    dash = _equity_dashboard_for_dp()
    a = compute_dp_equity_dashboard(dash, epsilon=1.0, seed=99,
                                            suppression_threshold=5)
    b = compute_dp_equity_dashboard(dash, epsilon=1.0, seed=99,
                                            suppression_threshold=5)
    assert a.n_total_decisions_noised == b.n_total_decisions_noised
    assert a.segments == b.segments


# ─────────────────────── AUDIT-3: Right-to-explanation ───────────────────────

def test_rte_full_decision_card_yields_template_rationale(chf_decision_card):
    r = compute_right_to_explanation(chf_decision_card)
    assert r.method == "deterministic_template"
    assert r.decision_action == "home_with_care"
    assert "GDPR" in r.legal_basis
    assert "AI Act" in r.legal_basis


def test_rte_factor_table_carries_weights(chf_decision_card):
    r = compute_right_to_explanation(chf_decision_card)
    assert len(r.factors_considered) == 4
    assert "LACE_length_of_stay" in r.factors_weight
    assert "LACE_comorbidity" in r.factors_weight


def test_rte_data_sources_list(chf_decision_card):
    r = compute_right_to_explanation(chf_decision_card)
    sources_blob = " ".join(r.data_sources).lower()
    assert "lace" in sources_blob


# ─────────────────────── AUDIT-4: Patient-facing summary ───────────────────────

def test_patient_audit_summary_phi_safe(chf_decision_card):
    r = compute_patient_audit_summary(chf_decision_card)
    # No PHI tokens leak through factor names / explanations
    blob = repr(r.features_considered).lower()
    for forbidden in ("ssn", "mrn", "birthdate", "email", "phone"):
        assert forbidden not in blob


def test_patient_audit_summary_routes_per_action(chf_decision_card):
    """home_with_care -> 'extra support' headline."""
    r = compute_patient_audit_summary(chf_decision_card)
    assert "support" in r.headline_question.lower()


def test_patient_audit_citations_dedup(chf_decision_card):
    r = compute_patient_audit_summary(chf_decision_card)
    assert len(r.citations_used) == len(
        {c["source_id"] for c in r.citations_used})


# ─────────────────────── End-to-end AUDIT chain ───────────────────────

def test_full_audit_chain_card_to_patient_summary(chf_decision_card):
    """One DecisionCard flows through all 4 AUDIT surfaces:
       1. compute_right_to_explanation  -> narrative rationale
       2. compute_patient_audit_summary -> structured features list
       3. compute_merkle_audit_root over the audit log
       4. compute_dp_equity_dashboard over a population sample
    """
    rte = compute_right_to_explanation(chf_decision_card)
    summary = compute_patient_audit_summary(chf_decision_card)

    events = [{"event_id": f"e-{i}", "outcome_hash": f"h-{i}"}
                for i in range(10)]
    chain = compute_merkle_audit_root(events)

    cohort = [{"decision_card": chf_decision_card,
                  "demographics": {"age": 75, "race": "black", "sex": "female",
                                      "insurance_type": "medicare"}}
                for _ in range(15)]
    dash = compute_equity_dashboard(cohort)
    dp = compute_dp_equity_dashboard(dash, epsilon=1.0,
                                            suppression_threshold=5, seed=42)

    # All four surfaces produced, well-formed
    assert rte.plain_language_rationale
    assert summary.headline_question
    assert chain.merkle_root
    assert dp.n_total_decisions_noised >= 0
