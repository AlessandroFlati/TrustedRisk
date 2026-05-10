"""Unit tests for AUDIT-1/2/3/4 audit + privacy modules."""
from __future__ import annotations

import pytest

from a2a_agent.merkle_audit import (
    build_inclusion_proof,
    compute_merkle_audit_root,
    verify_inclusion,
)
from a2a_agent.dp_equity import compute_dp_equity_dashboard
from a2a_agent.right_to_explanation import compute_right_to_explanation
from a2a_agent.patient_audit_summary import compute_patient_audit_summary
from shared.schemas import EquityDashboard, EquitySegment, MerkleProofStep


# ───────────────────────────────────────────────────────────────────
# AUDIT-1 -- Merkle audit chain
# ───────────────────────────────────────────────────────────────────

def _events(n: int = 4) -> list[dict]:
    return [{"event_id": f"e-{i}", "tool": "x", "outcome_hash": f"h-{i}"}
            for i in range(n)]


def test_merkle_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a list"):
        compute_merkle_audit_root("nope")  # type: ignore[arg-type]


def test_merkle_empty_list_yields_canonical_root():
    r = compute_merkle_audit_root([])
    assert r.n_events == 0
    assert r.merkle_root  # canonical empty root


def test_merkle_single_event_root_equals_leaf():
    r = compute_merkle_audit_root(_events(1))
    assert r.n_events == 1
    assert r.merkle_root == r.leaf_hashes[0]


def test_merkle_root_deterministic_across_calls():
    a = compute_merkle_audit_root(_events(8))
    b = compute_merkle_audit_root(_events(8))
    assert a.merkle_root == b.merkle_root


def test_merkle_root_changes_on_event_modification():
    base = compute_merkle_audit_root(_events(8))
    events = _events(8)
    events[3]["outcome_hash"] = "TAMPERED"
    tampered = compute_merkle_audit_root(events)
    assert tampered.merkle_root != base.merkle_root


def test_inclusion_proof_verifies():
    events = _events(8)
    proof = build_inclusion_proof(events, target_event_id="e-3")
    assert proof.verified is True
    # External verifier rebuilds the root
    assert verify_inclusion(proof.leaf_hash, proof.proof_steps,
                                  proof.merkle_root) is True


def test_inclusion_proof_for_unknown_event_raises():
    with pytest.raises(ValueError, match="not found"):
        build_inclusion_proof(_events(4), target_event_id="not-here")


def test_inclusion_proof_fails_on_tampered_leaf():
    """Verifying with a different leaf must fail."""
    events = _events(8)
    proof = build_inclusion_proof(events, target_event_id="e-3")
    # Use a wrong leaf hash
    bad = "0" * 64
    assert verify_inclusion(bad, proof.proof_steps, proof.merkle_root) is False


def test_inclusion_proof_handles_odd_leaves():
    """Odd-leaf trees use the duplicate-last-leaf convention."""
    events = _events(7)   # odd
    proof = build_inclusion_proof(events, target_event_id="e-6")
    assert proof.verified is True


def test_height_grows_with_event_count():
    h2 = compute_merkle_audit_root(_events(2)).height
    h8 = compute_merkle_audit_root(_events(8)).height
    h32 = compute_merkle_audit_root(_events(32)).height
    assert h2 < h8 < h32


# ───────────────────────────────────────────────────────────────────
# AUDIT-2 -- DP equity dashboard
# ───────────────────────────────────────────────────────────────────

def _equity_dashboard(seg_counts: list[int]) -> EquityDashboard:
    """Build a synthetic EquityDashboard with `seg_counts` sized segments."""
    segments = []
    for i, n in enumerate(seg_counts):
        segments.append(EquitySegment(
            subgroup_dimension="race",
            subgroup_value=f"group-{i}",
            n_decisions=n,
            action_counts={"home_with_care": n // 2,
                              "discharge_home": n // 2},
            n_abstained=max(0, n // 10),
            avg_risk=0.20,
            intervention_rate=0.5,
            abstention_rate=0.1,
            confidence_high_rate=0.7,
        ))
    return EquityDashboard(
        n_total_decisions=sum(seg_counts),
        segments=segments,
        max_intervention_rate_disparity=0.0,
        max_abstention_rate_disparity=0.0,
        max_avg_risk_disparity=0.0,
        rationale="synthetic test fixture",
    )


def test_dp_invalid_epsilon_raises():
    d = _equity_dashboard([10, 10])
    with pytest.raises(ValueError, match="epsilon"):
        compute_dp_equity_dashboard(d, epsilon=0)


def test_dp_invalid_threshold_raises():
    d = _equity_dashboard([10])
    with pytest.raises(ValueError, match="suppression_threshold"):
        compute_dp_equity_dashboard(d, suppression_threshold=-1)


def test_dp_suppresses_small_segments():
    d = _equity_dashboard([2, 3, 50])  # first two below threshold
    r = compute_dp_equity_dashboard(d, suppression_threshold=5, seed=0)
    assert len(r.suppressed_segments) == 2
    assert len(r.segments) == 1


def test_dp_noised_counts_close_to_truth_with_high_epsilon():
    """ε very large -> noise -> 0, noised ≈ raw."""
    d = _equity_dashboard([100, 200])
    r = compute_dp_equity_dashboard(d, epsilon=100.0, seed=42)
    for seg, raw in zip(r.segments, d.segments):
        assert abs(seg["n_decisions_noised"] - raw.n_decisions) <= 5


def test_dp_seed_makes_results_reproducible():
    d = _equity_dashboard([50, 50])
    a = compute_dp_equity_dashboard(d, epsilon=1.0, seed=7)
    b = compute_dp_equity_dashboard(d, epsilon=1.0, seed=7)
    assert a.segments == b.segments
    assert a.n_total_decisions_noised == b.n_total_decisions_noised


def test_dp_dict_input_coerced():
    d = _equity_dashboard([20])
    r = compute_dp_equity_dashboard(d.model_dump(), epsilon=1.0, seed=0,
                                            suppression_threshold=5)
    assert r.epsilon == 1.0


def test_dp_references_include_dwork():
    r = compute_dp_equity_dashboard(_equity_dashboard([20]),
                                            epsilon=1.0, seed=0,
                                            suppression_threshold=5)
    assert any("Dwork" in ref for ref in r.references)


# ───────────────────────────────────────────────────────────────────
# AUDIT-3 -- Right-to-explanation
# ───────────────────────────────────────────────────────────────────

def _decision_card(*, action="home_with_care", confidence="medium",
                       prob=0.30) -> dict:
    return {
        "recommendation": {"action": action, "confidence": confidence},
        "reasoning": {"risk_estimate": {
            "model_name": "lace-plus-bayesian-v1",
            "probability_mean": prob,
            "lace_raw_score": 10,
            "contributing_factors": [
                {"name": "LACE_length_of_stay", "raw_value": 5,
                 "lace_points": 2, "weight": 0.30},
                {"name": "LACE_acuity", "raw_value": 1,
                 "lace_points": 3, "weight": 0.25},
                {"name": "LACE_comorbidity", "raw_value": 4,
                 "lace_points": 3, "weight": 0.30},
                {"name": "LACE_ed_visits_6mo", "raw_value": 1,
                 "lace_points": 1, "weight": 0.15},
            ],
            "fhir_observations_used": ["obs-1", "obs-2", "obs-3"],
        }},
        "validation": {
            "grounding": {"sub_claims": [{
                "evidence_sources": [{"source_id": "ACC-2014-1",
                                          "source_type": "guideline_passage",
                                          "excerpt": "Heart failure patients..."}]
            }]},
            "phi_check": {"risk_level": "none"},
        },
        "audit": {"request_id": "req-001"},
        "abstain": [],
    }


@pytest.fixture(autouse=True)
def _disable_llm_for_rte(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


def test_rte_invalid_input_raises():
    with pytest.raises(ValueError, match="DecisionCard"):
        compute_right_to_explanation("nope")  # type: ignore[arg-type]


def test_rte_extracts_action_and_confidence():
    r = compute_right_to_explanation(_decision_card())
    assert r.decision_action == "home_with_care"
    assert r.decision_confidence == "medium"
    assert r.request_id == "req-001"


def test_rte_factors_listed_with_weights():
    r = compute_right_to_explanation(_decision_card())
    assert len(r.factors_considered) == 4
    assert "LACE_length_of_stay" in r.factors_considered
    assert r.factors_weight["LACE_length_of_stay"] == 0.30


def test_rte_data_sources_include_model():
    r = compute_right_to_explanation(_decision_card())
    sources_blob = " ".join(r.data_sources).lower()
    assert "lace-plus-bayesian" in sources_blob


def test_rte_deterministic_template_when_llm_disabled():
    r = compute_right_to_explanation(_decision_card())
    assert r.method == "deterministic_template"
    assert "GDPR" in r.plain_language_rationale


def test_rte_legal_basis_present():
    r = compute_right_to_explanation(_decision_card())
    assert "GDPR" in r.legal_basis
    assert "AI Act" in r.legal_basis


def test_rte_abstain_triggers_surfaced():
    card = _decision_card()
    card["abstain"] = [{"type": "out_of_distribution",
                            "detail": "LACE bucket empty"}]
    r = compute_right_to_explanation(card)
    blob = r.plain_language_rationale.lower()
    assert "abstain" in blob or "human" in blob or "clinician" in blob


def test_rte_blocks_dose_change_phrases_from_llm(monkeypatch):
    """If the LLM tries to inject 'take more', fall back to template."""
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    from a2a_agent import right_to_explanation as mod
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: "You should take more warfarin tonight.")
    r = compute_right_to_explanation(_decision_card())
    assert r.method == "deterministic_template"
    assert any("dose_change_phrase_blocked" in w
                  for w in r.safety_warnings)


# ───────────────────────────────────────────────────────────────────
# AUDIT-4 -- Patient-facing audit summary
# ───────────────────────────────────────────────────────────────────

def test_patient_audit_invalid_input_raises():
    with pytest.raises(ValueError, match="DecisionCard"):
        compute_patient_audit_summary(42)  # type: ignore[arg-type]


def test_patient_audit_features_table_carries_explanations():
    r = compute_patient_audit_summary(_decision_card())
    assert len(r.features_considered) == 4
    blob = " ".join(f["explanation"] for f in r.features_considered).lower()
    assert "hospital" in blob


def test_patient_audit_citations_dedup_and_listed():
    r = compute_patient_audit_summary(_decision_card())
    assert len(r.citations_used) == 1
    assert r.citations_used[0]["source_id"] == "ACC-2014-1"


def test_patient_audit_headline_for_each_action():
    for action, expected in [
        ("discharge_home", "go home"),
        ("home_with_care", "extra support"),
        ("snf", "skilled nursing"),
        ("continued_admission", "stay in the hospital"),
    ]:
        r = compute_patient_audit_summary(_decision_card(action=action))
        assert expected in r.headline_question.lower()


def test_patient_audit_no_recommendation_explained():
    card = _decision_card()
    card["recommendation"] = None
    r = compute_patient_audit_summary(card)
    assert "no automatic" in r.headline_question.lower() or \
           "considered" in r.headline_question.lower()


def test_patient_audit_request_id_propagated():
    r = compute_patient_audit_summary(_decision_card())
    assert r.request_id == "req-001"


def test_patient_audit_no_phi_in_features():
    """Feature names + values must NOT contain PHI tokens (the report is
    PHI-safe by construction)."""
    r = compute_patient_audit_summary(_decision_card())
    blob = repr(r.features_considered).lower()
    for forbidden in ("ssn", "mrn", "birthdate", "email"):
        assert forbidden not in blob


def test_patient_audit_abstain_triggers_listed():
    card = _decision_card()
    card["abstain"] = [{"type": "evidence_insufficient", "detail": "x"}]
    r = compute_patient_audit_summary(card)
    assert "evidence_insufficient" in r.abstain_triggers_present
