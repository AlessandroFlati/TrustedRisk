"""Functional tests for the core risk + decision pipeline.

Chains: compute_readmission_risk -> compute_decision_utility ->
compute_fairness_audit -> compute_counterfactual_explanation. The
canonical 4-tool sequence that drives the safe-discharge agent.
"""
from __future__ import annotations

import asyncio


def test_decision_utility_chf_dominant_action_well_formed(chf_risk_estimate):
    """Decision utility takes outcome_probs (action × outcome -> mean/ci95)."""
    from mcp_server.tools.decision_utility import compute_decision_utility
    outcome_probs = {
        "home_with_care": {
            "well": {"mean": 0.66, "ci95": [0.61, 0.71]},
            "readmitted": {"mean": 0.34, "ci95": [0.29, 0.39]},
        },
        "snf": {
            "well": {"mean": 0.78, "ci95": [0.73, 0.83]},
            "readmitted": {"mean": 0.22, "ci95": [0.17, 0.27]},
        },
        "discharge_home": {
            "well": {"mean": 0.50, "ci95": [0.43, 0.57]},
            "readmitted": {"mean": 0.50, "ci95": [0.43, 0.57]},
        },
    }
    r = asyncio.run(compute_decision_utility(
        outcome_probs=outcome_probs, n_monte_carlo=300))
    assert 0.0 <= r.dominance_confidence <= 1.0
    assert r.dominant_action is not None


def test_fairness_audit_subgroup_drift(chf_risk_estimate):
    """Fairness audit should produce a SubgroupCalibration entry per
    declared demographic dimension."""
    from mcp_server.tools.fairness_audit import compute_fairness_audit
    from shared.schemas import RiskEstimate

    re = RiskEstimate.model_validate(chf_risk_estimate)
    r = asyncio.run(compute_fairness_audit(
        risk=re,
        patient_demographics={"age": 75, "race": "black",
                                  "insurance_type": "medicare"},
    ))
    assert r.n_subgroups_assessed >= 1
    assert r.confidence_action in ("no_action", "flag_for_review",
                                          "downgrade_confidence",
                                          "abstain_recommended")


def test_counterfactual_explanation_chf(chf_risk_estimate):
    """Counterfactual should produce per-factor sensitivity for each LACE
    component (the report uses `factors` per the schema)."""
    from mcp_server.tools.counterfactual_explanation import (
        compute_counterfactual_explanation,
    )
    from shared.schemas import RiskEstimate

    re = RiskEstimate.model_validate({
        **chf_risk_estimate,
        "lace_raw_score": 10,
        "probability_mean": 0.183,
        "contributing_factors": [
            {"name": "LACE_length_of_stay", "raw_value": 6,
              "lace_points": 3, "weight": 0.25},
            {"name": "LACE_acuity", "raw_value": 1,
              "lace_points": 3, "weight": 0.25},
            {"name": "LACE_comorbidity", "raw_value": 4,
              "lace_points": 2, "weight": 0.30},
            {"name": "LACE_ed_visits_6mo", "raw_value": 1,
              "lace_points": 2, "weight": 0.20},
        ],
    })
    r = asyncio.run(compute_counterfactual_explanation(risk=re))
    assert len(r.factors) == 4
    assert r.most_influential_factor in {
        "LACE_length_of_stay", "LACE_acuity",
        "LACE_comorbidity", "LACE_ed_visits_6mo",
    }


def test_grounding_supported_for_well_referenced_claim(monkeypatch):
    """ground_claim returns a supported verdict when corpus has matching
    evidence. Mock the corpus retrieval to keep the test deterministic."""
    from mcp_server.tools import ground_claim as gc_mod
    from shared.schemas import EvidenceSource

    fake = [
        EvidenceSource(source_type="guideline_passage",
                          source_id="ACC-2014-1",
                          excerpt="Heart failure GDMT...",
                          relevance_score=0.85, recency_days=None),
    ]

    monkeypatch.setattr(gc_mod, "_retrieve_corpus_evidence",
                          lambda sub, top_k=2: list(fake))
    # Also mock the FHIR retrieval to return empty
    monkeypatch.setattr(gc_mod, "_retrieve_fhir_evidence",
                          lambda bundle, sub, kind: [])
    # Mock fhir bundle fetch since SHARP context isn't bound
    async def _no_bundle(_pid):
        return None
    monkeypatch.setattr(gc_mod, "fetch_patient_bundle", _no_bundle)
    async def _resolve(explicit):
        return explicit or "pt-test"
    monkeypatch.setattr(gc_mod, "resolve_patient_id", _resolve)

    r = asyncio.run(gc_mod.ground_claim(
        claim_text="patient stable for discharge with GDMT",
        sources=["guidelines"]))
    assert r.overall_verdict in ("supported", "partially_supported")
    assert r.sub_claims


def test_phi_detection_returns_report():
    from mcp_server.tools.detect_phi import detect_phi
    r = asyncio.run(detect_phi(
        "Pt John Doe MRN 12345 DOB 1955-04-12 555-867-5309"))
    assert hasattr(r, "entities_found")
    assert hasattr(r, "risk_level")


def test_medication_reconciliation_with_explicit_lists():
    from mcp_server.tools.medication_reconciliation import (
        compute_medication_reconciliation,
    )
    r = asyncio.run(compute_medication_reconciliation(
        patient_id=None,
        admission_meds=[
            {"name": "warfarin 5 mg", "drug_class": "anticoagulant_vka",
              "status": "active"},
            {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor",
              "status": "active"},
        ],
        discharge_meds=[
            {"name": "warfarin 5 mg", "drug_class": "anticoagulant_vka",
              "status": "active"},
            {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor",
              "status": "active"},
            {"name": "spironolactone 25 mg", "drug_class": "mra",
              "status": "active"},
        ],
        monitoring_window_hours=48,
    ))
    added_names = {m.name.lower() for m in r.added}
    assert "spironolactone 25 mg" in added_names


def test_polypharmacy_concerns_aceI_mra_hyperkalemia():
    from mcp_server.tools.polypharmacy_concerns import (
        detect_polypharmacy_concerns,
    )
    r = asyncio.run(detect_polypharmacy_concerns(
        medications=[
            {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor",
              "status": "active"},
            {"name": "spironolactone 25 mg", "drug_class": "mra",
              "status": "active"},
        ],
    ))
    assert r.polypharmacy_severity in ("none", "low", "medium", "high")
    # interactions list should flag hyperkalemia
    blob = " ".join((i.detail + " " + i.mechanism).lower()
                       for i in r.interactions)
    assert "potassium" in blob or "hyperkalemia" in blob


def test_lab_trend_analysis_with_explicit_observations():
    from mcp_server.tools.lab_trend_analysis import compute_lab_trend_analysis
    r = asyncio.run(compute_lab_trend_analysis(
        observations=[
            {"name": "creatinine", "value": 1.0, "unit": "mg/dL",
              "observed_at": "2024-04-01T08:00:00Z"},
            {"name": "creatinine", "value": 1.4, "unit": "mg/dL",
              "observed_at": "2024-04-04T08:00:00Z"},
            {"name": "creatinine", "value": 1.8, "unit": "mg/dL",
              "observed_at": "2024-04-07T08:00:00Z"},
        ],
        min_observations_per_lab=2,
    ))
    assert len(r.trends) >= 1
    cr = next((t for t in r.trends if "creatinine" in t.lab_name.lower()),
                 None)
    assert cr is not None
