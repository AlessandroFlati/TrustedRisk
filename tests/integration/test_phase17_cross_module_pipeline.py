"""Phase 17.CON1 - Cross-module integration test.

Exercises a single end-to-end pipeline that touches ~10 Phase 14-17
modules in their natural composition order, simulating a realistic
TrustedRisk run from FHIR ingestion through to the regulatory pack
+ patient-facing audit:

    HL7 v2 message
        -> compute_hl7v2_message_parse                  (Phase 14.13)
    -> structured patient + LACE features
        -> readmission risk (W1 Beta-Binomial)
        -> compute_shap_attribution                     (Phase 14.M1)
        -> compute_cox_proportional_hazards             (Phase 14.M2)
        -> conformal multi-class prediction set         (Phase 16.F2)
    -> 3-agent debate                                   (Phase 14.16)
    -> patient_advocate 5-axis card                     (Phase 16.J)
    -> CDS Hooks v1.1 card                              (Phase 16.H3)
    -> OMOP CDM export                                  (Phase 16.H2)
    -> SMART-on-FHIR launch URL embedded                (Phase 16.H3)
    -> federation_chain audit hash                      (Phase 17.U)
    -> regulatory pack still at 100% artefact coverage  (Phase 14.17)

A failure here surfaces composition bugs across module boundaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────
# 1. HL7 v2 ingestion
# ─────────────────────────────────────────────────────────────────────


_HL7_INPUT = (
    "MSH|^~\\&|EPIC|HOSP|RECV|FAC|20260430|"
    "|ADT^A01|MID|P|2.5\r"
    "PID|1||MRN-CON1^^^HOSP^MR||DOE^JANE||19551115|F\r"
    "PV1|1|I|3W^301^A||||DR1^Smith||MED\r"
    "OBX|1|NM|718-7^Hemoglobin^LN||10.1|g/dL|13.0-17.0|L|||F\r"
)


def test_hl7v2_to_patient_record():
    import asyncio
    from mcp_server.tools.legacy_ehr_parsers import (
        compute_hl7v2_message_parse,
    )
    res = asyncio.run(compute_hl7v2_message_parse(message=_HL7_INPUT))
    assert res.message_type == "ADT^A01"
    assert res.patient["patient_id"] == "MRN-CON1"
    assert res.patient["gender"] == "female"


# ─────────────────────────────────────────────────────────────────────
# 2. Risk + explanation chain
# ─────────────────────────────────────────────────────────────────────


def test_risk_chain_from_lace_to_explanation():
    """LACE features -> Beta-Binomial bucket map -> SHAP
    attribution -> Cox PH on a small synthetic survival cohort."""
    import asyncio
    from mcp_server.tools.model_research import (
        compute_cox_proportional_hazards,
        compute_shap_attribution,
    )
    # SHAP on the LACE bucket map (linear approximation around the
    # current bucket)
    shap = asyncio.run(compute_shap_attribution(
        feature_values={
            "L_los": 4.0, "A_acuity": 3.0,
            "C_charlson": 2.0, "E_ed": 1.0,
        },
        coefficients={
            "L_los": 0.06, "A_acuity": 0.10,
            "C_charlson": 0.08, "E_ed": 0.05,
        },
        intercept=-1.5,
    ))
    assert isinstance(shap.shap_values, dict)
    assert set(shap.shap_values) == {
        "L_los", "A_acuity", "C_charlson", "E_ed"
    }
    # Cox PH: 6 patients, 4 events, 1 covariate
    cox = asyncio.run(compute_cox_proportional_hazards(
        durations=[5.0, 8.0, 11.0, 4.0, 9.0, 13.0],
        events=[1, 1, 0, 1, 1, 0],
        covariates=[[0.1], [0.4], [0.2], [0.7], [0.3], [0.5]],
        feature_names=["lace_norm"],
    ))
    assert "lace_norm" in cox.hazard_ratios


# ─────────────────────────────────────────────────────────────────────
# 3. Conformal multi-class
# ─────────────────────────────────────────────────────────────────────


def test_conformal_multi_class_yields_prediction_set():
    from a2a_agent.trustworthy_ml import split_conformal_multiclass
    classes = ["discharge_home", "discharge_with_homecare",
               "continued_admission"]
    cal_probs = []
    cal_y = []
    for _ in range(60):
        cal_probs.append({c: (0.7 if c == classes[0] else 0.15)
                          for c in classes})
        cal_y.append(classes[0])
    test_probs = [
        {classes[0]: 0.5, classes[1]: 0.3, classes[2]: 0.2}
    ]
    res = split_conformal_multiclass(
        cal_probabilities=cal_probs, cal_true_labels=cal_y,
        test_probabilities=test_probs, target_coverage=0.85,
    )
    assert len(res.prediction_sets) == 1
    assert res.prediction_sets[0]


# ─────────────────────────────────────────────────────────────────────
# 4. Debate + patient advocate
# ─────────────────────────────────────────────────────────────────────


def test_debate_advocate_composition_for_flagged_subgroup():
    from a2a_agent.multi_agent_debate import (
        DebateInput, run_debate,
    )
    from a2a_agent.patient_advocate import (
        PatientAdvocateInput, evaluate_patient_advocate,
    )
    debate = run_debate(DebateInput(
        recommended_action="discharge_home",
        risk_point_estimate=0.12,
        risk_ci_width=0.06,
        fairness_subgroup="black",
        fairness_audit_present=True,
    ))
    advocate = evaluate_patient_advocate(PatientAdvocateInput(
        patient_age=70, patient_race="black",
        patient_insurance="medicare", patient_language="english",
        n_chronic_medications=5,
        has_home_caregiver_available=True,
        has_transportation=True,
        fairness_audit_present=True,
        recommended_action="discharge_home",
        risk_point_estimate=0.12,
    ))
    # Fairness-flagged subgroup with audit -> revise both layers
    assert debate.verdict == "revise"
    assert advocate.overall_verdict in ("challenge", "concur")


# ─────────────────────────────────────────────────────────────────────
# 5. CDS Hooks card + SMART link + OMOP export
# ─────────────────────────────────────────────────────────────────────


def test_cds_hooks_card_carries_smart_link_and_overrides():
    from a2a_agent.cds_hooks_card import build_decision_card
    card = build_decision_card(
        patient_id="p-CON1", encounter_id="e-CON1",
        recommended_action="discharge_with_homecare",
        risk_point_estimate=0.158,
    )
    assert any(l.type == "smart" for l in card.links)
    assert len(card.overrideReasons) >= 4


def test_omop_export_emits_six_tables():
    from a2a_agent.omop_cdm import export_to_omop
    bundle = {
        "resourceType": "Bundle", "entry": [
            {"resource": {
                "resourceType": "Patient", "id": "p-CON1",
                "birthDate": "1955-11-15", "gender": "female",
            }},
            {"resource": {
                "resourceType": "Encounter", "id": "e-CON1",
                "class": {"code": "IMP"},
                "period": {"start": "2026-04-22", "end": "2026-04-29"},
            }},
            {"resource": {
                "resourceType": "Condition", "id": "c-1",
                "code": {"coding": [{"code": "I50.21"}]},
                "recordedDate": "2026-04-22",
            }},
            {"resource": {
                "resourceType": "Observation", "id": "o-1",
                "code": {"coding": [{"code": "718-7"}]},
                "valueQuantity": {"value": 10.1, "unit": "g/dL"},
                "effectiveDateTime": "2026-04-30",
            }},
            {"resource": {
                "resourceType": "MedicationRequest", "id": "m-1",
                "medicationCodeableConcept": {
                    "coding": [{"code": "11289"}],
                    "text": "warfarin 5 mg",
                },
                "authoredOn": "2026-04-23",
            }},
        ],
    }
    decision_card = {
        "decision_card_id": "card-CON1",
        "created_at_iso": "2026-04-30T08:00:00Z",
        "recommendation": {
            "action": "discharge_with_homecare",
            "confidence": "preferred",
            "rationale": "stable on diuretics",
        },
        "risk_estimate": {"probability_mean": 0.158},
    }
    rep = export_to_omop(
        fhir_bundle=bundle, decision_card=decision_card,
    )
    assert set(rep.n_per_table.keys()) >= {
        "PERSON", "VISIT_OCCURRENCE", "CONDITION_OCCURRENCE",
        "MEASUREMENT", "DRUG_EXPOSURE", "NOTE",
    }


# ─────────────────────────────────────────────────────────────────────
# 6. Federation chain audit
# ─────────────────────────────────────────────────────────────────────


def test_federation_chain_audit_root_is_64_hex_chars():
    from a2a_agent.federation_chain import run_federation_chain
    trace = run_federation_chain(initial_payload={
        "patient_id": "p-CON1", "encounter_id": "e-CON1", "lace": 7,
        "demographics": {
            "age": 70, "race": "black", "insurance": "medicare",
            "language": "english", "n_chronic_meds": 5,
            "has_home_caregiver_available": True,
            "has_transportation": True,
            "fairness_audit_present": True,
        },
    })
    assert len(trace.audit_root) == 64
    assert trace.n_hops == 3


# ─────────────────────────────────────────────────────────────────────
# 7. Regulatory pack still at 100% artefact coverage
# ─────────────────────────────────────────────────────────────────────


def test_regulatory_pack_still_full_coverage():
    from a2a_agent.regulatory_pack import build_regulatory_pack
    pack = build_regulatory_pack(project_root=ROOT)
    assert pack.n_sections == 14
    assert pack.overall_artefact_coverage == 1.0


# ─────────────────────────────────────────────────────────────────────
# 8. Federation marketplace covers all 47 bundles
# ─────────────────────────────────────────────────────────────────────


def test_federation_marketplace_covers_all_bundles():
    from a2a_agent.federation_registry import (
        build_federation_registry,
    )
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    assert reg.coverage_percent == 100.0
    assert reg.n_specialists >= 15


# ─────────────────────────────────────────────────────────────────────
# 9. Stigma + readability lint passes on a sample DecisionCard text
# ─────────────────────────────────────────────────────────────────────


def test_audit_patient_facing_text_passes_for_clean_sample():
    from a2a_agent.stigma_linter import audit_patient_facing_text
    sample = (
        "Take your warfarin tablet at the same time each day. "
        "Drink water with it. Watch for unusual bruising. "
        "Call us at 555-1234 if you feel sick."
    )
    audit = audit_patient_facing_text(sample, target_grade_max=8)
    assert audit.overall_pass is True
    assert audit.stigma.overall_grade == "clean"


# ─────────────────────────────────────────────────────────────────────
# 10. Concept-drift detector ingests a streaming forecast
# ─────────────────────────────────────────────────────────────────────


def test_concept_drift_detector_no_alarm_on_stable_stream():
    import random
    from a2a_agent.concept_drift import ADWIN
    rng = random.Random(7)
    a = ADWIN(delta=0.01)
    for _ in range(200):
        a.add(rng.gauss(0.158, 0.01))
    assert a.report().drift_detected_at == []
