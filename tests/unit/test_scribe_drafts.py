"""Unit tests for the four Phase 2.2 scribe-agent tools."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.admission_hnp_draft import compute_admission_hnp_draft
from mcp_server.tools.consult_letter_draft import compute_consult_letter_draft
from mcp_server.tools.discharge_summary_draft import (
    compute_discharge_summary_draft,
)
from mcp_server.tools.progress_note_draft import compute_progress_note_draft


def _run(coro):
    return asyncio.run(coro)


def _bundle() -> dict:
    """A reasonably full FHIR bundle covering the resources the scribe
    tools consume."""
    return {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {
            "resourceType": "Patient", "id": "pt-jane",
            "name": [{"given": ["Jane"], "family": "Doe"}],
            "birthDate": "1955-01-01", "gender": "female",
        }},
        {"resource": {
            "resourceType": "Encounter", "id": "enc-1",
            "class": {"code": "IMP", "display": "Inpatient"},
            "period": {"start": "2026-04-20T00:00:00Z",
                          "end": "2026-04-29T00:00:00Z"},
        }},
        {"resource": {
            "resourceType": "Condition", "id": "cond-chf",
            "code": {
                "coding": [{
                    "code": "I50.21",
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "display": "Acute on chronic systolic CHF",
                }],
                "text": "Acute on chronic systolic CHF",
            },
            "recordedDate": "2026-04-20",
        }},
        {"resource": {
            "resourceType": "Condition", "id": "cond-aki",
            "code": {
                "coding": [{
                    "code": "N17.9",
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "display": "AKI unspecified",
                }],
                "text": "AKI unspecified",
            },
        }},
        {"resource": {
            "resourceType": "Observation", "id": "obs-cr",
            "code": {"coding": [{"code": "2160-0", "display": "Creatinine"}]},
            "valueQuantity": {"value": 1.4, "unit": "mg/dL"},
            "effectiveDateTime": "2026-04-28T08:00:00Z",
        }},
        {"resource": {
            "resourceType": "Observation", "id": "obs-bnp",
            "code": {"coding": [{"code": "30934-4", "display": "BNP"}]},
            "valueQuantity": {"value": 800, "unit": "pg/mL"},
            "effectiveDateTime": "2026-04-26T10:00:00Z",
        }},
        {"resource": {
            "resourceType": "MedicationRequest", "id": "med-furo",
            "medicationCodeableConcept": {"text": "furosemide 40 mg PO daily"},
            "authoredOn": "2026-04-21",
        }},
        {"resource": {
            "resourceType": "MedicationRequest", "id": "med-lis",
            "medicationCodeableConcept": {"text": "lisinopril 10 mg PO daily"},
            "authoredOn": "2026-04-21",
        }},
        {"resource": {
            "resourceType": "AllergyIntolerance", "id": "all-pen",
            "code": {"text": "Penicillin (hives)"},
        }},
        {"resource": {
            "resourceType": "Procedure", "id": "proc-echo",
            "code": {"text": "Transthoracic echocardiogram"},
            "performedDateTime": "2026-04-22",
        }},
    ]}


# ─────────────────────── progress_note ───────────────────────

def test_progress_note_has_soap_sections():
    out = _run(compute_progress_note_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle(),
        subjective_text="Patient reports improved dyspnea overnight.",
    ))
    assert out.note_type == "progress_note"
    sections = {s.section_id for s in out.sections}
    assert sections == {"subjective", "objective", "assessment", "plan"}


def test_progress_note_objective_cites_observations_and_meds():
    out = _run(compute_progress_note_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle(),
        subjective_text="Patient reports improved dyspnea overnight.",
        plan_overrides={"cond-chf": "Continue diuresis; goal -1L/24h"},
    ))
    obj = next(s for s in out.sections if s.section_id == "objective")
    cited = set(obj.cited_evidence_ids)
    assert "obs-cr" in cited
    assert "med-furo" in cited


def test_progress_note_plan_uses_overrides_when_supplied():
    out = _run(compute_progress_note_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle(),
        subjective_text="Patient reports improved dyspnea overnight.",
        plan_overrides={"cond-chf": "Continue diuresis; goal -1L/24h"},
    ))
    plan = next(s for s in out.sections if s.section_id == "plan")
    assert "Continue diuresis" in plan.body
    # cond-aki has no override, falls back to placeholder
    assert "[plan TBD]" in plan.body


def test_progress_note_abstains_on_empty_bundle():
    out = _run(compute_progress_note_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle={"resourceType": "Bundle", "entry": []},
    ))
    assert out.abstain_recommended is True


# ─────────────────────── discharge_summary ───────────────────────

def test_discharge_summary_has_11_sections():
    out = _run(compute_discharge_summary_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle(),
        hospital_course_text="Admitted with acute on chronic systolic CHF + AKI; "
                                  "diuresed with furosemide; renal recovery; discharged stable.",
        discharge_disposition="home_with_care",
        followup_plan=["Cardiology in 7-14 days"],
        patient_instructions=["Daily weights", "Low-salt diet"],
    ))
    assert out.note_type == "discharge_summary"
    section_ids = [s.section_id for s in out.sections]
    expected = {
        "demographics", "encounter", "diagnoses", "hospital_course",
        "procedures", "medications", "allergies", "labs",
        "condition_disposition", "followup", "patient_instructions",
    }
    assert expected <= set(section_ids)


def test_discharge_summary_demographics_cites_patient():
    out = _run(compute_discharge_summary_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle(),
        hospital_course_text="Admitted with CHF; diuresed; renal recovery; stable.",
        discharge_disposition="home_with_care",
        followup_plan=["Cardiology in 7-14 days"],
        patient_instructions=["Daily weights", "Low-salt diet"],
    ))
    demo = next(s for s in out.sections if s.section_id == "demographics")
    assert "pt-jane" in demo.cited_evidence_ids


def test_discharge_summary_carries_decision_card_action():
    decision_card = {
        "recommendation": {"action": "home_with_care", "confidence": "preferred"},
        "reasoning": {"free_text": "Stable on oral diuretics; low readmission risk."},
    }
    out = _run(compute_discharge_summary_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle(),
        decision_card=decision_card,
        followup_plan=["Cardiology in 7-14 days"],
        patient_instructions=["Daily weights", "Low-salt diet"],
    ))
    cd = next(s for s in out.sections if s.section_id == "condition_disposition")
    assert "home_with_care" in cd.body


# ─────────────────────── consult_letter ───────────────────────

def test_consult_letter_addresses_referrer():
    out = _run(compute_consult_letter_draft(
        patient_reference="Patient/pt-jane",
        referring_clinician="Smith",
        consultation_reason="Worsening renal function on ACE-I",
        fhir_bundle=_bundle(),
        consultant_specialty="nephrology",
        hpi_summary="72yo F with CHF on ACE-I; rising Cr over 2 weeks.",
        assessment_text="Likely pre-renal AKI on ACE-I; hold and recheck.",
        recommendations=["Hold lisinopril", "Recheck Cr in 48h"],
    ))
    salut = next(s for s in out.sections if s.section_id == "salutation")
    assert "Dr. Smith" in salut.body


def test_consult_letter_abstains_on_empty_reason():
    out = _run(compute_consult_letter_draft(
        patient_reference="Patient/pt-jane",
        referring_clinician="Smith",
        consultation_reason="",
        fhir_bundle=_bundle(),
    ))
    assert out.abstain_recommended is True


def test_consult_letter_recommendations_listed():
    out = _run(compute_consult_letter_draft(
        patient_reference="Patient/pt-jane",
        referring_clinician="Smith",
        consultation_reason="AKI workup",
        fhir_bundle=_bundle(),
        hpi_summary="72yo F with rising Cr on lisinopril.",
        assessment_text="Likely pre-renal AKI from RAAS inhibition.",
        recommendations=[
            "Hold lisinopril until creatinine recovers below 1.2 mg/dL",
            "Renal ultrasound to evaluate for obstruction",
        ],
    ))
    rec = next(s for s in out.sections if s.section_id == "recommendations")
    assert "Hold lisinopril" in rec.body
    assert "Renal ultrasound" in rec.body


# ─────────────────────── admission_hnp ───────────────────────

def test_admission_hnp_has_standard_sections():
    out = _run(compute_admission_hnp_draft(
        patient_reference="Patient/pt-jane",
        chief_complaint="Shortness of breath x 2 days",
        fhir_bundle=_bundle(),
        hpi_narrative="2-day history of progressive dyspnea + LE edema.",
        ros_narrative="Pertinent positives: orthopnea, PND. Negatives: chest pain, fever.",
        pe_narrative="Vitals stable. JVD 8 cm, bibasilar crackles, 2+ pitting edema.",
    ))
    assert out.note_type == "admission_hnp"
    section_ids = {s.section_id for s in out.sections}
    expected = {
        "chief_complaint", "hpi", "pmh", "psh", "medications",
        "allergies", "ros", "pe", "labs", "ap",
    }
    assert expected <= section_ids


def test_admission_hnp_abstains_on_empty_chief_complaint():
    out = _run(compute_admission_hnp_draft(
        patient_reference="Patient/pt-jane",
        chief_complaint="",
        fhir_bundle=_bundle(),
    ))
    assert out.abstain_recommended is True


def test_admission_hnp_ap_uses_problem_cited_ids():
    out = _run(compute_admission_hnp_draft(
        patient_reference="Patient/pt-jane",
        chief_complaint="SOB",
        fhir_bundle=_bundle(),
        hpi_narrative="2-day dyspnea progression.",
        ros_narrative="Orthopnea, PND. Denies chest pain.",
        pe_narrative="JVD 8cm, bibasilar crackles, 2+ pitting edema.",
        assessment_plan={"cond-chf": "Diurese with IV furosemide; goal -1L/24h"},
    ))
    ap = next(s for s in out.sections if s.section_id == "ap")
    assert "Diurese with IV furosemide" in ap.body
    assert "cond-chf" in ap.cited_evidence_ids


# ─────────────────────── shared properties ───────────────────────

_FULL_NARRATIVE_PROGRESS = {
    "subjective_text": "Patient reports improved dyspnea overnight.",
    "plan_overrides": {
        "cond-chf": "Continue diuresis; goal -1L/24h",
        "cond-aki": "Hold lisinopril; recheck Cr in 24h.",
    },
}
_FULL_NARRATIVE_DISCHARGE = {
    "hospital_course_text": "Admitted with CHF exacerbation; diuresed; discharged stable.",
    "discharge_disposition": "home_with_care",
    "followup_plan": ["Cardiology in 7-14 days"],
    "patient_instructions": ["Daily weights", "Low-salt diet"],
}
_FULL_NARRATIVE_CONSULT = {
    "hpi_summary": "72yo F with rising Cr on lisinopril.",
    "assessment_text": "Likely pre-renal AKI.",
    "recommendations": ["Hold lisinopril", "Recheck Cr in 48h"],
}
_FULL_NARRATIVE_HNP = {
    "hpi_narrative": "2-day dyspnea + LE edema.",
    "ros_narrative": "Orthopnea, PND. No fever.",
    "pe_narrative": "JVD 8cm, bibasilar crackles, 2+ edema.",
}


@pytest.mark.parametrize("tool_args", [
    ("progress_note", {
        "patient_reference": "Patient/pt-jane",
        "fhir_bundle": _bundle(),
        **_FULL_NARRATIVE_PROGRESS,
    }),
    ("discharge_summary", {
        "patient_reference": "Patient/pt-jane",
        "fhir_bundle": _bundle(),
        **_FULL_NARRATIVE_DISCHARGE,
    }),
    ("consult_letter", {
        "patient_reference": "Patient/pt-jane",
        "referring_clinician": "Smith",
        "consultation_reason": "AKI workup",
        "fhir_bundle": _bundle(),
        **_FULL_NARRATIVE_CONSULT,
    }),
    ("admission_hnp", {
        "patient_reference": "Patient/pt-jane",
        "chief_complaint": "SOB",
        "fhir_bundle": _bundle(),
        **_FULL_NARRATIVE_HNP,
    }),
])
def test_each_tool_returns_full_text_concatenating_sections(tool_args):
    note_type, kwargs = tool_args
    if note_type == "progress_note":
        out = _run(compute_progress_note_draft(**kwargs))
    elif note_type == "discharge_summary":
        out = _run(compute_discharge_summary_draft(**kwargs))
    elif note_type == "consult_letter":
        out = _run(compute_consult_letter_draft(**kwargs))
    else:
        out = _run(compute_admission_hnp_draft(**kwargs))
    for s in out.sections:
        assert s.body in out.full_text or s.title in out.full_text


@pytest.mark.parametrize("note_type", [
    "progress_note", "discharge_summary", "consult_letter", "admission_hnp",
])
def test_each_tool_estimates_reading_minutes(note_type):
    if note_type == "progress_note":
        out = _run(compute_progress_note_draft(
            patient_reference="Patient/pt-jane", fhir_bundle=_bundle(),
            **_FULL_NARRATIVE_PROGRESS,
        ))
    elif note_type == "discharge_summary":
        out = _run(compute_discharge_summary_draft(
            patient_reference="Patient/pt-jane", fhir_bundle=_bundle(),
            **_FULL_NARRATIVE_DISCHARGE,
        ))
    elif note_type == "consult_letter":
        out = _run(compute_consult_letter_draft(
            patient_reference="Patient/pt-jane",
            referring_clinician="x", consultation_reason="y",
            fhir_bundle=_bundle(),
            **_FULL_NARRATIVE_CONSULT,
        ))
    else:
        out = _run(compute_admission_hnp_draft(
            patient_reference="Patient/pt-jane",
            chief_complaint="abdominal pain",
            fhir_bundle=_bundle(),
            **_FULL_NARRATIVE_HNP,
        ))
    assert out.estimated_reading_minutes > 0
    assert out.coverage_pct >= 0


def test_no_llm_polish_by_default(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_SCRIBE_LLM_POLISH", raising=False)
    out = _run(compute_progress_note_draft(
        patient_reference="Patient/pt-jane", fhir_bundle=_bundle(),
        **_FULL_NARRATIVE_PROGRESS,
    ))
    assert out.contains_llm_polish is False
    assert out.llm_model_id is None
    for s in out.sections:
        assert s.is_llm_polished is False


# ─────────────────────── Phase 2 abstain guards ───────────────────────

def _bundle_no_patient() -> dict:
    """Bundle with all resource types EXCEPT Patient."""
    b = _bundle()
    b["entry"] = [
        e for e in b["entry"]
        if e.get("resource", {}).get("resourceType") != "Patient"
    ]
    return b


def _bundle_no_encounter() -> dict:
    """Bundle with all resource types EXCEPT Encounter."""
    b = _bundle()
    b["entry"] = [
        e for e in b["entry"]
        if e.get("resource", {}).get("resourceType") != "Encounter"
    ]
    return b


def test_discharge_summary_abstains_when_bundle_has_no_patient_resource():
    out = _run(compute_discharge_summary_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle_no_patient(),
        hospital_course_text="CHF exacerbation; diuresed; discharged.",
        discharge_disposition="home_with_care",
        followup_plan=["Cardiology in 7 days"],
        patient_instructions=["Daily weights"],
    ))
    assert out.abstain_recommended is True
    assert "missing_patient_resource" in (out.abstain_reason or "")
    assert out.sections == []


def test_discharge_summary_abstains_when_bundle_has_no_encounter_resource():
    out = _run(compute_discharge_summary_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=_bundle_no_encounter(),
        hospital_course_text="CHF exacerbation; diuresed; discharged.",
        discharge_disposition="home_with_care",
        followup_plan=["Cardiology in 7 days"],
        patient_instructions=["Daily weights"],
    ))
    assert out.abstain_recommended is True
    assert "missing_encounter_resource" in (out.abstain_reason or "")
    assert out.sections == []


def test_consult_letter_abstains_when_bundle_has_no_patient_resource():
    out = _run(compute_consult_letter_draft(
        patient_reference="Patient/pt-jane",
        referring_clinician="Smith",
        consultation_reason="AKI workup",
        fhir_bundle=_bundle_no_patient(),
        hpi_summary="Rising Cr on lisinopril.",
        assessment_text="Likely pre-renal AKI.",
        recommendations=["Hold lisinopril", "Recheck Cr in 48h"],
    ))
    assert out.abstain_recommended is True
    assert "missing_patient_resource" in (out.abstain_reason or "")
    assert out.sections == []


def test_caregiver_handoff_abstains_when_decision_card_has_no_action():
    from mcp_server.tools.caregiver_handoff import compute_caregiver_handoff
    card_without_action = {
        "patient_reference": "Patient/pt-jane",
        "recommendation": {},  # no action field
    }
    out = _run(compute_caregiver_handoff(
        decision_card=card_without_action,
        fhir_bundle=_bundle(),
    ))
    assert out.abstain_recommended is True
    assert "missing_decision_card_action" in (out.abstain_reason or "")


def test_discharge_qa_abstains_when_no_decision_card():
    from mcp_server.tools.discharge_qa import compute_discharge_qa
    out = _run(compute_discharge_qa(
        questions=["When can I go home?"],
        decision_card=None,
    ))
    assert out.abstain_recommended is True
    assert "missing_decision_card" in (out.abstain_reason or "")


def test_discharge_qa_action_answer_without_action():
    from mcp_server.tools.discharge_qa import compute_discharge_qa
    card_without_action = {
        "patient_reference": "Patient/pt-jane",
        "recommendation": {},  # no action
    }
    out = _run(compute_discharge_qa(
        questions=["When can I go home?"],
        decision_card=card_without_action,
    ))
    # Tool should not abstain (card IS present), but item answer must
    # not contain a fabricated placeholder.
    assert out.abstain_recommended is False
    assert out.items
    answer = out.items[0].answer
    assert "[discharge plan not specified]" not in answer
    assert "Cannot answer without a discharge action" in answer
