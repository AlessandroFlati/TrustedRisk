"""healthcare.compute_discharge_summary_draft -- Phase 2.2 SCRIBE-2.

Multi-section discharge summary per Joint Commission standards.
Pure-deterministic extraction from a FHIR Bundle + optional
DecisionCard + hospital-course narrative.

Sections:
  1. Patient demographics
  2. Admission + discharge dates (Encounter)
  3. Admitting + discharge diagnoses
  4. Hospital course narrative (clinician-supplied or DecisionCard
     reasoning)
  5. Procedures performed
  6. Discharge medications + reconciliation deltas
  7. Allergies
  8. Pertinent labs / imaging
  9. Discharge condition + disposition
 10. Follow-up plan
 11. Patient instructions
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from shared.schemas import ClinicalNoteDraft, ClinicalNoteSection

from ..fhir.client import fetch_patient_bundle, resolve_patient_id
from ._scribe_helpers import (
    coverage_pct,
    estimate_reading_minutes,
    extract_active_problems,
    extract_allergies,
    extract_encounter,
    extract_medications,
    extract_patient_summary,
    extract_procedures,
    extract_recent_observations,
    extract_soap_section,
    llm_polish_enabled,
    llm_polish_sections,
)


async def compute_discharge_summary_draft(
    patient_reference: str,
    fhir_bundle: dict | None = None,
    decision_card: dict | None = None,
    hospital_course_text: str | None = None,
    discharge_disposition: str | None = None,
    followup_plan: list[str] | None = None,
    patient_instructions: list[str] | None = None,
) -> ClinicalNoteDraft:
    """Draft a Joint-Commission-shaped discharge summary.

    Args:
        patient_reference: FHIR Patient reference.
        fhir_bundle: FHIR R4 Bundle.
        decision_card: Optional DecisionCard payload (provides the
            hospital-course narrative when no clinician text is
            supplied).
        hospital_course_text: Free-text narrative of the admission.
        discharge_disposition: One of discharge_home / home_with_care
            / snf / continued_admission etc.
        followup_plan: List of bullet items for follow-up.
        patient_instructions: List of patient-language instructions.

    Returns:
        ClinicalNoteDraft with the 11 standard sections.
    """
    # SHARP-on-MCP fallback: when the caller does not pass a bundle, fetch
    # the bound patient's bundle directly from the FHIR server using
    # X-Patient-ID + X-FHIR-Server-URL. patient_reference is treated as a
    # display label only.
    if not fhir_bundle or not fhir_bundle.get("entry"):
        try:
            _pid = await resolve_patient_id(None)
            fhir_bundle = await fetch_patient_bundle(_pid)
        except Exception:
            pass

    # noqa: ABSTAIN-GUARD -- fail-fast when the FHIR bundle is absent or empty.
    # A discharge summary without a FHIR Bundle produces a document whose
    # demographics, diagnoses, medications, labs, and procedures sections are
    # all placeholders.  The tool returns abstain_recommended=True with an
    # explicit reason so a downstream orchestrator cannot present the empty
    # draft as a real discharge summary.
    if not fhir_bundle or not fhir_bundle.get("entry"):
        return ClinicalNoteDraft(
            note_type="discharge_summary",
            patient_reference=patient_reference,
            encounter_reference=None,
            authored_at_iso=None,
            sections=[],
            full_text="",
            n_cite_backs=0,
            coverage_pct=0.0,
            estimated_reading_minutes=0.0,
            abstain_recommended=True,
            abstain_reason=(
                "missing_fhir_bundle: compute_discharge_summary_draft requires "
                "a non-empty FHIR Bundle with at least one entry. "
                "No demo fallback in live mode."
            ),
            references=[
                "Joint Commission Hospital Accreditation Standards (2024).",
                "AHRQ Project RED (Re-Engineered Discharge) bundle.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- the discharge summary has 4 sections that are
    # NOT extractable from a FHIR Bundle (they are clinician-supplied):
    # hospital_course narrative, follow-up plan, patient instructions,
    # discharge disposition.  Producing a draft with placeholder text in
    # any of these sections would let a downstream caller render
    # "[Hospital course narrative not provided]" as a real document.
    # Fail-fast and ask the caller to supply the missing inputs.
    # Smart-recovery: read missing narrative sections verbatim from a
    # DocumentReference clinical-note when the EHR has one. Each fallback
    # is read-only on the bundle; nothing is fabricated.
    if not (hospital_course_text or "").strip():
        recovered = extract_soap_section(fhir_bundle, "hospital_course")
        if recovered:
            hospital_course_text = recovered
    if not (discharge_disposition or "").strip():
        recovered = extract_soap_section(fhir_bundle, "discharge_disposition")
        if recovered:
            discharge_disposition = recovered.split("\n", 1)[0].strip()
    if not followup_plan:
        recovered = extract_soap_section(fhir_bundle, "followup_plan")
        if recovered:
            followup_plan = [
                line.strip(" -*")
                for line in recovered.splitlines()
                if line.strip(" -*")
            ]
    if not patient_instructions:
        recovered = extract_soap_section(fhir_bundle, "patient_instructions")
        if recovered:
            patient_instructions = [
                line.strip(" -*")
                for line in recovered.splitlines()
                if line.strip(" -*")
            ]

    has_course_narrative = bool(
        (hospital_course_text or "").strip()
        or ((decision_card or {}).get("reasoning") or {}).get("free_text")
    )
    has_disposition = bool(
        (discharge_disposition or "").strip()
        or ((decision_card or {}).get("recommendation") or {}).get("action")
    )
    missing_clinician_inputs: list[str] = []
    if not has_course_narrative:
        missing_clinician_inputs.append(
            "hospital_course_text (free-text narrative) OR "
            "decision_card.reasoning.free_text"
        )
    if not has_disposition:
        missing_clinician_inputs.append(
            "discharge_disposition (one of: discharge_home / home_with_care "
            "/ snf / rehab / hospice / continued_admission) OR "
            "decision_card.recommendation.action"
        )
    if not followup_plan:
        missing_clinician_inputs.append(
            "followup_plan (list of follow-up items: PCP visit window, "
            "specialist referrals, pending labs/imaging)"
        )
    if not patient_instructions:
        missing_clinician_inputs.append(
            "patient_instructions (list of patient-language instructions: "
            "activity, diet, red flags, when to return)"
        )
    if missing_clinician_inputs:
        return ClinicalNoteDraft(
            note_type="discharge_summary",
            patient_reference=patient_reference,
            encounter_reference=None,
            authored_at_iso=None,
            sections=[],
            full_text="",
            n_cite_backs=0,
            coverage_pct=0.0,
            estimated_reading_minutes=0.0,
            abstain_recommended=True,
            abstain_reason=(
                "missing_clinician_inputs: compute_discharge_summary_draft "
                "cannot produce a Joint-Commission-shaped draft when the "
                "clinician-supplied sections are absent. The FHIR Bundle "
                "fills demographics, encounter dates, diagnoses, "
                "procedures, medications, allergies, and labs; the "
                "following sections require explicit input from the "
                "treating clinician and were not supplied: "
                + "; ".join(missing_clinician_inputs) + "."
            ),
            references=[
                "Joint Commission Hospital Accreditation Standards (2024).",
                "AHRQ Project RED (Re-Engineered Discharge) bundle.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- fail-fast when the Patient resource is absent.
    demo_text, demo_cited, patient_complete = extract_patient_summary(fhir_bundle)
    if not patient_complete:
        return ClinicalNoteDraft(
            note_type="discharge_summary",
            patient_reference=patient_reference,
            encounter_reference=None,
            authored_at_iso=None,
            sections=[],
            full_text="",
            n_cite_backs=0,
            coverage_pct=0.0,
            estimated_reading_minutes=0.0,
            abstain_recommended=True,
            abstain_reason=(
                "missing_patient_resource: bundle does not contain a Patient "
                "resource for the active patient_id, or the Patient resource "
                "is missing name, birthDate, or gender. A discharge summary "
                "cannot be drafted without verified patient demographics."
            ),
            references=[
                "Joint Commission Hospital Accreditation Standards (2024).",
                "AHRQ Project RED (Re-Engineered Discharge) bundle.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- fail-fast when no Encounter resource is present.
    encounter_id, encounter_summary = extract_encounter(fhir_bundle)
    if encounter_id is None:
        return ClinicalNoteDraft(
            note_type="discharge_summary",
            patient_reference=patient_reference,
            encounter_reference=None,
            authored_at_iso=None,
            sections=[],
            full_text="",
            n_cite_backs=0,
            coverage_pct=0.0,
            estimated_reading_minutes=0.0,
            abstain_recommended=True,
            abstain_reason=(
                "missing_encounter_resource: bundle does not contain any "
                "Encounter for the active patient_id. A discharge summary "
                "requires an Encounter to anchor admission and discharge dates."
            ),
            references=[
                "Joint Commission Hospital Accreditation Standards (2024).",
                "AHRQ Project RED (Re-Engineered Discharge) bundle.",
            ],
        )

    sections: list[ClinicalNoteSection] = []

    # 1. Demographics
    sections.append(ClinicalNoteSection(
        section_id="demographics", title="Patient Demographics",
        body=demo_text, cited_evidence_ids=demo_cited,
    ))

    # 2. Encounter dates
    # encounter_summary is None when the Encounter resource has no
    # period.start (encounter present but dates not recorded in FHIR).
    sections.append(ClinicalNoteSection(
        section_id="encounter",
        title="Admission and Discharge",
        body=(
            encounter_summary
            if encounter_summary
            else f"Encounter {encounter_id} (admission/discharge dates not recorded in FHIR)."
        ),
        cited_evidence_ids=[encounter_id],
    ))

    # 3. Diagnoses (admitting + discharge from active problem list)
    problem_bullets, problem_cited = extract_active_problems(fhir_bundle)
    sections.append(ClinicalNoteSection(
        section_id="diagnoses", title="Diagnoses",
        body="\n".join(problem_bullets) if problem_bullets
            else "[No coded diagnoses extracted]",
        cited_evidence_ids=problem_cited,
    ))

    # 4. Hospital course
    course_body = hospital_course_text or (
        (decision_card or {}).get("reasoning", {}).get("free_text")
        if decision_card else None
    )
    sections.append(ClinicalNoteSection(
        section_id="hospital_course", title="Hospital Course",
        body=course_body,
    ))

    # 5. Procedures
    proc_bullets, proc_cited = extract_procedures(fhir_bundle)
    sections.append(ClinicalNoteSection(
        section_id="procedures", title="Procedures",
        body="\n".join(proc_bullets) if proc_bullets
            else "[No procedures recorded]",
        cited_evidence_ids=proc_cited,
    ))

    # 6. Discharge medications
    med_bullets, med_cited = extract_medications(fhir_bundle)
    sections.append(ClinicalNoteSection(
        section_id="medications", title="Discharge Medications",
        body="\n".join(med_bullets) if med_bullets
            else "[No medication requests extracted]",
        cited_evidence_ids=med_cited,
    ))

    # 7. Allergies
    allergy_bullets, allergy_cited = extract_allergies(fhir_bundle)
    sections.append(ClinicalNoteSection(
        section_id="allergies", title="Allergies",
        body="\n".join(allergy_bullets) if allergy_bullets
            else "  - NKDA (no known drug allergies recorded)",
        cited_evidence_ids=allergy_cited,
    ))

    # 8. Pertinent labs
    lab_bullets, lab_cited = extract_recent_observations(
        fhir_bundle, max_items=12,
    )
    sections.append(ClinicalNoteSection(
        section_id="labs", title="Pertinent Labs and Imaging",
        body="\n".join(lab_bullets) if lab_bullets
            else "[No recent observations extracted]",
        cited_evidence_ids=lab_cited,
    ))

    # 9. Discharge condition + disposition
    cond_body_lines: list[str] = []
    if decision_card and "recommendation" in decision_card:
        action = decision_card.get("recommendation", {}).get("action")
        confidence = decision_card.get("recommendation", {}).get("confidence")
        cond_body_lines.append(
            f"Recommended action: {action} (confidence {confidence})"
        )
    if discharge_disposition:
        cond_body_lines.append(f"Disposition: {discharge_disposition}")
    sections.append(ClinicalNoteSection(
        section_id="condition_disposition",
        title="Discharge Condition and Disposition",
        body="\n".join(cond_body_lines),
    ))

    # 10. Follow-up
    sections.append(ClinicalNoteSection(
        section_id="followup", title="Follow-up",
        body="\n".join(f"  - {item}" for item in followup_plan),
    ))

    # 11. Patient instructions
    sections.append(ClinicalNoteSection(
        section_id="patient_instructions",
        title="Patient Instructions",
        body="\n".join(f"  - {item}" for item in patient_instructions),
    ))

    contains_polish = False
    polish_model: str | None = None
    if llm_polish_enabled():
        sections, polish_model = await llm_polish_sections(sections)
        if polish_model is not None:
            contains_polish = True
            for s in sections:
                s.is_llm_polished = True

    full_text = "\n\n".join(
        f"## {s.title}\n{s.body}" for s in sections
    )
    n_cite_backs = sum(len(s.cited_evidence_ids) for s in sections)
    cov = coverage_pct(sections, fhir_bundle)

    abstain_recommended = (
        not problem_bullets and not med_bullets and not lab_bullets
    )
    abstain_reason = (
        "FHIR bundle had no Conditions, MedicationRequests, or "
        "Observations; discharge summary would be empty of structured "
        "data."
        if abstain_recommended else None
    )

    return ClinicalNoteDraft(
        note_type="discharge_summary",
        patient_reference=patient_reference,
        encounter_reference=encounter_id,
        authored_at_iso=datetime.now(timezone.utc).isoformat(),
        sections=sections,
        full_text=full_text,
        contains_llm_polish=contains_polish,
        llm_model_id=polish_model,
        n_cite_backs=n_cite_backs,
        coverage_pct=cov,
        estimated_reading_minutes=estimate_reading_minutes(full_text),
        abstain_recommended=abstain_recommended,
        abstain_reason=abstain_reason,
        references=[
            "Joint Commission Hospital Accreditation Standards (2024).",
            "AHRQ Project RED (Re-Engineered Discharge) bundle.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_discharge_summary_draft)
