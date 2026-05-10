"""healthcare.compute_admission_hnp_draft -- Phase 2.2 SCRIBE-4.

Admission History and Physical (H&P). Pure-deterministic extraction
from a FHIR Bundle + clinician-supplied narrative blocks.

Sections (US standard):
  1. Chief complaint
  2. History of present illness (HPI)
  3. Past medical / surgical history (PMH/PSH)
  4. Medications
  5. Allergies
  6. Review of systems (ROS) -- placeholders
  7. Physical Examination -- placeholders
  8. Labs / Imaging
  9. Assessment + Plan (problem-by-problem)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from shared.schemas import ClinicalNoteDraft, ClinicalNoteSection

from ._scribe_helpers import (
    coverage_pct,
    estimate_reading_minutes,
    extract_active_problems,
    extract_allergies,
    extract_encounter,
    extract_medications,
    extract_procedures,
    extract_recent_observations,
    extract_soap_section,
    llm_polish_enabled,
    llm_polish_sections,
)



async def compute_admission_hnp_draft(
    patient_reference: str,
    chief_complaint: str,
    fhir_bundle: dict | None = None,
    hpi_narrative: str | None = None,
    pmh_narrative: str | None = None,
    psh_narrative: str | None = None,
    ros_narrative: str | None = None,
    pe_narrative: str | None = None,
    assessment_plan: dict[str, str] | None = None,
) -> ClinicalNoteDraft:
    """Draft an admission History and Physical.

    Args:
        patient_reference: FHIR Patient reference.
        chief_complaint: Patient-stated reason for admission. MUST be
            non-empty.
        fhir_bundle: FHIR R4 Bundle.
        hpi_narrative: Free-text HPI from the admitting clinician.
        pmh_narrative: Past Medical History narrative (else falls back
            to extracted Conditions).
        psh_narrative: Past Surgical History (else falls back to
            extracted Procedures).
        ros_narrative: ROS narrative (else uses placeholder block).
        pe_narrative: Physical Exam narrative (else uses placeholder).
        assessment_plan: Optional `problem_id -> plan_text` map.

    Returns:
        ClinicalNoteDraft with the H&P sections.
    """
    # noqa: ABSTAIN-GUARD -- fail-fast when the FHIR bundle is absent or empty.
    # An H&P without structured patient data would produce a note whose every
    # section is a placeholder; an orchestrator upstream cannot distinguish that
    # from a real note and may present it to a clinician as if it were complete.
    if not fhir_bundle or not fhir_bundle.get("entry"):
        return ClinicalNoteDraft(
            note_type="admission_hnp",
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
                "missing_fhir_bundle: compute_admission_hnp_draft requires "
                "a non-empty FHIR Bundle with at least one entry. "
                "No demo fallback in live mode."
            ),
            references=[
                "Bates' Guide to Physical Examination and History Taking, 13th ed.",
                "Joint Commission Hospital Accreditation Standards (2024) "
                "history-and-physical requirements.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- the H&P has 4 sections that are NOT
    # extractable from a FHIR Bundle: chief_complaint, HPI narrative,
    # ROS narrative, PE narrative. These are clinician-supplied. A note
    # rendered with `[chief complaint not provided]` or the default
    # `_DEFAULT_ROS_BLOCK`/`_DEFAULT_PE_BLOCK` placeholder strings is
    # not a clinically usable artefact.
    #
    # Smart-recovery: when the EHR carries a previously-dictated H&P
    # / progress note as a DocumentReference, we read the matching
    # sections verbatim. No fabrication: when a section is missing, the
    # abstain below fires unchanged.
    if not (hpi_narrative or "").strip():
        recovered = extract_soap_section(fhir_bundle, "hpi")
        if recovered:
            hpi_narrative = recovered
    if not (pmh_narrative or "").strip():
        recovered = extract_soap_section(fhir_bundle, "pmh")
        if recovered:
            pmh_narrative = recovered
    if not (psh_narrative or "").strip():
        recovered = extract_soap_section(fhir_bundle, "psh")
        if recovered:
            psh_narrative = recovered
    if not (ros_narrative or "").strip():
        recovered = extract_soap_section(fhir_bundle, "ros")
        if recovered:
            ros_narrative = recovered
    if not (pe_narrative or "").strip():
        recovered = extract_soap_section(fhir_bundle, "pe")
        if recovered:
            pe_narrative = recovered

    missing_clinician_inputs: list[str] = []
    if not (chief_complaint or "").strip():
        missing_clinician_inputs.append(
            "chief_complaint (patient-stated reason for admission)"
        )
    if not (hpi_narrative or "").strip():
        missing_clinician_inputs.append(
            "hpi_narrative (free-text history of present illness)"
        )
    if not (ros_narrative or "").strip():
        missing_clinician_inputs.append(
            "ros_narrative (free-text review of systems)"
        )
    if not (pe_narrative or "").strip():
        missing_clinician_inputs.append(
            "pe_narrative (free-text physical examination findings)"
        )
    if missing_clinician_inputs:
        return ClinicalNoteDraft(
            note_type="admission_hnp",
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
                "missing_clinician_inputs: compute_admission_hnp_draft "
                "cannot produce an H&P when the clinician-supplied "
                "narrative sections are absent. The FHIR Bundle fills "
                "PMH, PSH, medications, allergies, and labs; the "
                "following sections require explicit clinician input "
                "and were not supplied: "
                + "; ".join(missing_clinician_inputs) + "."
            ),
            references=[
                "Bates' Guide to Physical Examination and History Taking, 13th ed.",
                "Joint Commission Hospital Accreditation Standards (2024) "
                "history-and-physical requirements.",
            ],
        )

    sections: list[ClinicalNoteSection] = []
    encounter_id, _ = extract_encounter(fhir_bundle)

    # 1. Chief complaint
    sections.append(ClinicalNoteSection(
        section_id="chief_complaint", title="Chief Complaint",
        body=chief_complaint,
    ))

    # 2. HPI
    sections.append(ClinicalNoteSection(
        section_id="hpi", title="History of Present Illness",
        body=hpi_narrative,
    ))

    # 3. PMH/PSH
    pmh_lines: list[str] = []
    pmh_cited: list[str] = []
    if pmh_narrative:
        pmh_lines.append(pmh_narrative)
    problem_bullets, problem_cited = extract_active_problems(fhir_bundle)
    if problem_bullets:
        if pmh_lines:
            pmh_lines.append("")
        pmh_lines.append("Active conditions on chart:")
        pmh_lines.extend(problem_bullets)
        pmh_cited.extend(problem_cited)
    if not pmh_lines:
        pmh_lines.append("[No PMH narrative or coded conditions]")
    sections.append(ClinicalNoteSection(
        section_id="pmh", title="Past Medical History",
        body="\n".join(pmh_lines), cited_evidence_ids=pmh_cited,
    ))

    psh_lines: list[str] = []
    psh_cited: list[str] = []
    if psh_narrative:
        psh_lines.append(psh_narrative)
    proc_bullets, proc_cited = extract_procedures(fhir_bundle)
    if proc_bullets:
        if psh_lines:
            psh_lines.append("")
        psh_lines.append("Past procedures on chart:")
        psh_lines.extend(proc_bullets)
        psh_cited.extend(proc_cited)
    if not psh_lines:
        psh_lines.append("[No PSH narrative or coded procedures]")
    sections.append(ClinicalNoteSection(
        section_id="psh", title="Past Surgical History",
        body="\n".join(psh_lines), cited_evidence_ids=psh_cited,
    ))

    # 4. Medications
    med_bullets, med_cited = extract_medications(fhir_bundle)
    sections.append(ClinicalNoteSection(
        section_id="medications", title="Medications",
        body="\n".join(med_bullets) if med_bullets
            else "[No active medications recorded]",
        cited_evidence_ids=med_cited,
    ))

    # 5. Allergies
    allergy_bullets, allergy_cited = extract_allergies(fhir_bundle)
    sections.append(ClinicalNoteSection(
        section_id="allergies", title="Allergies",
        body="\n".join(allergy_bullets) if allergy_bullets
            else "  - NKDA (no known drug allergies recorded)",
        cited_evidence_ids=allergy_cited,
    ))

    # 6. ROS
    sections.append(ClinicalNoteSection(
        section_id="ros", title="Review of Systems",
        body=ros_narrative,
    ))

    # 7. PE
    sections.append(ClinicalNoteSection(
        section_id="pe", title="Physical Examination",
        body=pe_narrative,
    ))

    # 8. Labs / imaging
    obs_bullets, obs_cited = extract_recent_observations(
        fhir_bundle, max_items=10,
    )
    sections.append(ClinicalNoteSection(
        section_id="labs", title="Labs and Imaging",
        body="\n".join(obs_bullets) if obs_bullets
            else "[No recent observations extracted]",
        cited_evidence_ids=obs_cited,
    ))

    # 9. A&P
    ap_lines: list[str] = []
    assessment_plan = assessment_plan or {}
    for cited_id in problem_cited:
        plan_text = assessment_plan.get(cited_id, "[plan TBD]")
        ap_lines.append(f"  - {cited_id}: {plan_text}")
    if not ap_lines:
        ap_lines.append(
            "[No coded problems extracted; A&P to be documented "
            "after history/exam completion.]"
        )
    sections.append(ClinicalNoteSection(
        section_id="ap", title="Assessment and Plan",
        body="\n".join(ap_lines), cited_evidence_ids=problem_cited,
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

    return ClinicalNoteDraft(
        note_type="admission_hnp",
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
        abstain_recommended=False,
        abstain_reason=None,
        references=[
            "Bates' Guide to Physical Examination and History Taking, 13th ed.",
            "Joint Commission Hospital Accreditation Standards (2024) "
            "history-and-physical requirements.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_admission_hnp_draft)
