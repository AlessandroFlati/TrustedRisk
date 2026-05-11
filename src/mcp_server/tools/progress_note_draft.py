"""healthcare.compute_progress_note_draft -- Phase 2.2 SCRIBE-1.

Daily inpatient progress note in SOAP format. Pure-deterministic
extraction from a FHIR Bundle + a clinician-supplied `subjective_text`
(voice dictation or text input) + the prior note's plan deltas.

SOAP sections:
  - Subjective: clinician-supplied text (required; tool abstains if absent)
  - Objective: vitals + recent labs + active medications
  - Assessment: active problem list
  - Plan: per-problem [plan TBD] placeholders (clinician fills in inline)

Cite-back: every Objective bullet cites the FHIR Observation /
MedicationRequest ID; every Assessment bullet cites the Condition ID.
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
    extract_encounter,
    extract_medications,
    extract_recent_observations,
    extract_soap_section,
    llm_polish_enabled,
    llm_polish_sections,
    render_bullets_section,
)


async def compute_progress_note_draft(
    patient_reference: str,
    fhir_bundle: dict | None = None,
    subjective_text: str | None = None,
    plan_overrides: dict[str, str] | None = None,
) -> ClinicalNoteDraft:
    """Draft a daily progress note in SOAP format.

    Args:
        patient_reference: FHIR Patient reference.
        fhir_bundle: FHIR R4 Bundle with Observations + Conditions +
            MedicationRequests + (optional) Encounter.
        subjective_text: Clinician-provided subjective entry. May come
            from a voice-to-text dictation upstream.
        plan_overrides: Optional map of `problem_id -> plan_text` from
            the prior note or the clinician -- when present, replaces
            the per-problem `[plan TBD]` placeholder.

    Returns:
        ClinicalNoteDraft with SOAP-shaped sections.
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
    # A progress note without a FHIR Bundle has no vitals, medications, or
    # active problems; the Objective and Assessment sections would be entirely
    # placeholder text, which a downstream orchestrator cannot distinguish from
    # a real note.
    if not fhir_bundle or not fhir_bundle.get("entry"):
        return ClinicalNoteDraft(
            note_type="progress_note",
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
                "missing_fhir_bundle: compute_progress_note_draft requires "
                "a non-empty FHIR Bundle with at least one entry. "
                "No demo fallback in live mode."
            ),
            references=[
                "Weed LL. Medical records that guide and teach. NEJM 1968 (SOAP origin).",
                "Joint Commission Hospital Accreditation Standards (2024) "
                "documentation requirements.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- the SOAP Subjective section is
    # clinician-dictated and CANNOT be reconstructed from FHIR. A note
    # with `[No subjective entry provided]` is a placeholder, not a
    # clinically usable note. (Per-problem plan_overrides is optional:
    # the per-problem `[plan TBD]` is a structured ask the clinician
    # must complete inline, not a fabricated content -- the section
    # makes the gap explicit and traceable.)
    #
    # Smart-recovery: when the clinician has previously dictated a
    # SUBJECTIVE / HPI section in a DocumentReference attached to the
    # patient (e.g. yesterday's progress note), we honour it here
    # instead of forcing the chat-side to re-dictate. This is NOT
    # fabrication -- the text is read verbatim from a real
    # DocumentReference body. When no such section exists, the abstain
    # below fires unchanged.
    if not (subjective_text or "").strip():
        recovered = (
            extract_soap_section(fhir_bundle, "subjective")
            or extract_soap_section(fhir_bundle, "hpi")
        )
        if recovered:
            subjective_text = recovered
    missing_clinician_inputs: list[str] = []
    if not (subjective_text or "").strip():
        missing_clinician_inputs.append(
            "subjective_text (clinician-dictated subjective entry: "
            "patient-reported symptoms, complaints, goals)"
        )
    if missing_clinician_inputs:
        return ClinicalNoteDraft(
            note_type="progress_note",
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
                "missing_clinician_inputs: compute_progress_note_draft "
                "cannot produce a SOAP note when the clinician-supplied "
                "sections are absent. The FHIR Bundle fills the "
                "Objective (vitals, meds) and Assessment (active "
                "problems) sections; the following sections require "
                "explicit clinician input and were not supplied: "
                + "; ".join(missing_clinician_inputs) + "."
            ),
            references=[
                "Weed LL. Medical records that guide and teach. NEJM 1968 (SOAP origin).",
                "Joint Commission Hospital Accreditation Standards (2024) "
                "documentation requirements.",
            ],
        )

    encounter_id, _ = extract_encounter(fhir_bundle)

    # Subjective
    subj_body = subjective_text.strip()
    subj = ClinicalNoteSection(
        section_id="subjective", title="Subjective", body=subj_body,
        cited_evidence_ids=([encounter_id] if encounter_id else []),
    )

    # Objective: recent observations + medications
    vitals_bullets, vitals_cited = extract_recent_observations(
        fhir_bundle, max_items=10,
    )
    med_bullets, med_cited = extract_medications(fhir_bundle)
    obj_lines: list[str] = []
    if vitals_bullets:
        obj_lines.append("Vitals + labs (most recent):")
        obj_lines.extend(vitals_bullets)
    if med_bullets:
        if obj_lines:
            obj_lines.append("")
        obj_lines.append("Active medications:")
        obj_lines.extend(med_bullets)
    obj_body = (
        "\n".join(obj_lines)
        if obj_lines
        else "[No objective data extracted from the FHIR bundle.]"
    )
    obj = ClinicalNoteSection(
        section_id="objective", title="Objective", body=obj_body,
        cited_evidence_ids=vitals_cited + med_cited,
    )

    # Assessment
    problem_bullets, problem_cited = extract_active_problems(fhir_bundle)
    assess_body = (
        "\n".join(problem_bullets)
        if problem_bullets
        else "[No active problems extracted from the FHIR bundle.]"
    )
    assess = ClinicalNoteSection(
        section_id="assessment", title="Assessment", body=assess_body,
        cited_evidence_ids=problem_cited,
    )

    # Plan: per-problem placeholders or overrides
    plan_overrides = plan_overrides or {}
    plan_lines: list[str] = []
    for cited_id in problem_cited:
        override = plan_overrides.get(cited_id)
        if override:
            plan_lines.append(f"  - {cited_id}: {override}")
        else:
            plan_lines.append(f"  - {cited_id}: [plan TBD]")
    if not plan_lines:
        plan_lines.append("[No active problems -- plan not applicable.]")
    plan = ClinicalNoteSection(
        section_id="plan", title="Plan", body="\n".join(plan_lines),
        cited_evidence_ids=problem_cited,
    )

    sections = [subj, obj, assess, plan]
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
        not problem_bullets and not vitals_bullets and not med_bullets
    )
    abstain_reason = (
        "FHIR bundle had no Conditions, Observations, or MedicationRequests; "
        "the progress note would be empty of structured data."
        if abstain_recommended else None
    )

    return ClinicalNoteDraft(
        note_type="progress_note",
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
            "Weed LL. Medical records that guide and teach. NEJM 1968 (SOAP origin).",
            "Joint Commission Hospital Accreditation Standards (2024) "
            "documentation requirements.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_progress_note_draft)
