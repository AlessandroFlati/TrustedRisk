"""healthcare.compute_consult_letter_draft -- Phase 2.2 SCRIBE-3.

Specialty consult / referral letter. Drafted from a FHIR Bundle + a
referring-clinician identifier + the consultation reason + (optional)
the consultant's recommendations.

Sections:
  1. Salutation (referrer)
  2. Patient demographics
  3. Reason for consultation
  4. Brief HPI
  5. Pertinent findings (PE, labs, imaging)
  6. Assessment
  7. Recommendations
  8. Closing

Use case: outpatient or inpatient referral letters that consultants
draft back to the primary team or the patient's PCP. Replaces the
~ 30 min/letter dictation burden.
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
    extract_patient_summary,
    extract_recent_observations,
    extract_soap_section,
    llm_polish_enabled,
    llm_polish_sections,
)


def _derive_hpi_from_bundle(fhir_bundle: dict | None) -> str:
    """Compose a brief HPI from FHIR-structured data on the bundle.

    The output is a deterministic narrative restatement of:
      - Encounter.reasonCode of the most recent encounter,
      - Condition.code of the active problem list,
      - Observation values flagged as "pertinent" (recent vital signs +
        recent labs).

    Every string in the output traces back to a concrete FHIR field;
    nothing is invented. Returns "" when the bundle has none of the
    above (the caller then abstains).
    """
    if not isinstance(fhir_bundle, dict):
        return ""
    bits: list[str] = []
    _enc_id, encounter_summary = extract_encounter(fhir_bundle)
    if encounter_summary:
        bits.append(f"Patient presenting with {encounter_summary}.")
    problem_bullets, _ = extract_active_problems(fhir_bundle)
    if problem_bullets:
        bits.append(
            "Active problems on the chart: " + "; ".join(problem_bullets[:6])
            + "."
        )
    obs_bullets, _ = extract_recent_observations(fhir_bundle, max_items=6)
    if obs_bullets:
        bits.append("Pertinent findings: " + "; ".join(obs_bullets) + ".")
    return " ".join(bits)


def _extract_clinical_notes_text(fhir_bundle: dict | None) -> list[str]:
    """Pull plaintext bodies of `DocumentReference` resources whose
    category is `clinical-note` (US Core). Each body is returned as a
    raw string -- the caller decides how to use it (assessment text,
    HPI, etc.).
    """
    import base64
    if not isinstance(fhir_bundle, dict):
        return []
    out: list[str] = []
    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "DocumentReference":
            continue
        is_clinical = False
        for cat in r.get("category") or []:
            for c in (cat.get("coding") or []):
                if c.get("code") == "clinical-note":
                    is_clinical = True
        if not is_clinical:
            continue
        for content in r.get("content") or []:
            att = content.get("attachment") or {}
            data = att.get("data")
            if not isinstance(data, str):
                continue
            try:
                body = base64.b64decode(data).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if body.strip():
                out.append(body.strip())
    return out


def _derive_assessment_from_bundle(fhir_bundle: dict | None) -> str:
    """Pull assessment narrative from a clinical-note DocumentReference.

    Looks for a section header such as "ASSESSMENT", "A&P", or
    "ASSESSMENT AND PLAN" inside the plaintext body and returns the
    paragraph(s) that follow. Returns "" when no clinical note is
    present or no assessment header is found -- the caller then abstains.
    """
    import re
    notes = _extract_clinical_notes_text(fhir_bundle)
    if not notes:
        return ""
    headers = re.compile(
        r"(?:^|\n)\s*(ASSESSMENT(?:\s+AND\s+PLAN)?|A\s*&\s*P|IMPRESSION)\s*:?\s*",
        re.IGNORECASE,
    )
    for body in notes:
        m = headers.search(body)
        if not m:
            continue
        tail = body[m.end():].strip()
        # Stop at the next section header to keep the assessment scoped.
        next_header = re.search(
            r"\n\s*(PLAN|RECOMMENDATIONS|FOLLOW[-\s]*UP|SIGNATURE|"
            r"DISPOSITION)\s*:?",
            tail,
            re.IGNORECASE,
        )
        if next_header:
            tail = tail[: next_header.start()].strip()
        if tail:
            return tail
    return ""


def _derive_recommendations_from_bundle(
    fhir_bundle: dict | None,
) -> list[str]:
    """Pull recommendations from `CarePlan.activity[].detail.description`,
    `ServiceRequest.code.text` / `note`, and `Goal.description.text` on
    the bundle. Returns the structured items only; never invents
    boilerplate. An empty list signals the caller to abstain.
    """
    if not isinstance(fhir_bundle, dict):
        return []
    out: list[str] = []
    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict):
            continue
        rt = r.get("resourceType")
        if rt == "CarePlan":
            for act in r.get("activity") or []:
                detail = act.get("detail") or {}
                desc = detail.get("description")
                if isinstance(desc, str) and desc.strip():
                    out.append(desc.strip())
        elif rt == "ServiceRequest":
            code = r.get("code") or {}
            text = (code.get("text") or "").strip()
            if not text:
                for c in (code.get("coding") or []):
                    text = (c.get("display") or "").strip()
                    if text:
                        break
            if text:
                out.append(text)
            for note in (r.get("note") or []):
                t = (note.get("text") or "").strip()
                if t:
                    out.append(t)
        elif rt == "Goal":
            desc = (r.get("description") or {}).get("text")
            if isinstance(desc, str) and desc.strip():
                out.append(desc.strip())
    return out


async def compute_consult_letter_draft(
    patient_reference: str,
    referring_clinician: str,
    consultation_reason: str,
    fhir_bundle: dict | None = None,
    consultant_specialty: str | None = None,
    hpi_summary: str | None = None,
    assessment_text: str | None = None,
    recommendations: list[str] | None = None,
) -> ClinicalNoteDraft:
    """Draft a consult/referral letter back to the referring clinician.

    Args:
        patient_reference: FHIR Patient reference.
        referring_clinician: Name (or NPI) of the referring clinician.
        consultation_reason: Free-text statement of the question being
            consulted on.
        fhir_bundle: FHIR R4 Bundle for the patient.
        consultant_specialty: Optional consultant specialty (e.g.
            'cardiology', 'nephrology').
        hpi_summary: Optional HPI summary text from the consultant.
        assessment_text: Optional consultant assessment narrative.
        recommendations: Optional list of recommendations.

    Returns:
        ClinicalNoteDraft with the consult letter sections.
    """
    # SHARP-on-MCP fallback: when the caller does not pass a bundle, fetch
    # the bound patient's bundle directly from the FHIR server using
    # X-Patient-ID + X-FHIR-Server-URL. patient_reference is treated as a
    # display label only -- the canonical FHIR resource id comes from the
    # SHARP context, not from the free-text reference the agent passes.
    if not fhir_bundle or not fhir_bundle.get("entry"):
        try:
            _pid = await resolve_patient_id(None)
            fhir_bundle = await fetch_patient_bundle(_pid)
        except Exception:
            pass  # falls through to the abstain below when context is also absent

    # noqa: ABSTAIN-GUARD -- fail-fast when the FHIR bundle is absent or empty.
    # A consult letter without structured patient data would produce a letter
    # whose demographics, findings, and problem-list sections are all
    # placeholders; a downstream orchestrator cannot verify it as a real note.
    if not fhir_bundle or not fhir_bundle.get("entry"):
        return ClinicalNoteDraft(
            note_type="consult_letter",
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
                "missing_fhir_bundle: compute_consult_letter_draft requires "
                "a non-empty FHIR Bundle with at least one entry. "
                "No demo fallback in live mode."
            ),
            references=[
                "American Medical Association Style Guide for Clinical Letters.",
                "AHRQ TeamSTEPPS hand-off communication standards.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- the consult letter has 3 sections that are
    # NOT extractable from a FHIR Bundle (clinician-supplied): the HPI
    # summary, the consultant's assessment narrative, and the
    # recommendations list. A letter where any of these renders as
    # "[X to be completed]" is not a clinically usable artefact.
    # consultation_reason is the question the referring clinician asks;
    # without it the consult letter has no purpose, so abstain hard.
    # Smart-recovery: a chart-attached DocumentReference whose body has
    # a "Reason for consultation" section carries the same intent --
    # read it verbatim before abstaining.
    if not (consultation_reason or "").strip():
        recovered = extract_soap_section(fhir_bundle, "consultation_reason")
        if recovered:
            consultation_reason = recovered
    if not (consultation_reason or "").strip():
        return ClinicalNoteDraft(
            note_type="consult_letter",
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
                "missing_consultation_reason: "
                "compute_consult_letter_draft requires the question the "
                "referring clinician is asking. A letter with no reason "
                "for consultation is not a clinically usable artefact."
            ),
            references=[
                "American Medical Association Style Guide for Clinical Letters.",
                "AHRQ TeamSTEPPS hand-off communication standards.",
            ],
        )

    # The remaining 3 narrative sections (HPI, assessment, recs) are
    # filled from concrete FHIR resources when the caller does not pass
    # them explicitly. Every fallback path traces to a structured field
    # on the bundle: HPI from Encounter+Conditions+Observations,
    # assessment from a DocumentReference clinical-note ASSESSMENT
    # section, recommendations from CarePlan.activity / ServiceRequest /
    # Goal. Boilerplate is never injected -- if the bundle has nothing
    # to say, the tool abstains below.
    if not (hpi_summary or "").strip():
        derived_hpi = _derive_hpi_from_bundle(fhir_bundle)
        if derived_hpi:
            hpi_summary = derived_hpi
    if not (assessment_text or "").strip():
        derived_assessment = _derive_assessment_from_bundle(fhir_bundle)
        if derived_assessment:
            assessment_text = derived_assessment
    if not recommendations:
        derived_recs = _derive_recommendations_from_bundle(fhir_bundle)
        if derived_recs:
            recommendations = derived_recs

    # If, after auto-derivation, any of HPI/assessment/recs is still
    # empty AND the bundle has nothing to offer, abstain. This is the
    # rare path: an FHIR Bundle with only a Patient resource and no
    # Conditions/Observations.
    missing_clinician_inputs: list[str] = []
    if not (hpi_summary or "").strip():
        missing_clinician_inputs.append(
            "hpi_summary (free-text history of present illness: onset, "
            "character, severity, modifying factors)"
        )
    if not (assessment_text or "").strip():
        missing_clinician_inputs.append(
            "assessment_text (consultant's narrative assessment)"
        )
    if not recommendations:
        missing_clinician_inputs.append(
            "recommendations (list of recommendations the consultant is "
            "returning to the referring clinician)"
        )
    if missing_clinician_inputs:
        return ClinicalNoteDraft(
            note_type="consult_letter",
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
                "missing_clinician_inputs: compute_consult_letter_draft "
                "cannot produce a consult letter when the consultant's "
                "narrative sections are absent. The FHIR Bundle fills "
                "demographics and findings; the following sections "
                "require explicit consultant input and were not "
                "supplied: " + "; ".join(missing_clinician_inputs) + "."
            ),
            references=[
                "American Medical Association Style Guide for Clinical Letters.",
                "AHRQ TeamSTEPPS hand-off communication standards.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- fail-fast when the Patient resource is absent.
    demo_text, demo_cited, patient_complete = extract_patient_summary(fhir_bundle)
    if not patient_complete:
        return ClinicalNoteDraft(
            note_type="consult_letter",
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
                "is missing name, birthDate, or gender. A consult letter "
                "cannot be drafted without verified patient demographics."
            ),
            references=[
                "American Medical Association Style Guide for Clinical Letters.",
                "AHRQ TeamSTEPPS hand-off communication standards.",
            ],
        )

    sections: list[ClinicalNoteSection] = []
    encounter_id, _ = extract_encounter(fhir_bundle)

    # 1. Salutation
    salutation_text = f"Dear Dr. {referring_clinician},"
    sections.append(ClinicalNoteSection(
        section_id="salutation", title="Salutation",
        body=salutation_text,
    ))

    # 2. Demographics
    intro_lines = [demo_text]
    intro_lines.append(
        f"Thank you for referring this patient to "
        f"{consultant_specialty or 'our specialty'} for evaluation."
    )
    sections.append(ClinicalNoteSection(
        section_id="patient_summary", title="Patient",
        body="\n".join(intro_lines),
        cited_evidence_ids=demo_cited,
    ))

    # 3. Reason for consultation
    sections.append(ClinicalNoteSection(
        section_id="reason", title="Reason for Consultation",
        body=consultation_reason,
    ))

    # 4. HPI
    sections.append(ClinicalNoteSection(
        section_id="hpi", title="Brief History of Present Illness",
        body=hpi_summary,
    ))

    # 5. Pertinent findings
    obs_bullets, obs_cited = extract_recent_observations(
        fhir_bundle, max_items=8,
    )
    findings_text = (
        "\n".join(obs_bullets)
        if obs_bullets
        else "[No recent labs/observations to summarise]"
    )
    sections.append(ClinicalNoteSection(
        section_id="findings", title="Pertinent Findings",
        body=findings_text, cited_evidence_ids=obs_cited,
    ))

    # 6. Assessment (problem list + consultant text)
    problem_bullets, problem_cited = extract_active_problems(fhir_bundle)
    assess_lines: list[str] = []
    if assessment_text:
        assess_lines.append(assessment_text)
    if problem_bullets:
        if assess_lines:
            assess_lines.append("")
        assess_lines.append("Active problems:")
        assess_lines.extend(problem_bullets)
    sections.append(ClinicalNoteSection(
        section_id="assessment", title="Assessment",
        body="\n".join(assess_lines),
        cited_evidence_ids=problem_cited,
    ))

    # 7. Recommendations
    rec_lines = [f"  - {rec}" for rec in recommendations]
    sections.append(ClinicalNoteSection(
        section_id="recommendations", title="Recommendations",
        body="\n".join(rec_lines),
    ))

    # 8. Closing
    closing_text = (
        "Thank you for the opportunity to participate in the patient's "
        "care. Please contact me with any questions or for further "
        f"discussion.\n\nSincerely,\n{consultant_specialty or 'Consulting Clinician'}"
    )
    sections.append(ClinicalNoteSection(
        section_id="closing", title="Closing", body=closing_text,
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
        note_type="consult_letter",
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
            "American Medical Association Style Guide for Clinical Letters.",
            "AHRQ TeamSTEPPS hand-off communication standards.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_consult_letter_draft)
