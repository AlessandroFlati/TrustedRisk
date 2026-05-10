"""Per-specialist keyword router for the A2A chat surface.

Each specialist's `build_specialist_handler` previously returned a
discovery summary regardless of the prompt: useful for marketplace
discovery, useless for actually exercising the tools from a chat
client. This module declares, per specialist slug, a small set of
routes mapping keyword triggers to a concrete tool callable.

A chat prompt against a specialist runs through `route_for(slug, prompt)`;
the first route whose keyword tuple matches the prompt wins. The
dispatcher then calls the bound async tool with inputs drawn from the
live FHIR context (supplied via `message.metadata`), extracts a
one-line summary from the Pydantic dump, and surfaces the full result
as an A2A artifact.

Route inputs are intentionally empty: the dispatcher's `_run_specialist_route`
overlays inputs from the live FHIR bundle and message metadata. If the
required inputs are absent, the tool abstains rather than running on
fabricated data.

Falling back to the discovery summary is deliberate: a vague prompt
('hello', 'who are you?') still hits the catalog instead of guessing
a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from mcp_server.tools.acs_disposition_decision import (
    compute_acs_disposition_decision,
)
from mcp_server.tools.admission_hnp_draft import compute_admission_hnp_draft
from mcp_server.tools.admission_triage import compute_admission_triage
from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
from mcp_server.tools.auto_coding import compute_cpt_suggest, compute_icd10_suggest
from mcp_server.tools.care_gap_detector import compute_care_gap_detector
from mcp_server.tools.caregiver_handoff import compute_caregiver_handoff
from mcp_server.tools.clinical_deterioration_score import (
    compute_clinical_deterioration_score,
)
from mcp_server.tools.consult_letter_draft import compute_consult_letter_draft
from mcp_server.tools.contrast_safety_check import compute_contrast_safety_check
from mcp_server.tools.differential_diagnosis_ranker import (
    compute_differential_diagnosis_ranker,
)
from mcp_server.tools.discharge_counseling import compute_discharge_counseling
from mcp_server.tools.discharge_summary_draft import compute_discharge_summary_draft
from mcp_server.tools.dka_severity import compute_dka_severity
from mcp_server.tools.empiric_antibiotic_selection import (
    compute_empiric_antibiotic_selection,
)
from mcp_server.tools.expected_value_of_intervention import (
    compute_expected_value_of_intervention,
)
from mcp_server.tools.heart_score import compute_heart_score
from mcp_server.tools.insurance_appeals import (
    compute_appeal_escalation_path,
    compute_appeal_letter_draft,
    compute_denial_letter_parse,
)
from mcp_server.tools.massive_transfusion_protocol import (
    compute_massive_transfusion_protocol,
)
from mcp_server.tools.medication_reconciliation import (
    compute_medication_reconciliation,
)
from mcp_server.tools.multimodal import (
    compute_dicom_sr_ingest,
    compute_ecg_qt_analyzer,
)
from mcp_server.tools.patient_faq import compute_patient_faq
from mcp_server.tools.pediatric_early_warning import compute_pediatric_early_warning
from mcp_server.tools.pgx import (
    compute_pgx_dose_adjustment,
    compute_pgx_drug_alternatives,
)
from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
from mcp_server.tools.preadmit_triage import (
    compute_symptom_red_flag_check,
    compute_when_to_seek_care,
)
from mcp_server.tools.preeclampsia_assessment import (
    compute_preeclampsia_assessment,
)
from mcp_server.tools.progress_note_draft import compute_progress_note_draft
from mcp_server.tools.psychiatric_admission_decision import (
    compute_psychiatric_admission_decision,
)
from mcp_server.tools.stroke_severity import compute_stroke_severity
from mcp_server.tools.stroke_thrombolysis_eligibility import (
    compute_stroke_thrombolysis_eligibility,
)
from mcp_server.tools.suicide_risk_assessment import compute_suicide_risk_assessment
from mcp_server.tools.trauma_severity_score import compute_trauma_severity_score
from mcp_server.tools.weight_based_dosing import compute_weight_based_dosing


async def _chained_appeal_letter(
    *,
    fhir_bundle: dict | None = None,
    letter_text: str | None = None,
    patient_summary: str | None = None,
    medical_necessity_argument: str | None = None,
) -> Any:
    """Parse the denial letter, then draft the appeal off the parsed
    DenialLetterParseReport. Avoids hand-crafting the parsed payload.

    When the caller did not pass `letter_text`, `patient_summary`, or
    `medical_necessity_argument` explicitly, derives each of them from
    concrete FHIR resources on `fhir_bundle`:
      - letter_text: from a DocumentReference whose attachment plaintext
        contains payer denial language (or simply the first
        DocumentReference if there is only one).
      - patient_summary: from Patient demographics + active Conditions.
      - medical_necessity_argument: from a DocumentReference whose
        plaintext body has an `ASSESSMENT` section (typically a
        clinician progress note).
    Each fallback traces to a structured FHIR field; nothing is
    invented. When the bundle has nothing usable, the underlying tools
    raise their own structured abstain rather than this helper.
    """
    if fhir_bundle is not None and not letter_text:
        letter_text = _extract_denial_letter_text(fhir_bundle)
    if fhir_bundle is not None and not patient_summary:
        patient_summary = _extract_patient_summary_text(fhir_bundle)
    if fhir_bundle is not None and not medical_necessity_argument:
        medical_necessity_argument = _extract_assessment_text(fhir_bundle)

    parsed = await compute_denial_letter_parse(letter_text=letter_text or "")
    return await compute_appeal_letter_draft(
        parsed_denial=parsed,
        patient_summary=patient_summary or "",
        medical_necessity_argument=medical_necessity_argument or "",
    )


async def _chained_appeal_escalation_path(
    *,
    fhir_bundle: dict | None = None,
    payer: str | None = None,
    starting_level: str = "internal_first_level",
) -> Any:
    """Resolve the appeal escalation path. When `payer` is missing,
    derives it from `Coverage.payor[].display` on the bundle.
    """
    if not payer and fhir_bundle is not None:
        payer = _extract_payer_name(fhir_bundle) or ""
    return await compute_appeal_escalation_path(
        payer=payer or "",
        starting_level=starting_level,
    )


def _extract_denial_letter_text(fhir_bundle: dict) -> str:
    """Pull the plaintext body of the most likely denial-letter
    DocumentReference. Looks for `denial` or `not approved` in the
    description / attachment.title; otherwise returns the first
    DocumentReference body that decodes cleanly.
    """
    import base64
    candidates: list[tuple[int, str]] = []
    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "DocumentReference":
            continue
        descr = (r.get("description") or "").lower()
        for content in r.get("content") or []:
            att = content.get("attachment") or {}
            data = att.get("data")
            if not isinstance(data, str):
                continue
            try:
                body = base64.b64decode(data).decode("utf-8", errors="ignore")
            except Exception:
                continue
            title = (att.get("title") or "").lower()
            score = 0
            if any(k in descr for k in ("denial", "denied",
                                              "not approved")):
                score += 10
            if any(k in title for k in ("denial", "denied",
                                              "not approved")):
                score += 10
            if "denied" in body.lower() or "denial" in body.lower():
                score += 5
            candidates.append((score, body))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _extract_patient_summary_text(fhir_bundle: dict) -> str:
    """Compose a 1-2 sentence patient summary from Patient demographics
    plus active Conditions on the bundle.
    """
    import datetime as _dt
    bits: list[str] = []
    patient = next(
        (e.get("resource") for e in fhir_bundle.get("entry") or []
         if isinstance(e, dict)
         and isinstance(e.get("resource"), dict)
         and e["resource"].get("resourceType") == "Patient"),
        None,
    )
    if isinstance(patient, dict):
        bd = patient.get("birthDate")
        age = None
        if isinstance(bd, str):
            try:
                born = _dt.date.fromisoformat(bd[:10])
                age = (_dt.date.today() - born).days // 365
            except ValueError:
                pass
        gender = patient.get("gender")
        descriptor = []
        if age is not None:
            descriptor.append(f"{age}-year-old")
        if gender:
            descriptor.append(gender)
        bits.append("Patient is a " + " ".join(descriptor) + ".")
    conditions: list[str] = []
    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Condition":
            continue
        clinical = (r.get("clinicalStatus") or {}).get("coding") or []
        if not any(c.get("code") == "active" for c in clinical):
            continue
        code = r.get("code") or {}
        text = (code.get("text") or "").strip()
        if not text:
            for c in (code.get("coding") or []):
                disp = (c.get("display") or "").strip()
                if disp:
                    text = disp
                    break
        if text:
            conditions.append(text)
    if conditions:
        bits.append(
            "Active problem list: " + "; ".join(conditions[:6]) + "."
        )
    return " ".join(bits)


def _extract_assessment_text(fhir_bundle: dict) -> str:
    """Pull the ASSESSMENT section from a clinical-note
    DocumentReference. Mirrors the helper used by the consult-letter
    tool. Returns "" when no clinical note has an ASSESSMENT header.
    """
    import base64, re
    headers = re.compile(
        r"(?:^|\n)\s*(ASSESSMENT(?:\s+AND\s+PLAN)?|A\s*&\s*P|IMPRESSION)\s*:?\s*",
        re.IGNORECASE,
    )
    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "DocumentReference":
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
            m = headers.search(body)
            if not m:
                continue
            tail = body[m.end():].strip()
            stop = re.search(
                r"\n\s*(PLAN|RECOMMENDATIONS|FOLLOW[-\s]*UP|"
                r"SIGNATURE|DISPOSITION)\s*:?",
                tail, re.IGNORECASE,
            )
            if stop:
                tail = tail[: stop.start()].strip()
            if tail:
                return tail
    return ""


def _extract_payer_name(fhir_bundle: dict) -> str:
    """Pull payer display name from the first active Coverage."""
    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Coverage":
            continue
        for p in r.get("payor") or []:
            disp = (p.get("display") or "").strip()
            if disp:
                return disp
    return ""


# --------------------------------------------------------------- route model


@dataclass(frozen=True)
class Route:
    keywords: tuple[str, ...]
    callable: Callable[..., Awaitable[Any]]
    tool_name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    summary_fields: tuple[str, ...] = field(default_factory=tuple)
    skill_label: str = ""


def _r(
    keywords: tuple[str, ...],
    fn: Callable[..., Awaitable[Any]],
    *,
    inputs: dict[str, Any] | None = None,
    summary_fields: tuple[str, ...] = (),
    skill_label: str = "",
) -> Route:
    return Route(
        keywords=tuple(k.lower() for k in keywords),
        callable=fn,
        tool_name=fn.__name__,
        inputs=inputs or {},
        summary_fields=summary_fields,
        skill_label=skill_label or fn.__name__,
    )


# --------------------------------------------------------------- routes


SPECIALIST_ROUTES: dict[str, list[Route]] = {
    "trustedrisk-acute": [
        _r(("news2", "deterioration", "met call", "met-call", "rapid response",
            "spiked a temp", "rr climbing", "rr is climbing",
            "vitals trending", "vitals concerning",
            "pt is decompensating", "patient is decompensating"),
           compute_clinical_deterioration_score,
           inputs={},
           summary_fields=("score_total", "severity_tier", "recommended_response"),
           skill_label="NEWS2 deterioration score"),
        _r(("triage", "esi", "ed admission"),
           compute_admission_triage,
           inputs={},
           summary_fields=("esi_level", "priority", "disposition"),
           skill_label="ED admission triage"),
        _r(("nihss", "stroke severity", "tpa", "thrombolysis", "lkw", "last known well"),
           compute_stroke_thrombolysis_eligibility,
           inputs={},
           summary_fields=("eligibility", "recommended_decision"),
           skill_label="Stroke reperfusion eligibility"),
        _r(("stroke", "neuro deficit"),
           compute_stroke_severity,
           inputs={},
           summary_fields=("score_total", "severity_tier", "lvo_suspected"),
           skill_label="NIHSS stroke severity"),
        _r(("heart score", "chest pain", "acs",
            "clutching his chest", "clutching her chest", "diaphoretic"),
           compute_heart_score,
           inputs={},
           summary_fields=("score_total", "risk_band", "thirty_day_mace_risk_pct"),
           skill_label="HEART score for chest pain"),
        _r(("polytrauma", "trauma activation", "iss score", "iss total",
            "trauma severity"),
           compute_trauma_severity_score,
           inputs={},
           summary_fields=("iss_total", "severity_tier", "trts_total"),
           skill_label="Trauma severity (ISS + T-RTS)"),
        _r(("massive transfusion", "mtp", "abc score"),
           compute_massive_transfusion_protocol,
           inputs={},
           summary_fields=("activate_mtp", "txa_recommended", "abc_score_total"),
           skill_label="Massive transfusion protocol gating"),
        _r(("dka", "ketoacidosis"),
           compute_dka_severity,
           inputs={},
           summary_fields=("severity_tier", "recommended_disposition"),
           skill_label="DKA severity"),
        _r(("contrast", "iodinated", "egfr"),
           compute_contrast_safety_check,
           inputs={},
           summary_fields=("safe_to_administer", "risk_tier", "premedication_required"),
           skill_label="IV contrast safety"),
        _r(("aki", "kdigo", "creatinine"),
           compute_aki_kdigo_stage,
           inputs={},
           summary_fields=("stage", "rifle_class", "dialysis_consideration"),
           skill_label="KDIGO AKI staging"),
        _r(("preeclampsia", "meows", "obstetric", "pregnan"),
           compute_preeclampsia_assessment,
           inputs={},
           summary_fields=("severity", "recommended_action"),
           skill_label="ACOG preeclampsia assessment"),
    ],

    "trustedrisk-discharge": [
        _r(("polypharmacy", "drug interaction", "ddi"),
           detect_polypharmacy_concerns,
           inputs={},
           summary_fields=("polypharmacy_severity", "n_high_risk", "n_medications"),
           skill_label="Polypharmacy and DDI scan"),
        _r(("medication reconciliation", "med rec", "med list"),
           compute_medication_reconciliation,
           inputs={},
           summary_fields=("n_discrepancies", "summary"),
           skill_label="Medication reconciliation"),
        _r(("counseling", "patient handout", "after-visit", "after visit"),
           compute_discharge_counseling,
           inputs={},
           summary_fields=("estimated_reading_minutes",),
           skill_label="Discharge counseling"),
        _r(("antibiotic", "antimicrobial", "empiric"),
           compute_empiric_antibiotic_selection,
           inputs={},
           summary_fields=("primary_regimen", "rationale"),
           skill_label="Empiric antibiotic selection"),
        _r(("care gap", "screening", "hedis gap"),
           compute_care_gap_detector,
           inputs={},
           summary_fields=("n_gaps",),
           skill_label="Care gap detector"),
    ],

    "trustedrisk-evidence": [
        _r(("differential", "ddx"),
           compute_differential_diagnosis_ranker,
           inputs={},
           summary_fields=("n_items", "overall_confidence"),
           skill_label="Grounded differential diagnosis"),
    ],

    "trustedrisk-mental-health": [
        _r(("suicide", "self-harm", "self harm", "ending things",
            "ideation", "cssrs", "c-ssrs"),
           compute_suicide_risk_assessment,
           inputs={},
           summary_fields=("risk_level", "recommended_disposition"),
           skill_label="C-SSRS suicide risk assessment"),
        _r(("psychiatric admission", "involuntary", "5150", "section 12",
            "danger to self", "danger to others", "grave disab", "manic"),
           compute_psychiatric_admission_decision,
           inputs={},
           summary_fields=("disposition", "legal_pathway"),
           skill_label="Psychiatric admission disposition"),
    ],

    "trustedrisk-pediatric": [
        _r(("pews", "early warning", "pediatric", "kid", "child", "retract",
            "young child", "infant", "toddler", "rapid breathing",
            "tachypnea", "year-old with cough", "year old with cough",
            "mom brought in", "parents brought in"),
           compute_pediatric_early_warning,
           inputs={},
           summary_fields=("score_total", "severity_tier", "recommended_response"),
           skill_label="Pediatric early warning (PEWS)"),
        _r(("dose", "dosing", "weight-based", "weight based", "ceftriaxone",
            "amox", "mg/kg"),
           compute_weight_based_dosing,
           inputs={},
           summary_fields=("recommended_dose_mg", "frequency", "rationale"),
           skill_label="Weight-based pediatric dosing"),
    ],

    "trustedrisk-pa": [
        _r(("prior auth", "pa letter", "auth letter", "draft", "request",
            "ozempic", "glp-1", "mri", "semaglutide"),
           _chained_appeal_letter,
           inputs={},
           summary_fields=("appeal_level", "n_cite_backs"),
           skill_label="PA appeal letter draft"),
    ],

    "trustedrisk-patient": [
        _r(("post-discharge", "after surgery", "what should i", "what does",
            "watch for", "warning sign", "blood thinner", "what each"),
           compute_patient_faq,
           inputs={},
           summary_fields=("answer_text",),
           skill_label="Patient FAQ"),
        _r(("handoff", "caregiver", "family"),
           compute_caregiver_handoff,
           inputs={},
           summary_fields=("estimated_reading_minutes",),
           skill_label="Caregiver hand-off"),
    ],

    "trustedrisk-coder": [
        _r(("icd-10", "icd10", "icd"),
           compute_icd10_suggest,
           inputs={},
           summary_fields=("n_suggestions", "primary_icd10"),
           skill_label="ICD-10 auto-coding"),
        _r(("cpt",),
           compute_cpt_suggest,
           inputs={},
           summary_fields=("n_suggestions", "primary_cpt"),
           skill_label="CPT auto-coding"),
    ],

    "trustedrisk-pgx": [
        _r(("clopidogrel", "warfarin", "ssri", "tca", "metabolizer",
            "cyp", "vkorc", "pgx", "pharmacogenomic"),
           compute_pgx_drug_alternatives,
           inputs={},
           summary_fields=("recommended_alternative", "rationale"),
           skill_label="PGx drug alternatives"),
    ],

    "trustedrisk-preadmit": [
        _r(("red flag", "warning sign", "should i", "is this serious",
            "go to the er"),
           compute_symptom_red_flag_check,
           inputs={},
           summary_fields=("urgency", "matched_red_flags"),
           skill_label="Symptom red-flag check"),
        _r(("when to seek", "wait or go", "see a doctor", "schedule"),
           compute_when_to_seek_care,
           inputs={},
           summary_fields=("urgency", "recommended_route"),
           skill_label="When-to-seek-care guidance"),
    ],

    # Population-level capabilities (HEDIS Stars forecast, syndromic
    # surveillance, vaccine outreach, outbreak heatmap) are intentionally
    # NOT exposed as single-patient specialist routes: their required
    # inputs (cohort_summary, observed_counts_by_syndrome, overdue_by_vaccine,
    # population_by_geo) are panel-/county-level and cannot come from a
    # single patient's FHIR bundle. The same capabilities remain available
    # via the population-level workflows: hedis_quality_improvement,
    # population_outreach, care_gap_closure_pipeline.

    "trustedrisk-population": [
        _r(("cost-effectiveness", "cost effectiveness", "qaly", "icer",
            "evoi", "formulary", "tirzepatide", "screening"),
           compute_expected_value_of_intervention,
           inputs={},
           summary_fields=("icer_per_qaly_usd", "is_cost_effective", "evoi_usd"),
           skill_label="Cost-effectiveness + EVOI"),
    ],

    "trustedrisk-appeals": [
        _r(("denial", "denied", "rejected", "parse"),
           compute_denial_letter_parse,
           inputs={},
           summary_fields=("payer", "denial_reasons", "appeal_window_days"),
           skill_label="Denial letter parser"),
        _r(("appeal letter", "draft appeal", "push back",
            "infliximab", "remicade", "biologic", "pushed back",
            "denied her", "denied him", "denied medication"),
           _chained_appeal_letter,
           inputs={},
           summary_fields=("appeal_level", "n_cite_backs"),
           skill_label="Appeal letter draft"),
        _r(("escalation", "next step", "external review", "peer-to-peer"),
           _chained_appeal_escalation_path,
           inputs={},
           summary_fields=("steps", "max_levels"),
           skill_label="Appeal escalation path"),
    ],

    "trustedrisk-multimodal": [
        _r(("ecg", "qt", "qtc", "12-lead", "12 lead", "amiodarone",
            "methadone", "torsades"),
           compute_ecg_qt_analyzer,
           inputs={},
           summary_fields=("qtc_bazett_ms", "qtc_fridericia_ms", "is_prolonged"),
           skill_label="ECG QT analysis"),
        _r(("dicom", "imaging report", "ct report", "radiology"),
           compute_dicom_sr_ingest,
           inputs={},
           summary_fields=("modality", "body_part", "rationale"),
           skill_label="DICOM SR ingest"),
    ],

    "trustedrisk-scribe": [
        _r(("h&p", "h and p", "history and physical", "admission note"),
           compute_admission_hnp_draft,
           inputs={},
           summary_fields=("note_type", "coverage_pct",
                              "estimated_reading_minutes"),
           skill_label="Admission H&P draft"),
        _r(("discharge summary", "discharge note"),
           compute_discharge_summary_draft,
           inputs={},
           summary_fields=("coverage_pct", "estimated_reading_minutes"),
           skill_label="Discharge summary draft"),
        _r(("consult", "consultation", "letter to"),
           compute_consult_letter_draft,
           inputs={},
           summary_fields=("coverage_pct", "estimated_reading_minutes"),
           skill_label="Consult letter draft"),
        _r(("progress note", "soap", "post-op", "follow-up note"),
           compute_progress_note_draft,
           inputs={},
           summary_fields=("note_type", "coverage_pct"),
           skill_label="Progress note draft"),
    ],
}


# --------------------------------------------------------------- public API


def route_for(slug: str, prompt: str) -> Route | None:
    """Return the first route whose keyword tuple matches the prompt."""
    routes = SPECIALIST_ROUTES.get(slug)
    if not routes:
        return None
    p = (prompt or "").lower()
    for r in routes:
        if any(k in p for k in r.keywords):
            return r
    return None


def routes_for(slug: str) -> list[Route]:
    return list(SPECIALIST_ROUTES.get(slug, []))
