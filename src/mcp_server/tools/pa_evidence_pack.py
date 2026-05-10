"""healthcare.compute_pa_evidence_pack -- Phase 2.1 PA-1.

Builds a structured evidence pack from a FHIR Bundle (or
chart-intelligence-extracted entities) supporting a Prior Authorization
request. The pack is the canonical substrate for every downstream PA
tool: rules-match, appeal-likelihood, letter-draft.

Pure-deterministic: no LLM. Every evidence item carries a cite-back to
the FHIR resource ID or chart line span -- so the LLM polish layer
downstream cannot invent clinical facts.

Evidence-strength score computation (0-1):
    - 0.30 × dx_specificity     (specific ICD-10 codes present)
    - 0.25 × prior_treatment_depth (number of treatments tried, capped)
    - 0.20 × recent_observations (lab/vital recency within 90 days)
    - 0.15 × contraindication_documentation
    - 0.10 × narrative_present  (chart excerpts supporting necessity)

Sources: AMA 2024 Prior Authorization Survey; AHRQ medical-necessity
checklist; payer-specific Coverage Determination policies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from shared.schemas import (
    PAEvidenceItem,
    PAEvidencePack,
    PARequestedService,
)


# ─────────────────────────────────────────────────────────────────────
# Evidence extraction helpers
# ─────────────────────────────────────────────────────────────────────

def _is_specific_icd10(code: str | None) -> bool:
    """True if the code is non-empty and not an unspecified
    placeholder (the .9 / .99 'unspecified' codes)."""
    if not code:
        return False
    code = code.upper()
    # Reject the most common unspecified codes (rough heuristic)
    if code.endswith(".9") or code.endswith(".99"):
        return False
    if "UNSPECIFIED" in code:
        return False
    return True


def _observation_age_days(obs_iso: str | None) -> int | None:
    if not obs_iso:
        return None
    try:
        when = datetime.fromisoformat(obs_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - when
    return max(0, int(delta.total_seconds() / 86400))


def _extract_diagnoses(bundle_entries: Iterable[dict]) -> list[PAEvidenceItem]:
    out: list[PAEvidenceItem] = []
    for entry in bundle_entries:
        resource = entry.get("resource", {}) if isinstance(entry, dict) else {}
        if resource.get("resourceType") != "Condition":
            continue
        code_block = resource.get("code", {})
        coding = (code_block.get("coding") or [{}])[0]
        code = coding.get("code")
        text = code_block.get("text") or coding.get("display") or "unknown"
        relevance = 0.9 if _is_specific_icd10(code) else 0.5
        out.append(PAEvidenceItem(
            kind="condition",
            fhir_resource_id=resource.get("id"),
            text=str(text),
            code=code,
            code_system=coding.get("system"),
            timestamp_iso=resource.get("recordedDate"),
            relevance_score=relevance,
        ))
    return out


def _extract_observations(bundle_entries: Iterable[dict]) -> list[PAEvidenceItem]:
    out: list[PAEvidenceItem] = []
    for entry in bundle_entries:
        resource = entry.get("resource", {}) if isinstance(entry, dict) else {}
        if resource.get("resourceType") != "Observation":
            continue
        code_block = resource.get("code", {})
        coding = (code_block.get("coding") or [{}])[0]
        eff = (resource.get("effectiveDateTime")
                  or resource.get("effectiveInstant"))
        age_days = _observation_age_days(eff)
        # Recency drives relevance -- within 30 days = 1.0, 30-90 = 0.7,
        # 90-180 = 0.4, >180 = 0.2
        if age_days is None:
            relevance = 0.4
        elif age_days <= 30:
            relevance = 1.0
        elif age_days <= 90:
            relevance = 0.7
        elif age_days <= 180:
            relevance = 0.4
        else:
            relevance = 0.2
        value_text = (
            resource.get("valueQuantity", {}).get("value")
            or resource.get("valueString")
            or "(unstructured)"
        )
        unit = resource.get("valueQuantity", {}).get("unit", "")
        text = (
            f"{coding.get('display') or code_block.get('text', 'observation')}: "
            f"{value_text} {unit}".strip()
        )
        out.append(PAEvidenceItem(
            kind="observation",
            fhir_resource_id=resource.get("id"),
            text=text,
            code=coding.get("code"),
            code_system=coding.get("system"),
            timestamp_iso=eff,
            relevance_score=relevance,
        ))
    return out


def _extract_prior_treatments(
    bundle_entries: Iterable[dict],
) -> list[PAEvidenceItem]:
    out: list[PAEvidenceItem] = []
    for entry in bundle_entries:
        resource = entry.get("resource", {}) if isinstance(entry, dict) else {}
        rt = resource.get("resourceType")
        if rt == "MedicationRequest":
            med_block = resource.get(
                "medicationCodeableConcept", {}
            ) or resource.get("medicationReference", {})
            text = (
                med_block.get("text")
                or (med_block.get("coding") or [{}])[0].get("display")
                or "unknown medication"
            )
            out.append(PAEvidenceItem(
                kind="medication_history",
                fhir_resource_id=resource.get("id"),
                text=str(text),
                code=(med_block.get("coding") or [{}])[0].get("code"),
                code_system=(
                    (med_block.get("coding") or [{}])[0].get("system")
                ),
                timestamp_iso=resource.get("authoredOn"),
                relevance_score=0.7,
            ))
        elif rt == "Procedure":
            proc_block = resource.get("code", {})
            coding = (proc_block.get("coding") or [{}])[0]
            out.append(PAEvidenceItem(
                kind="procedure",
                fhir_resource_id=resource.get("id"),
                text=str(
                    proc_block.get("text") or coding.get("display") or "procedure"
                ),
                code=coding.get("code"),
                code_system=coding.get("system"),
                timestamp_iso=resource.get("performedDateTime"),
                relevance_score=0.7,
            ))
    return out


def _strength_score(
    diagnoses: list[PAEvidenceItem],
    prior_treatments: list[PAEvidenceItem],
    observations: list[PAEvidenceItem],
    contraindications: list[PAEvidenceItem],
    chart_excerpts: list[PAEvidenceItem],
) -> float:
    dx_specificity = (
        sum(d.relevance_score for d in diagnoses) / max(len(diagnoses), 1)
        if diagnoses else 0.0
    )
    treatment_depth = min(len(prior_treatments) / 3.0, 1.0)
    recent_obs = (
        sum(o.relevance_score for o in observations) / max(len(observations), 1)
        if observations else 0.0
    )
    contra = min(len(contraindications) / 2.0, 1.0)
    narrative = min(len(chart_excerpts) / 2.0, 1.0)
    score = (
        0.30 * dx_specificity
        + 0.25 * treatment_depth
        + 0.20 * recent_obs
        + 0.15 * contra
        + 0.10 * narrative
    )
    return round(min(max(score, 0.0), 1.0), 3)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

async def compute_pa_evidence_pack(
    patient_reference: str,
    requested_service: PARequestedService | dict,
    payer: str,
    fhir_bundle: dict | None = None,
    chart_excerpts: list[dict] | None = None,
    contraindications: list[dict] | None = None,
) -> PAEvidencePack:
    """Build a structured PA evidence pack from a FHIR Bundle + optional
    chart excerpts.

    Args:
        patient_reference: FHIR Patient reference (e.g. "Patient/abc").
        requested_service: PARequestedService describing what's being
            authorized.
        payer: One of the supported PAYER_ID values.
        fhir_bundle: FHIR R4 Bundle (`Bundle.entry[]`) -- typical source
            for diagnoses / observations / med-history / procedures.
        chart_excerpts: Optional structured chart-intelligence output
            (e.g. from compute_clinical_ner) surfaced as supporting
            narrative.
        contraindications: Optional structured contraindication-to-
            alternative items (drives step-therapy override arguments).

    Returns:
        PAEvidencePack with the aggregated, cite-backable evidence.
    """
    if isinstance(requested_service, dict):
        requested_service = PARequestedService.model_validate(requested_service)

    # noqa: ABSTAIN-GUARD -- fail-fast when the FHIR bundle is absent or empty.
    # A PA evidence pack without a FHIR Bundle has no diagnoses, observations,
    # or prior-treatment evidence; every downstream tool (rules-match, appeal-
    # likelihood, letter-draft) would produce a groundless output.
    if not fhir_bundle or not fhir_bundle.get("entry"):
        return PAEvidencePack(
            patient_reference=patient_reference,
            requested_service=requested_service,
            payer=payer,                    # type: ignore[arg-type]
            diagnoses=[],
            relevant_observations=[],
            prior_treatments_tried=[],
            contraindications_to_alternatives=[],
            chart_excerpts=[],
            evidence_strength_score=0.0,
            n_evidence_items=0,
            abstain_recommended=True,
            abstain_reason=(
                "missing_fhir_bundle: compute_pa_evidence_pack requires "
                "a non-empty FHIR Bundle with at least one entry. "
                "No demo fallback in live mode."
            ),
            rationale="Abstained -- no FHIR Bundle provided.",
            references=[
                "AMA 2024 Prior Authorization Physician Survey.",
                "AHRQ medical-necessity documentation checklist.",
                "ICD-10-CM Official Guidelines for Coding and Reporting.",
            ],
        )

    entries = (
        (fhir_bundle or {}).get("entry", [])
        if isinstance(fhir_bundle, dict) else []
    )
    diagnoses = _extract_diagnoses(entries)
    observations = _extract_observations(entries)
    prior_treatments = _extract_prior_treatments(entries)

    chart_items: list[PAEvidenceItem] = []
    for ex in chart_excerpts or []:
        chart_items.append(PAEvidenceItem(
            kind="chart_excerpt",
            fhir_resource_id=ex.get("fhir_resource_id"),
            chart_line_span=tuple(ex["chart_line_span"])
                if ex.get("chart_line_span") else None,
            text=str(ex.get("text", "")),
            relevance_score=float(ex.get("relevance_score", 0.6)),
        ))

    contra_items: list[PAEvidenceItem] = []
    for ci in contraindications or []:
        contra_items.append(PAEvidenceItem(
            kind="chart_excerpt",
            fhir_resource_id=ci.get("fhir_resource_id"),
            text=str(ci.get("text", "")),
            relevance_score=float(ci.get("relevance_score", 0.7)),
        ))

    score = _strength_score(
        diagnoses, prior_treatments, observations, contra_items, chart_items,
    )
    n_total = (
        len(diagnoses) + len(observations) + len(prior_treatments)
        + len(contra_items) + len(chart_items)
    )

    abstain = False
    abstain_reason: str | None = None
    if not diagnoses:
        abstain = True
        abstain_reason = (
            "No Condition resources found in the FHIR bundle -- PA "
            "submission requires a coded diagnosis."
        )
    elif n_total < 3:
        abstain = True
        abstain_reason = (
            f"Evidence pack has only {n_total} item(s); below minimum "
            "threshold of 3 for a credible PA submission."
        )

    rationale = (
        f"Evidence pack for {requested_service.service_type} on "
        f"{patient_reference!r} against payer {payer!r}: "
        f"{len(diagnoses)} diagnosis/diagnoses, "
        f"{len(observations)} observation(s), "
        f"{len(prior_treatments)} prior treatment(s), "
        f"{len(contra_items)} contraindication(s), "
        f"{len(chart_items)} chart excerpt(s). "
        f"Evidence strength score = {score:.3f}."
    )

    return PAEvidencePack(
        patient_reference=patient_reference,
        requested_service=requested_service,
        payer=payer,                    # type: ignore[arg-type]
        diagnoses=diagnoses,
        relevant_observations=observations,
        prior_treatments_tried=prior_treatments,
        contraindications_to_alternatives=contra_items,
        chart_excerpts=chart_items,
        evidence_strength_score=score,
        n_evidence_items=n_total,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
        rationale=rationale,
        references=[
            "AMA 2024 Prior Authorization Physician Survey.",
            "AHRQ medical-necessity documentation checklist.",
            "ICD-10-CM Official Guidelines for Coding and Reporting.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_pa_evidence_pack)
