"""healthcare.compute_resolve_active_meds -- multi-source active-med resolver.

Real FHIR servers expose meds as `MedicationRequest`. Prompt Opinion's
synthetic-patient store scopes only `MedicationAdministration` and
`DocumentReference`, with the actual drug list embedded as markdown
inside the clinical-note body (Synthea-style header `# Medications`).

This tool tries the canonical FHIR sources in order, then falls back to
parsing `# Medications` sections out of free-text clinical notes. The
LLM should call this BEFORE `compute_rxnorm_ddi_lookup` /
`detect_polypharmacy_concerns` whenever the user asks "what is this
patient on?" so the downstream stateless tools have a list to chew on.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


_MED_SECTION_HEADER_RE = re.compile(
    r"^\s*#+\s*medications?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEXT_SECTION_HEADER_RE = re.compile(r"^\s*#+\s+\S", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)
_NEGATIVE_RE = re.compile(
    r"\bno\s+(active\s+)?medications?\b|\bnone\b|\bno\s+known\s+meds?\b",
    re.IGNORECASE,
)


class ResolvedMedication(BaseModel):
    """One active medication, with the source that surfaced it."""

    name: str
    source: Literal["MedicationRequest", "MedicationAdministration",
                       "DocumentReference"]
    document_date: str | None = None


class ActiveMedListReport(BaseModel):
    """Active medications resolved from FHIR via the SHARP context."""

    medications: list[ResolvedMedication] = Field(default_factory=list)
    distinct_names: list[str] = Field(default_factory=list)
    n_medication_request: int = 0
    n_medication_administration: int = 0
    n_documents_scanned: int = 0
    n_documents_with_med_section: int = 0
    method: Literal[
        "medication_request",
        "medication_administration",
        "document_reference_text",
        "no_meds_found",
    ]
    rationale: str


def _med_name_from_codeable(res: dict[str, Any]) -> str | None:
    """Pull a human-readable drug name from a FHIR Med* resource."""
    code = (
        res.get("medicationCodeableConcept")
        or res.get("medication", {}).get("concept")
        or {}
    )
    if isinstance(code, dict):
        text = code.get("text")
        if text:
            return str(text).strip()
        for c in code.get("coding") or []:
            disp = c.get("display")
            if disp:
                return str(disp).strip()
    ref = res.get("medicationReference") or {}
    if isinstance(ref, dict) and ref.get("display"):
        return str(ref["display"]).strip()
    return None


def _extract_meds_from_note(text: str) -> list[str]:
    """Pull bullet items out of a Synthea `# Medications` section."""
    if not text:
        return []
    header_match = _MED_SECTION_HEADER_RE.search(text)
    if not header_match:
        return []
    section_start = header_match.end()
    rest = text[section_start:]
    next_header = _NEXT_SECTION_HEADER_RE.search(rest)
    section = rest[: next_header.start()] if next_header else rest
    if _NEGATIVE_RE.search(section):
        return []
    bullets = [m.group(1).strip() for m in _BULLET_RE.finditer(section)]
    if bullets:
        return bullets
    # Fallback: lines of plain prose, one med per line.
    candidates = [
        ln.strip() for ln in section.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return candidates


def _decode_document_text(doc: dict[str, Any]) -> str:
    """Pull plaintext content out of a FHIR DocumentReference."""
    chunks: list[str] = []
    for content in doc.get("content") or []:
        attachment = content.get("attachment") or {}
        ctype = (attachment.get("contentType") or "").lower()
        if "text" not in ctype and "markdown" not in ctype and ctype != "":
            continue
        data = attachment.get("data")
        if data:
            try:
                import base64
                chunks.append(base64.b64decode(data).decode("utf-8", "ignore"))
            except Exception:
                pass
        if attachment.get("title"):
            chunks.append(str(attachment["title"]))
    description = doc.get("description")
    if description:
        chunks.append(str(description))
    return "\n".join(chunks)


def _doc_date(doc: dict[str, Any]) -> str | None:
    return (
        doc.get("date")
        or doc.get("indexed")
        or (doc.get("context", {}) or {}).get("period", {}).get("start")
    )


async def compute_resolve_active_meds(
    patient_id: str | None = None,
) -> ActiveMedListReport:
    """Resolve the active medication list for the SHARP-bound patient.

    Source order:
      1. `MedicationRequest` (status=active or unspecified).
      2. `MedicationAdministration` (used by PO synthetic-patient store).
      3. `DocumentReference` plaintext bodies, parsing `# Medications`
         sections out of Synthea-style clinical notes.

    The first non-empty source wins. The report's `method` field tells
    the caller which source produced the list, so downstream tools (DDI
    scan, polypharmacy detector) can decide how much to trust it.
    """
    pid = await resolve_patient_id(patient_id)
    bundle = await fetch_patient_bundle(pid)
    entries = bundle.get("entry") or []

    med_requests: list[dict[str, Any]] = []
    med_admins: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for e in entries:
        res = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(res, dict):
            continue
        rt = res.get("resourceType")
        if rt == "MedicationRequest":
            med_requests.append(res)
        elif rt == "MedicationAdministration":
            med_admins.append(res)
        elif rt == "DocumentReference":
            documents.append(res)

    resolved: list[ResolvedMedication] = []

    for mr in med_requests:
        status = (mr.get("status") or "active").lower()
        if status in ("stopped", "cancelled", "entered-in-error"):
            continue
        name = _med_name_from_codeable(mr)
        if name:
            resolved.append(ResolvedMedication(
                name=name, source="MedicationRequest"))
    if resolved:
        return _finalize(
            resolved, med_requests, med_admins, documents, 0,
            method="medication_request",
            rationale=(
                f"Resolved {len(resolved)} medications from "
                f"{len(med_requests)} MedicationRequest resources."
            ),
        )

    for ma in med_admins:
        status = (ma.get("status") or "completed").lower()
        if status in ("entered-in-error", "stopped"):
            continue
        name = _med_name_from_codeable(ma)
        if name:
            resolved.append(ResolvedMedication(
                name=name, source="MedicationAdministration"))
    if resolved:
        return _finalize(
            resolved, med_requests, med_admins, documents, 0,
            method="medication_administration",
            rationale=(
                f"Resolved {len(resolved)} medications from "
                f"{len(med_admins)} MedicationAdministration resources."
            ),
        )

    n_with_meds = 0
    documents_sorted = sorted(
        documents, key=lambda d: (_doc_date(d) or ""), reverse=True)
    for doc in documents_sorted:
        text = _decode_document_text(doc)
        meds = _extract_meds_from_note(text)
        if meds:
            n_with_meds += 1
            doc_date = _doc_date(doc)
            for m in meds:
                resolved.append(ResolvedMedication(
                    name=m, source="DocumentReference",
                    document_date=doc_date))
            # Most-recent note wins -- stop at the first note that lists meds.
            break

    if resolved:
        return _finalize(
            resolved, med_requests, med_admins, documents, n_with_meds,
            method="document_reference_text",
            rationale=(
                f"FHIR Med* resources empty; parsed `# Medications` from "
                f"the most recent of {len(documents)} clinical notes."
            ),
        )

    return _finalize(
        [], med_requests, med_admins, documents, n_with_meds,
        method="no_meds_found",
        rationale=(
            f"No active medications found across "
            f"{len(med_requests)} MedicationRequest, "
            f"{len(med_admins)} MedicationAdministration, and "
            f"{len(documents)} DocumentReference resources."
        ),
    )


def _finalize(
    resolved: list[ResolvedMedication],
    med_requests: list[dict[str, Any]],
    med_admins: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    n_with_meds: int,
    *,
    method: Literal[
        "medication_request",
        "medication_administration",
        "document_reference_text",
        "no_meds_found",
    ],
    rationale: str,
) -> ActiveMedListReport:
    seen: set[str] = set()
    distinct: list[str] = []
    for m in resolved:
        key = m.name.lower()
        if key in seen:
            continue
        seen.add(key)
        distinct.append(m.name)
    return ActiveMedListReport(
        medications=resolved,
        distinct_names=distinct,
        n_medication_request=len(med_requests),
        n_medication_administration=len(med_admins),
        n_documents_scanned=len(documents),
        n_documents_with_med_section=n_with_meds,
        method=method,
        rationale=rationale,
    )


def register(mcp) -> None:
    mcp.tool()(compute_resolve_active_meds)
