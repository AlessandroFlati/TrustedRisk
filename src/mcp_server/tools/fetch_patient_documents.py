"""healthcare.compute_fetch_patient_documents -- expose chart text to the LLM.

Many TrustedRisk tools (`compute_clinical_ner`, `compute_icd10_suggest`,
`compute_progress_note_draft`, ...) take free-text clinical notes as
input. On a real FHIR server the calling agent typically has its own
DocumentReference fetch path. On Prompt Opinion the OAuth scope grants
`patient/DocumentReference.rs` but the BYO LLM has no built-in fetch
primitive -- so without an MCP tool that surfaces note text, the LLM
has nothing to feed into the NER chain even though the data is there.

This tool is the bridge. It pulls the most recent N DocumentReference
resources for the SHARP-bound patient, decodes their attachments, and
returns one record per note with title, date, and plaintext body.
"""

from __future__ import annotations

import base64
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


class PatientDocument(BaseModel):
    """One clinical note pulled from a DocumentReference resource."""

    document_id: str | None = None
    title: str | None = None
    document_type: str | None = None
    date: str | None = None
    content_type: str | None = None
    body: str
    body_length: int


class PatientDocumentsReport(BaseModel):
    """Most-recent-first list of clinical notes for the active patient."""

    documents: list[PatientDocument] = Field(default_factory=list)
    n_total: int = 0
    n_returned: int = 0
    method: Literal["document_reference", "no_documents"]
    rationale: str


def _decode_document(doc: dict[str, Any]) -> tuple[str, str | None]:
    """Return (body_text, content_type) from a FHIR DocumentReference."""
    chunks: list[str] = []
    content_type: str | None = None
    for content in doc.get("content") or []:
        attachment = content.get("attachment") or {}
        ctype = attachment.get("contentType")
        if ctype and not content_type:
            content_type = str(ctype)
        data = attachment.get("data")
        if data:
            try:
                chunks.append(base64.b64decode(data).decode("utf-8", "ignore"))
            except Exception:
                pass
    return "\n".join(chunks), content_type


def _doc_date(doc: dict[str, Any]) -> str | None:
    return (
        doc.get("date")
        or doc.get("indexed")
        or (doc.get("context", {}) or {}).get("period", {}).get("start")
    )


def _doc_title(doc: dict[str, Any]) -> str | None:
    for content in doc.get("content") or []:
        attachment = content.get("attachment") or {}
        if attachment.get("title"):
            return str(attachment["title"])
    if doc.get("description"):
        return str(doc["description"])
    return None


def _doc_type(doc: dict[str, Any]) -> str | None:
    type_obj = doc.get("type") or {}
    if isinstance(type_obj, dict):
        if type_obj.get("text"):
            return str(type_obj["text"])
        for c in type_obj.get("coding") or []:
            if c.get("display"):
                return str(c["display"])
    return None


async def compute_fetch_patient_documents(
    patient_id: str | None = None,
    max_documents: int = 5,
    keyword_filter: str | None = None,
) -> PatientDocumentsReport:
    """Fetch DocumentReference clinical notes for the active patient.

    The patient is taken from the SHARP-on-MCP X-Patient-ID header, NOT
    from the chat prompt. Leave `patient_id` null in every normal call.

    Args:
        patient_id: LEAVE NULL / OMIT for normal use. The tool reads the
            patient from the X-Patient-ID header on the MCP request. This
            argument exists ONLY as an explicit override for advanced
            multi-patient orchestration; passing a free-text label from
            the prompt (e.g. "Marcus") will fail because real FHIR
            servers index Patient by their canonical resource id (usually
            a UUID), not by display name.
        max_documents: cap on returned documents (default 5, most recent
            first). Keep low to fit the LLM context budget.
        keyword_filter: optional case-insensitive substring; only notes
            whose body contains it are returned. Useful when looking
            specifically for "discharge", "admission", "consultation",
            "operative", etc.

    Returns:
        PatientDocumentsReport with one record per note. Use the `body`
        field as input to `compute_clinical_ner`,
        `compute_icd10_suggest`, `compute_progress_note_draft`, or any
        other free-text tool.
    """
    if max_documents <= 0:
        max_documents = 1
    if max_documents > 25:
        max_documents = 25

    pid = await resolve_patient_id(patient_id)
    bundle = await fetch_patient_bundle(pid)
    entries = bundle.get("entry") or []

    raw_docs: list[dict[str, Any]] = []
    for e in entries:
        res = e.get("resource") if isinstance(e, dict) else None
        if isinstance(res, dict) and res.get("resourceType") == "DocumentReference":
            raw_docs.append(res)

    raw_docs.sort(key=lambda d: (_doc_date(d) or ""), reverse=True)

    keyword_l = (keyword_filter or "").strip().lower() or None
    fallback_used = False

    def _collect(filter_str: str | None) -> list[PatientDocument]:
        out: list[PatientDocument] = []
        for doc in raw_docs:
            body, ctype = _decode_document(doc)
            if not body.strip():
                continue
            if filter_str and filter_str not in body.lower():
                continue
            out.append(PatientDocument(
                document_id=doc.get("id"),
                title=_doc_title(doc),
                document_type=_doc_type(doc),
                date=_doc_date(doc),
                content_type=ctype,
                body=body,
                body_length=len(body),
            ))
            if len(out) >= max_documents:
                break
        return out

    decoded = _collect(keyword_l)

    # Server-side fallback: if a keyword filter excluded every document
    # but the patient does have decodable notes, retry without the
    # filter. Synthea-style PO notes don't carry an explicit
    # "discharge"/"admission" word; the LLM should never have to know
    # to retry by hand.
    if not decoded and keyword_l is not None and raw_docs:
        decoded = _collect(None)
        if decoded:
            fallback_used = True

    if not decoded:
        return PatientDocumentsReport(
            documents=[],
            n_total=len(raw_docs),
            n_returned=0,
            method="no_documents",
            rationale=(
                f"Found {len(raw_docs)} DocumentReference resources but "
                f"none had decodable plaintext content."
            ),
        )

    rationale = (
        f"Returned the {len(decoded)} most recent of "
        f"{len(raw_docs)} DocumentReference resources"
    )
    if fallback_used:
        rationale += (
            f" (keyword_filter '{keyword_filter}' matched 0 notes; "
            f"fell back to most-recent unfiltered)"
        )
    elif keyword_filter:
        rationale += f" matching keyword '{keyword_filter}'"
    rationale += "."

    return PatientDocumentsReport(
        documents=decoded,
        n_total=len(raw_docs),
        n_returned=len(decoded),
        method="document_reference",
        rationale=rationale,
    )


def register(mcp) -> None:
    mcp.tool()(compute_fetch_patient_documents)
