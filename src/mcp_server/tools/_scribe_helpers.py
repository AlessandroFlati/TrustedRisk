"""Shared FHIR extraction + LLM-polish helpers for Phase 2.2 scribe tools.

The four scribe tools (progress_note, discharge_summary, consult_letter,
admission_hnp) all consume the same kinds of structured FHIR resources
and run them through a deterministic-template renderer. This module
factors out the common extraction + formatting logic so each tool's
public function stays readable.

LLM polish (`TRUSTEDRISK_SCRIBE_LLM_POLISH=1`) is paraphrase-only -- the
section bodies' clinical claims must trace to the structured input
even after polish.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable

from shared.schemas import ClinicalNoteSection


def llm_polish_enabled() -> bool:
    return os.environ.get("TRUSTEDRISK_SCRIBE_LLM_POLISH", "0").lower() in (
        "1", "true", "yes", "on",
    )


def evidence_id_for(resource: dict | None, fallback_kind: str = "evidence") -> str:
    if not resource:
        return f"chart:{fallback_kind}"
    return resource.get("id") or f"chart:{fallback_kind}:{abs(hash(repr(resource))) % 100_000}"


# ─────────────────────────────────────────────────────────────────────
# FHIR Bundle extractors
# ─────────────────────────────────────────────────────────────────────

def _entries(bundle: dict | None) -> list[dict]:
    if not isinstance(bundle, dict):
        return []
    return bundle.get("entry", []) or []


def _resource_id_from_entry(entry: dict | None) -> str | None:
    """Recover a canonical resource id from a transaction-bundle entry.

    FHIR transaction Bundles use POST + `urn:uuid:<uuid>` fullUrls, so a
    resource ingested directly from the wire may have an empty `id`
    field until the server commits the transaction. The fullUrl trailing
    UUID is the id every reasonable server assigns -- using it as a
    fallback keeps the offline / pre-commit ingest path consistent with
    a server-mediated ingest.
    """
    if not isinstance(entry, dict):
        return None
    full = entry.get("fullUrl")
    if isinstance(full, str) and ":" in full:
        return full.rsplit(":", 1)[-1] or None
    return None


def extract_patient_summary(
    bundle: dict | None,
) -> tuple[str, list[str], bool]:
    """Return (summary_text, cited_resource_ids, is_complete).

    ``is_complete`` is True only when a Patient resource is present in
    the bundle AND name, birthDate, and gender are all non-empty. Callers
    that require complete patient demographics must check ``is_complete``
    and abstain at the tool level when it is False.

    Returns ("", [], False) when no Patient resource is found.
    Returns (summary_text, cited, False) when a Patient resource is found
    but one or more critical demographic fields are absent.
    Returns (summary_text, cited, True) when all critical fields are present.
    """
    cited: list[str] = []
    patient = next(
        (e["resource"] for e in _entries(bundle)
            if e.get("resource", {}).get("resourceType") == "Patient"),
        None,
    )
    if patient is None:
        return "", cited, False

    cited.append(patient.get("id", "Patient"))

    name_block = (patient.get("name") or [{}])[0]
    given = " ".join(name_block.get("given", []) or [])
    family = name_block.get("family", "")
    full_name = f"{given} {family}".strip()

    bd = patient.get("birthDate")
    gender = patient.get("gender")

    is_complete = bool(full_name and bd and gender)

    age_str = ""
    if bd:
        try:
            born = datetime.fromisoformat(bd)
            age_str = str(
                (datetime.now(timezone.utc).date() - born.date()).days // 365
            )
        except ValueError:
            pass

    parts: list[str] = []
    if full_name:
        parts.append(full_name)
    if gender:
        parts.append(gender)
    if age_str:
        parts.append(f"age {age_str}")
    if bd:
        parts.append(f"DOB {bd}")
    summary = (", ".join(parts) + ".") if parts else ""

    return summary, cited, is_complete


def extract_active_problems(bundle: dict | None) -> tuple[list[str], list[str]]:
    """Return (bullets_with_cite, cited_resource_ids)."""
    cited: list[str] = []
    bullets: list[str] = []
    for entry in _entries(bundle):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        rid = resource.get("id", "Condition")
        cited.append(rid)
        code_block = resource.get("code", {})
        coding = (code_block.get("coding") or [{}])[0]
        text = (
            code_block.get("text")
            or coding.get("display")
            or coding.get("code")
            or "unspecified condition"
        )
        bullets.append(f"  - {text} (cite {rid})")
    return bullets, cited


def extract_recent_observations(
    bundle: dict | None, *, max_items: int = 8,
) -> tuple[list[str], list[str]]:
    cited: list[str] = []
    items: list[tuple[datetime, str, str]] = []
    for entry in _entries(bundle):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue
        rid = resource.get("id", "Observation")
        eff = (
            resource.get("effectiveDateTime")
            or resource.get("effectiveInstant")
            or resource.get("issued")
        )
        try:
            eff_dt = datetime.fromisoformat(eff.replace("Z", "+00:00")) if eff else None
        except (AttributeError, ValueError):
            eff_dt = None
        sort_key = eff_dt or datetime.fromtimestamp(0, tz=timezone.utc)
        coding = ((resource.get("code") or {}).get("coding") or [{}])[0]
        label = (
            coding.get("display")
            or (resource.get("code") or {}).get("text")
            or "observation"
        )
        value_text = (
            (resource.get("valueQuantity") or {}).get("value")
            or resource.get("valueString")
            or "(unstructured)"
        )
        unit = (resource.get("valueQuantity") or {}).get("unit", "")
        bullet = (
            f"  - {label}: {value_text} {unit} "
            f"on {eff or '[no date]'} (cite {rid})".rstrip()
        )
        items.append((sort_key, rid, bullet))
    items.sort(reverse=True)  # most recent first
    bullets: list[str] = []
    for _, rid, bullet in items[:max_items]:
        cited.append(rid)
        bullets.append(bullet)
    return bullets, cited


def extract_medications(bundle: dict | None) -> tuple[list[str], list[str]]:
    cited: list[str] = []
    bullets: list[str] = []
    for entry in _entries(bundle):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "MedicationRequest":
            continue
        rid = resource.get("id", "MedicationRequest")
        med_block = (
            resource.get("medicationCodeableConcept")
            or resource.get("medicationReference")
            or {}
        )
        text = (
            med_block.get("text")
            or (med_block.get("coding") or [{}])[0].get("display")
            or "unknown medication"
        )
        cited.append(rid)
        bullets.append(f"  - {text} (cite {rid})")
    return bullets, cited


def extract_allergies(bundle: dict | None) -> tuple[list[str], list[str]]:
    cited: list[str] = []
    bullets: list[str] = []
    for entry in _entries(bundle):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "AllergyIntolerance":
            continue
        rid = resource.get("id", "AllergyIntolerance")
        text = (
            (resource.get("code") or {}).get("text")
            or ((resource.get("code") or {}).get("coding") or [{}])[0].get("display")
            or "unspecified allergy"
        )
        cited.append(rid)
        bullets.append(f"  - {text} (cite {rid})")
    return bullets, cited


def extract_procedures(bundle: dict | None) -> tuple[list[str], list[str]]:
    cited: list[str] = []
    bullets: list[str] = []
    for entry in _entries(bundle):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Procedure":
            continue
        rid = resource.get("id", "Procedure")
        text = (
            (resource.get("code") or {}).get("text")
            or ((resource.get("code") or {}).get("coding") or [{}])[0].get("display")
            or "unspecified procedure"
        )
        when = resource.get("performedDateTime") or "date not recorded"
        cited.append(rid)
        bullets.append(f"  - {text} on {when} (cite {rid})")
    return bullets, cited


def extract_encounter(bundle: dict | None) -> tuple[str | None, str | None]:
    """Return (encounter_id, summary_text) for the most relevant Encounter.

    Selection priority is: (1) most recent inpatient `class=IMP` Encounter
    by `period.start`, (2) failing that, the most recent Encounter of any
    class. Naively picking the first Encounter in the bundle returns
    whatever the FHIR server enumerated first -- typically a decade-old
    visit on Synthea fixtures -- and produces a discharge note dated in
    the wrong year. Anchoring on the most recent inpatient stay is the
    documented contract for discharge_summary / progress_note / H&P.

    Resource ID resolution: an Encounter ingested from a FHIR transaction
    Bundle (POST + urn:uuid fullUrl) may not carry `resource.id` until
    after the server commits the transaction. As a fallback we read the
    entry's `fullUrl` and treat the trailing UUID as the encounter id,
    which is what every server uses to assign the canonical id anyway.
    """
    bundle_entries = _entries(bundle)
    encounters: list[tuple[dict, dict]] = [
        (entry, entry.get("resource", {}))
        for entry in bundle_entries
        if entry.get("resource", {}).get("resourceType") == "Encounter"
    ]
    if not encounters:
        return None, None

    def _start_key(item: tuple[dict, dict]) -> str:
        enc = item[1]
        # ISO-8601 strings sort lexicographically.
        return ((enc.get("period") or {}).get("start") or "")

    imp = [(e, r) for e, r in encounters
              if (r.get("class") or {}).get("code") == "IMP"]
    chosen_entry, chosen = (
        max(imp, key=_start_key) if imp else max(encounters, key=_start_key)
    )

    rid = chosen.get("id") or _resource_id_from_entry(chosen_entry)

    # An Encounter without an id cannot be used as a cite-back.
    # Return (None, None) so callers can abstain rather than rendering
    # placeholder text in the note body.
    if not rid:
        return None, None

    cls = ((chosen.get("class") or {}).get("display")
              or (chosen.get("class") or {}).get("code"))
    period = chosen.get("period") or {}
    start = period.get("start")
    end = period.get("end")

    # When period.start is absent we return (rid, None). Callers that
    # need the summary text (discharge_summary, progress_note) treat
    # None summary as "encounter present but dates not recorded" -- they
    # do NOT render a placeholder string.
    if not start:
        return rid, None

    end_str = end if end else "ongoing"
    return rid, f"Encounter {rid} (class {cls}, {start} to {end_str})."


# ─────────────────────────────────────────────────────────────────────
# Section helpers + LLM polish
# ─────────────────────────────────────────────────────────────────────

def render_bullets_section(
    section_id: str,
    title: str,
    bullets: list[str],
    cited: list[str],
    empty_text: str,
) -> ClinicalNoteSection:
    body = "\n".join(bullets) if bullets else empty_text
    return ClinicalNoteSection(
        section_id=section_id, title=title, body=body,
        cited_evidence_ids=cited,
    )


def estimate_reading_minutes(full_text: str, *, wpm: float = 220.0) -> float:
    """Approximate reading time in minutes at average prose pace."""
    words = max(1, len(full_text.split()))
    return round(words / wpm, 2)


def coverage_pct(
    sections: list[ClinicalNoteSection],
    bundle: dict | None,
    extra_evidence_ids: Iterable[str] = (),
) -> float:
    """Fraction of available evidence items referenced at least once."""
    referenced: set[str] = set()
    for s in sections:
        referenced.update(s.cited_evidence_ids)
    referenced.update(extra_evidence_ids)
    available: set[str] = set()
    for entry in _entries(bundle):
        resource = entry.get("resource", {})
        rid = resource.get("id")
        if rid:
            available.add(rid)
    if not available:
        return 0.0
    return round(len(referenced & available) / len(available), 3)


async def llm_polish_sections(
    sections: list[ClinicalNoteSection],
) -> tuple[list[ClinicalNoteSection], str | None]:
    """Optional LLM paraphrase pass (async -- must be awaited).

    For each section, calls the polish client with the section body +
    a paraphrase-only system prompt. The polish client's hallucination
    post-check rejects any output that drops cite-back IDs / drug
    names / ICD codes / dose values -- when rejected, the section keeps
    the deterministic body unchanged.

    Returns (polished_sections, model_id) where `model_id` is the
    aggregate id of the client used. None when no LLM is configured.
    """
    try:
        from a2a_agent.llm_polish import resolve_polish_client
    except Exception:
        return sections, None
    client = resolve_polish_client()
    if client.model_id is None:
        return sections, None

    system_prompt = (
        "You are a clinical-documentation editor. Paraphrase the text "
        "below to improve flow + readability while preserving every "
        "clinical fact, cite-back parenthetical (e.g. `(cite obs-cr)`), "
        "ICD-10 code (e.g. `I50.21`), drug name, dose, and timestamp "
        "EXACTLY. Do not invent any new clinical content. Do not "
        "remove or rewrite cite-backs. Output the polished text only "
        "-- no headers, no commentary."
    )

    polished: list[ClinicalNoteSection] = []
    for s in sections:
        res = await client.polish(s.body, system_prompt=system_prompt)
        if res.is_polished:
            polished.append(ClinicalNoteSection(
                section_id=s.section_id, title=s.title,
                body=res.polished_text,
                cited_evidence_ids=list(s.cited_evidence_ids),
                is_llm_polished=True,
            ))
        else:
            polished.append(s)
    return polished, client.model_id


# ─────────────────────────────────────────────────────────────────────
# SOAP section extraction from DocumentReference clinical-note bodies.
#
# Bypasses the dictation flow: when the chat-side never speaks the
# subjective entry but the EHR contains a DocumentReference whose
# clinical-note plaintext already has the SOAP sections labelled, we
# can recover the narrative from the bundle. Strict regex over a
# whitelist of section headers; no fabrication when a header is missing.
# ─────────────────────────────────────────────────────────────────────


# Section name -> alternative header tokens that mark the start of the
# section in a typed/dictated clinical note. Keys are the canonical
# section names the scribe tools consume.
_SOAP_SECTION_HEADERS: dict[str, tuple[str, ...]] = {
    "subjective":  ("SUBJECTIVE", "S:"),
    "objective":   ("OBJECTIVE", "O:"),
    "assessment":  ("ASSESSMENT AND PLAN", "ASSESSMENT", "A&P", "A AND P",
                     "IMPRESSION", "A:"),
    "plan":        ("PLAN", "P:", "RECOMMENDATIONS"),
    "hpi":         ("HISTORY OF PRESENT ILLNESS", "HPI", "PRESENT ILLNESS"),
    "pmh":         ("PAST MEDICAL HISTORY", "PMH", "MEDICAL HISTORY"),
    "psh":         ("PAST SURGICAL HISTORY", "PSH", "SURGICAL HISTORY"),
    "fh":          ("FAMILY HISTORY", "FH"),
    "sh":          ("SOCIAL HISTORY", "SH"),
    "ros":         ("REVIEW OF SYSTEMS", "ROS"),
    "pe":          ("PHYSICAL EXAMINATION", "PHYSICAL EXAM", "PE",
                     "EXAMINATION"),
    "labs":        ("LABS", "LABORATORY", "LAB DATA",
                     "LABORATORY DATA"),
    "imaging":     ("IMAGING", "RADIOLOGY"),
    "consultation_reason": (
        "REASON FOR CONSULTATION", "CONSULTATION REASON",
        "REASON FOR CONSULT", "REQUEST",
    ),
    "hospital_course":     ("HOSPITAL COURSE", "COURSE OF HOSPITALIZATION",
                                "COURSE IN HOSPITAL"),
    "discharge_disposition": ("DISCHARGE DISPOSITION", "DISPOSITION"),
    "patient_instructions":  ("PATIENT INSTRUCTIONS", "INSTRUCTIONS",
                                  "DISCHARGE INSTRUCTIONS"),
    "followup_plan":        ("FOLLOW-UP PLAN", "FOLLOWUP PLAN",
                                "FOLLOW UP", "FOLLOWUP",
                                "FOLLOW-UP", "FOLLOW-UP APPOINTMENTS"),
}


def _decode_attachment_body(content: dict) -> str:
    """Decode an attachment.data b64 to plaintext. Returns "" on any failure."""
    import base64
    att = content.get("attachment") or {}
    data = att.get("data")
    if not isinstance(data, str):
        return ""
    try:
        return base64.b64decode(data).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _iter_clinical_note_bodies(bundle: dict | None):
    """Yield each (DocumentReference, plaintext_body) from the bundle.

    Includes any DocumentReference whose attachment decodes; relies on
    the scribe pipeline downstream to score relevance via section
    coverage rather than filtering by category here.
    """
    if not isinstance(bundle, dict):
        return
    for entry in bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "DocumentReference":
            continue
        for content in r.get("content") or []:
            body = _decode_attachment_body(content) if isinstance(content, dict) else ""
            if body:
                yield r, body


def extract_soap_section(
    bundle: dict | None, section_name: str,
) -> str:
    """Pull the named SOAP section from the most relevant DocumentReference.

    Args:
        bundle: FHIR R4 Bundle.
        section_name: one of the keys of `_SOAP_SECTION_HEADERS`. Case
            insensitive.

    Returns:
        The plaintext body of the matched section, stripped of leading /
        trailing whitespace. Empty string when the bundle contains no
        DocumentReference whose plaintext has the requested header.

    The matcher walks every DocumentReference body, finds the first
    instance of any header alias in `_SOAP_SECTION_HEADERS[section_name]`,
    and returns the substring up to the next recognised SOAP header (or
    end of body). No fabrication: when no header matches, the empty
    string surfaces and the scribe tool's existing abstain path runs.
    """
    import re
    section_key = section_name.lower()
    aliases = _SOAP_SECTION_HEADERS.get(section_key)
    if not aliases:
        return ""
    # Build a regex that matches any alias as a header line. Header may
    # appear at start of body OR after a blank line; it ends with
    # optional ":" and whitespace before the actual content.
    alias_pattern = "|".join(
        re.escape(a) for a in aliases
    )
    # Header form: alias must be followed by a colon (with optional
    # whitespace) or by a newline; this avoids matching "PE" as the
    # start of "Penicillin" or "FH" inside "of his". Two-letter aliases
    # are particularly risky without this anchor.
    header_re = re.compile(
        rf"(?:^|\n)[ \t]*({alias_pattern})[ \t]*(?::[ \t]*|\n)",
        re.IGNORECASE,
    )
    # Set of all SOAP headers -- used to detect where the matched
    # section ends. Same anchor logic as the start regex.
    all_aliases: list[str] = []
    for aliases_set in _SOAP_SECTION_HEADERS.values():
        all_aliases.extend(aliases_set)
    stop_re = re.compile(
        r"\n[ \t]*(" + "|".join(re.escape(a) for a in all_aliases)
        + r")[ \t]*(?::[ \t]*|\n)",
        re.IGNORECASE,
    )

    best: tuple[int, str] = (-1, "")
    for r, body in _iter_clinical_note_bodies(bundle):
        m = header_re.search(body)
        if not m:
            continue
        tail = body[m.end():]
        stop_m = stop_re.search(tail)
        if stop_m:
            tail = tail[: stop_m.start()]
        tail = tail.strip()
        if not tail:
            continue
        # Score: prefer DocumentReference whose category is clinical-note
        # (PO uploads them as such); fall back to first match.
        score = 0
        for cat in r.get("category") or []:
            for c in (cat or {}).get("coding") or []:
                if (c.get("code") or "").lower() in (
                    "clinical-note", "progress-note", "consultation-note",
                    "discharge-summary", "history-and-physical",
                ):
                    score += 5
        if score > best[0]:
            best = (score, tail)
    return best[1]


def has_clinical_note(bundle: dict | None) -> bool:
    """True when the bundle has at least one DocumentReference body."""
    for _r, body in _iter_clinical_note_bodies(bundle):
        if body.strip():
            return True
    return False


def extract_full_chart_text(bundle: dict | None, max_chars: int = 12000) -> str:
    """Concat all DocumentReference plaintexts (up to max_chars).

    Used by `compute_icd10_suggest` / `compute_cpt_suggest` /
    `compute_coding_audit` when no `chart_text` is supplied. The cap
    keeps tokenisation costs bounded; the chart already drives the
    structured extraction in scribe tools.
    """
    parts: list[str] = []
    total = 0
    for _r, body in _iter_clinical_note_bodies(bundle):
        clean = body.strip()
        if not clean:
            continue
        if total + len(clean) > max_chars:
            parts.append(clean[: max_chars - total])
            break
        parts.append(clean)
        total += len(clean) + 2
    return "\n\n".join(parts)
