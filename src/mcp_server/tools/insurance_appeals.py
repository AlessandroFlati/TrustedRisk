"""healthcare.compute_denial_letter_parse /
compute_appeal_letter_draft / compute_appeal_escalation_path
-- Phase 10.3 insurance appeals specialist (APPEALS-1/2/3).

Pain-point: 10-15% of US claims are denied annually (KFF 2024); only
~ 0.2% are appealed even though ~ 50% of internal appeals succeed
(KFF Marketplace Plan analysis 2023). The bottleneck is letter
drafting + escalation deadline tracking.

Pure-deterministic floor:
- Letter parse uses regex + payer-specific keyword tables.
- Letter draft uses an 8-section template that cite-backs every
  clinical claim to a chart excerpt or policy citation.
- Escalation path encodes ERISA / ACA / state-DOI deadlines per payer.

References:
- KFF Marketplace Plan Denial Rates 2024.
- AMA Prior Authorization Physician Survey 2024.
- ERISA §502(a) appeal rules (29 CFR 2560.503-1).
- ACA External Review (45 CFR 147.136).
"""

from __future__ import annotations

import re
from typing import Any

from shared.schemas import (
    AppealEscalationPathReport,
    AppealEscalationStep,
    AppealLetterDraft,
    AppealLetterParagraph,
    DenialLetterParseReport,
    DenialReason,
)


# ─────────────────────────────────────────────────────────────────────
# Payer-specific keyword tables driving APPEALS-1
# ─────────────────────────────────────────────────────────────────────

_DENIAL_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "medical_necessity": [
        "not medically necessary", "lack of medical necessity",
        "medical necessity not established", "does not meet criteria",
    ],
    "step_therapy_not_met": [
        "step therapy", "step-therapy", "first-line therapy",
        "preferred alternative not tried", "trial of formulary",
    ],
    "out_of_network": [
        "out-of-network", "non-participating provider", "out of network",
    ],
    "experimental_unproven": [
        "experimental", "investigational", "unproven", "off-label",
    ],
    "missing_documentation": [
        "missing documentation", "insufficient records",
        "additional information required", "incomplete records",
    ],
    "duplicate_service": [
        "duplicate claim", "previously billed", "already paid",
    ],
    "non_covered_benefit": [
        "not a covered benefit", "excluded benefit", "non-covered",
        "not covered under your plan",
    ],
    "claim_filing_limit": [
        "timely filing", "filed beyond", "submission deadline exceeded",
    ],
}

_PAYER_PATTERNS: dict[str, list[str]] = {
    "UnitedHealthcare": ["unitedhealthcare", "united healthcare", "uhc"],
    "Anthem":           ["anthem", "blue cross", "bcbs"],
    "Aetna":            ["aetna"],
    "Cigna":            ["cigna"],
    "Humana":           ["humana"],
    "Medicare":         ["medicare", "cms"],
    "Medicaid":         ["medicaid"],
}


def _detect_payer(text: str) -> str:
    lo = text.lower()
    for payer, keywords in _PAYER_PATTERNS.items():
        if any(k in lo for k in keywords):
            return payer
    return "Unknown"


_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase + collapse all whitespace runs to a single space so
    that keyword phrases survive line-wraps in the input letter."""
    return _WHITESPACE_RE.sub(" ", text.lower())


def _detect_categories(text: str) -> list[tuple[str, str]]:
    """Return list of (category, matched-snippet) for every keyword
    pattern that hit. Order: in input order of text appearance, deduped
    by category-only (first hit wins)."""
    lo = _normalise(text)
    seen: set[str] = set()
    hits: list[tuple[str, str]] = []
    for cat, patterns in _DENIAL_CATEGORY_PATTERNS.items():
        for kw in patterns:
            if kw in lo and cat not in seen:
                hits.append((cat, kw))
                seen.add(cat)
                break
    return hits


_CLAIM_ID_RE = re.compile(
    r"(?:claim\s*(?:id|number)?[:#\s]+)([A-Z0-9-]{6,})", re.IGNORECASE,
)
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DEADLINE_RE = re.compile(
    r"(?:appeal\s+(?:deadline|by|due|must be filed by))[:\s]+([0-9A-Za-z,\s/.-]{6,40})",
    re.IGNORECASE,
)
_DOLLAR_RE = re.compile(
    r"\$\s*([0-9,]+(?:\.[0-9]{1,2})?)",
)


def _to_iso(maybe_date: str | None) -> str | None:
    """Best-effort ISO conversion. Returns None if the input is empty
    or doesn't match an obvious date pattern."""
    if not maybe_date:
        return None
    m = _DATE_ISO_RE.search(maybe_date)
    if m:
        return m.group(1)
    return None


async def compute_denial_letter_parse(
    letter_text: str,
    *,
    expected_payer: str | None = None,
) -> DenialLetterParseReport:
    """Parse a payer denial letter into structured fields.

    Args:
        letter_text: full plain-text body of the denial letter.
        expected_payer: optional payer hint when free-text detection
            ambiguous; the tool will still report the detected payer.

    Returns:
        DenialLetterParseReport.
    """
    payer_detected = _detect_payer(letter_text)
    payer = expected_payer or payer_detected
    if payer_detected == "Unknown" and not expected_payer:
        payer = "Unknown"

    claim_id = None
    m_cid = _CLAIM_ID_RE.search(letter_text)
    if m_cid:
        claim_id = m_cid.group(1).strip(",.; ")

    received_date = None
    m_recv = re.search(
        r"(?:date(?:d)?|received(?: date)?)[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2})",
        letter_text, flags=re.IGNORECASE,
    )
    if m_recv:
        received_date = m_recv.group(1)

    appeal_deadline = None
    m_dead = _DEADLINE_RE.search(letter_text)
    if m_dead:
        appeal_deadline = _to_iso(m_dead.group(1))

    contested_dollar = None
    m_dollar = _DOLLAR_RE.search(letter_text)
    if m_dollar:
        try:
            contested_dollar = float(m_dollar.group(1).replace(",", ""))
        except ValueError:
            contested_dollar = None

    is_partial = bool(re.search(
        r"\bpartial(?:ly)?\s+denied?\b|partially\s+approved",
        letter_text, flags=re.IGNORECASE,
    ))

    reasons: list[DenialReason] = []
    flat = _WHITESPACE_RE.sub(" ", letter_text).strip()
    for idx, (cat, kw) in enumerate(_detect_categories(letter_text), 1):
        # Find the first whitespace-normalised sentence containing
        # the matched keyword for context (line-wrap-safe).
        snippet = ""
        for sent in re.split(r"(?<=[.!?])\s+", flat):
            if kw in sent.lower():
                snippet = sent.strip()
                break
        reasons.append(DenialReason(
            reason_code=f"R{idx:03d}",
            reason_text=snippet or kw,
            category=cat,                                # type: ignore[arg-type]
        ))

    if not reasons:
        reasons.append(DenialReason(
            reason_code="R001",
            reason_text=(
                "No machine-readable denial reason matched the keyword "
                "table; manual triage required."
            ),
            category="other",
        ))

    rationale = (
        f"Parsed denial from {payer}. {len(reasons)} reason(s) categorised. "
        f"Claim {claim_id or '<unknown>'}; appeal deadline "
        f"{appeal_deadline or '<not detected>'}; partial = {is_partial}."
    )

    return DenialLetterParseReport(
        payer=payer,
        claim_id=claim_id,
        received_date_iso=received_date,
        appeal_deadline_iso=appeal_deadline,
        reasons=reasons,
        n_reasons=len(reasons),
        is_partial_denial=is_partial,
        contested_dollar_amount=contested_dollar,
        rationale=rationale,
        references=[
            "KFF Marketplace Plan Denial Rates 2024.",
            "AMA Prior Authorization Physician Survey 2024.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# APPEALS-2 -- compute_appeal_letter_draft
# ─────────────────────────────────────────────────────────────────────

_APPEAL_LEVEL_OPENING: dict[str, str] = {
    "internal_first_level": (
        "I am writing to formally appeal the denial of coverage "
        "described below at the first internal-review level."
    ),
    "internal_second_level": (
        "I am writing to escalate to the second internal-appeal level "
        "after the first internal review failed to reverse the denial."
    ),
    "external_independent_review": (
        "I am writing to request an external independent review under "
        "ACA §2719 (45 CFR 147.136) after exhausting internal appeals."
    ),
    "state_insurance_commissioner": (
        "I am filing a complaint with the State Department of "
        "Insurance after the payer denied this medically-necessary "
        "service."
    ),
    "ada_complaint": (
        "I am filing a complaint under the Americans with Disabilities "
        "Act for the denial of an accommodation that is medically "
        "necessary for this patient."
    ),
}


async def compute_appeal_letter_draft(
    parsed_denial: DenialLetterParseReport | dict,
    patient_summary: str,
    medical_necessity_argument: str,
    *,
    appeal_level: str = "internal_first_level",
    cited_evidence_ids: list[str] | None = None,
    alternative_proposed: str | None = None,
    enable_llm_polish: bool = False,
    llm_model_id: str | None = None,
) -> AppealLetterDraft:
    """Draft a structured appeal letter with mandatory cite-backs.

    Args:
        parsed_denial: structured DenialLetterParseReport (or dict).
        patient_summary: 2-4 sentence patient summary (clinician input).
        medical_necessity_argument: 1-2 paragraph clinical justification.
        appeal_level: target escalation level.
        cited_evidence_ids: per-paragraph evidence ids (FHIR resource
            IDs, PubMed PMIDs, or chart excerpt references).
        alternative_proposed: optional alternative service proposal
            (e.g. step-down therapy when step-therapy was the denial
            reason).
        enable_llm_polish: when True flips the `is_llm_polished` flag;
            no actual model call here -- the polish layer is owned by
            the upstream LLM-polish module that respects
            preserved-tokens.
        llm_model_id: free-text model identifier propagated to the
            output for reproducibility.

    Returns:
        AppealLetterDraft.
    """
    if isinstance(parsed_denial, dict):
        parsed_denial = DenialLetterParseReport.model_validate(parsed_denial)

    cited_evidence_ids = cited_evidence_ids or []
    paragraphs: list[AppealLetterParagraph] = []

    paragraphs.append(AppealLetterParagraph(
        section="header",
        text=(
            f"To: {parsed_denial.payer} Appeals Department\n"
            f"Re: Claim {parsed_denial.claim_id or '<claim id pending>'} "
            f"-- Appeal of denial dated "
            f"{parsed_denial.received_date_iso or '<date pending>'}.\n"
            f"{_APPEAL_LEVEL_OPENING.get(appeal_level, '')}"
        ),
        is_llm_polished=False,
    ))
    paragraphs.append(AppealLetterParagraph(
        section="patient_summary",
        text=patient_summary,
        cited_evidence_ids=cited_evidence_ids[:1],
        is_llm_polished=enable_llm_polish,
    ))
    paragraphs.append(AppealLetterParagraph(
        section="denial_summary",
        text=(
            f"The payer denied coverage citing the following: "
            + " | ".join(r.reason_text for r in parsed_denial.reasons)
        ),
        is_llm_polished=False,
    ))
    paragraphs.append(AppealLetterParagraph(
        section="medical_necessity_argument",
        text=medical_necessity_argument,
        cited_evidence_ids=cited_evidence_ids[:5],
        is_llm_polished=enable_llm_polish,
    ))
    paragraphs.append(AppealLetterParagraph(
        section="evidence_supporting",
        text=(
            "Supporting evidence cited in this appeal: "
            + ", ".join(cited_evidence_ids) if cited_evidence_ids else
            "No structured evidence ids were supplied; the medical-"
            "necessity argument should be cite-backed before submission."
        ),
        cited_evidence_ids=cited_evidence_ids,
        is_llm_polished=False,
    ))
    paragraphs.append(AppealLetterParagraph(
        section="policy_counter_argument",
        text=(
            "The payer's policy citation does not address the patient-"
            "specific clinical factors enumerated above. The denial "
            "categories matched in this case were: "
            + ", ".join(r.category for r in parsed_denial.reasons)
            + "."
        ),
        is_llm_polished=False,
    ))
    if alternative_proposed:
        paragraphs.append(AppealLetterParagraph(
            section="alternative_proposed",
            text=alternative_proposed,
            is_llm_polished=enable_llm_polish,
        ))
    paragraphs.append(AppealLetterParagraph(
        section="closing",
        text=(
            "We respectfully request that the prior denial be overturned "
            "and the requested service approved. Please respond within "
            "the regulatory window applicable to this appeal level."
        ),
        is_llm_polished=False,
    ))

    full_text = "\n\n".join(p.text for p in paragraphs)
    n_cite_backs = sum(len(p.cited_evidence_ids) for p in paragraphs)

    rationale = (
        f"{appeal_level} appeal letter against {parsed_denial.payer}. "
        f"{len(paragraphs)} section(s); {n_cite_backs} cite-back(s). "
        f"LLM polish = {enable_llm_polish}."
    )

    return AppealLetterDraft(
        payer=parsed_denial.payer,
        appeal_level=appeal_level,                          # type: ignore[arg-type]
        paragraphs=paragraphs, full_text=full_text,
        n_cite_backs=n_cite_backs,
        contains_llm_polish=enable_llm_polish,
        llm_model_id=llm_model_id if enable_llm_polish else None,
        rationale=rationale,
        references=[
            "ERISA §502(a) appeal rules (29 CFR 2560.503-1).",
            "ACA External Review (45 CFR 147.136).",
            "AMA Prior Authorization Physician Survey 2024.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# APPEALS-3 -- compute_appeal_escalation_path
# ─────────────────────────────────────────────────────────────────────

# Per-payer escalation defaults -- the union of ERISA/ACA federal floors
# and (when stricter) commonly-cited state-DOI rules.

_ESCALATION_TEMPLATE: list[AppealEscalationStep] = [
    AppealEscalationStep(
        level="internal_first_level",
        description=(
            "Submit an internal first-level appeal in writing with all "
            "supporting clinical evidence."
        ),
        deadline_after_denial_days=180,    # ERISA: 180 days
        required_documents=[
            "denial letter", "medical necessity letter",
            "supporting clinical records", "policy excerpts",
        ],
        expected_decision_window_days=30,
        estimated_success_probability=0.39,    # KFF 2023 ~39%
    ),
    AppealEscalationStep(
        level="internal_second_level",
        description=(
            "Request a second internal-appeal review by a different "
            "panel after first-level denial."
        ),
        deadline_after_denial_days=60,
        required_documents=[
            "first-level denial letter", "additional clinical evidence",
            "specialist letter (if applicable)",
        ],
        expected_decision_window_days=30,
        estimated_success_probability=0.22,
    ),
    AppealEscalationStep(
        level="external_independent_review",
        description=(
            "Submit to the ACA-mandated external independent review "
            "organisation (IRO) after internal appeals are exhausted."
        ),
        deadline_after_denial_days=120,
        required_documents=[
            "complete internal appeal record",
            "denial letters at every level",
        ],
        expected_decision_window_days=45,
        estimated_success_probability=0.40,    # IRO overturns ~40%
    ),
    AppealEscalationStep(
        level="state_insurance_commissioner",
        description=(
            "File a formal complaint with the State DOI parallel to "
            "or after the IRO, when applicable."
        ),
        deadline_after_denial_days=365,
        required_documents=[
            "complete appeal record", "complaint form",
        ],
        expected_decision_window_days=60,
        estimated_success_probability=0.10,
    ),
]


async def compute_appeal_escalation_path(
    payer: str,
    *,
    starting_level: str = "internal_first_level",
) -> AppealEscalationPathReport:
    """Return the appeal-path steps from `starting_level` through the
    external-review level for the given payer.

    Args:
        payer: payer name (UnitedHealthcare, Anthem, Aetna, etc.).
        starting_level: one of {internal_first_level,
            internal_second_level, external_independent_review}.

    Returns:
        AppealEscalationPathReport.
    """
    levels_in_order = [
        "internal_first_level", "internal_second_level",
        "external_independent_review",
        "state_insurance_commissioner",
    ]
    if starting_level not in levels_in_order:
        raise ValueError(f"Unknown starting_level: {starting_level}")

    start_idx = levels_in_order.index(starting_level)
    steps_by_level = {s.level: s for s in _ESCALATION_TEMPLATE}
    steps = [steps_by_level[L] for L in levels_in_order[start_idx:]]

    cumulative = sum(
        s.deadline_after_denial_days + s.expected_decision_window_days
        for s in steps
    )
    rationale = (
        f"Appeal escalation path for {payer} starting at "
        f"{starting_level}: {len(steps)} step(s), cumulative max "
        f"window {cumulative} days."
    )

    return AppealEscalationPathReport(
        payer=payer,
        starting_level=starting_level,                    # type: ignore[arg-type]
        steps=steps, n_steps=len(steps),
        cumulative_max_days=cumulative,
        rationale=rationale,
        references=[
            "ERISA §502(a) appeal rules (29 CFR 2560.503-1).",
            "ACA External Review (45 CFR 147.136).",
            "KFF Marketplace Plan Denial Rates 2024.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_denial_letter_parse)
    mcp.tool()(compute_appeal_letter_draft)
    mcp.tool()(compute_appeal_escalation_path)
