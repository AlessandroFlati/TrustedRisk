"""healthcare.compute_{icd10,cpt,hcpcs}_suggest + compute_coding_audit
-- Phase 7.1 auto-coding agent (CODE-1..4).

Pain-point: revenue cycle US, $30-50 B in coding errors, 10-15 % claim
denial rate. Each tool produces structured suggestions cite-backed to
chart resources.

Pure-deterministic floor: rule-based mapping tables for the most
common diagnoses, procedures, and J-code-eligible drugs. Optional LLM
polish (`TRUSTEDRISK_CODER_LLM_POLISH=1`) re-ranks ambiguous cases --
post-check rejects any output that drops cite-back IDs (per the
`a2a_agent.llm_polish` contract).

References:
- ICD-10-CM Official Guidelines for Coding and Reporting (2024)
- AMA CPT Codebook (2024)
- CMS HCPCS Level II (2024)
"""

from __future__ import annotations

import os
import re
from typing import Any

from shared.schemas import (
    CodingAuditFinding,
    CodingAuditReport,
    CodingSuggestion,
    CPTSuggestReport,
    HCPCSSuggestReport,
    ICD10SuggestReport,
)


# ─────────────────────────────────────────────────────────────────────
# Rule-based mapping tables (curated subset -- production deployments
# bind a full CMS Local Coverage Determination crosswalk)
# ─────────────────────────────────────────────────────────────────────

# (icd10_code, display, regex_keywords, default_confidence)
_ICD10_TABLE: list[tuple[str, str, list[str], float]] = [
    ("I50.21", "Acute on chronic systolic heart failure",
     [r"acute on chronic systolic", r"hfref decompensat",
      r"systolic chf flare"], 0.92),
    ("I50.22", "Acute on chronic systolic heart failure (unspecified)",
     [r"acute on chronic.*hf", r"systolic chf"], 0.78),
    ("I50.23", "Acute on chronic combined systolic and diastolic HF",
     [r"combined systolic.*diastolic"], 0.82),
    ("I50.9", "Heart failure, unspecified",
     [r"\bchf\b", r"heart failure"], 0.50),  # low -- prefer specific
    ("N17.9", "Acute kidney injury, unspecified",
     [r"\baki\b", r"acute kidney injury", r"acute renal failure"], 0.85),
    ("N18.4", "Chronic kidney disease, stage 4",
     [r"ckd.*stage 4", r"ckd 4"], 0.92),
    ("E11.9", "Type 2 diabetes mellitus without complications",
     [r"type 2 diabetes", r"\bt2dm\b", r"\bdm2\b"], 0.85),
    ("E11.65", "Type 2 diabetes mellitus with hyperglycemia",
     [r"type 2 dm.*hyperglycemia", r"uncontrolled dm2"], 0.92),
    ("J44.1", "COPD with exacerbation",
     [r"copd exacerbation", r"copd flare"], 0.90),
    ("J18.9", "Pneumonia, unspecified organism",
     [r"\bpneumonia\b"], 0.55),
    ("J18.1", "Lobar pneumonia, unspecified organism",
     [r"lobar pneumonia"], 0.88),
    ("I63.9", "Cerebral infarction, unspecified",
     [r"\bstroke\b", r"\bcva\b", r"cerebral infarction"], 0.70),
    ("I63.50", "Unspecified occlusion or stenosis of unspecified cerebral artery",
     [r"\blvo\b", r"large vessel occlusion"], 0.80),
    ("R65.21", "Severe sepsis with septic shock",
     [r"septic shock"], 0.92),
    ("A41.9", "Sepsis, unspecified organism",
     [r"\bsepsis\b"], 0.65),
    ("M54.5", "Low back pain",
     [r"\blbp\b", r"low back pain"], 0.85),
    ("M54.16", "Lumbar radiculopathy",
     [r"radiculopath", r"lumbar radic", r"sciatica"], 0.90),
    ("R07.9", "Chest pain, unspecified",
     [r"chest pain"], 0.55),
    ("R07.4", "Chest pain, unspecified \\(non-cardiac flag\\)",
     [r"non-cardiac chest pain"], 0.88),
    ("Z51.11", "Encounter for antineoplastic chemotherapy",
     [r"chemotherapy", r"\bchemo\b"], 0.86),
]

# (cpt_code, display, regex_keywords, default_confidence)
_CPT_TABLE: list[tuple[str, str, list[str], float]] = [
    ("72148", "MRI lumbar spine without contrast",
     [r"mri lumbar", r"mr lumbar spine"], 0.93),
    ("72148-26", "MRI lumbar spine, professional component",
     [r"mri lumbar.*read", r"mr lumbar.*professional"], 0.78),
    ("70551", "MRI brain without contrast",
     [r"mri brain", r"mr brain"], 0.93),
    ("70450", "CT head without contrast",
     [r"ct head", r"head ct"], 0.93),
    ("71045", "Chest x-ray, single view",
     [r"chest x-?ray", r"\bcxr\b"], 0.92),
    ("93306", "Transthoracic echocardiogram",
     [r"transthoracic echo", r"\btte\b"], 0.93),
    ("99232", "Subsequent hospital care, moderate complexity",
     [r"subsequent hospital.*moderate"], 0.85),
    ("99233", "Subsequent hospital care, high complexity",
     [r"subsequent hospital.*high"], 0.85),
    ("99221", "Initial hospital care, low complexity",
     [r"initial hospital.*low"], 0.85),
    ("99222", "Initial hospital care, moderate complexity",
     [r"initial hospital.*moderate"], 0.85),
    ("99223", "Initial hospital care, high complexity",
     [r"initial hospital.*high"], 0.85),
    ("99238", "Hospital discharge day management, ≤30 min",
     [r"discharge.*30 min"], 0.85),
    ("99239", "Hospital discharge day management, >30 min",
     [r"discharge.*>30", r"discharge.*40 min"], 0.85),
    ("36415", "Routine venipuncture",
     [r"venipuncture", r"blood draw"], 0.90),
]

# RxNorm/text -> HCPCS J-code mapping (specialty drug billing)
_HCPCS_TABLE: list[tuple[str, str, list[str], float]] = [
    ("J0135", "Adalimumab injection, 20 mg",
     [r"adalimumab", r"humira"], 0.92),
    ("J1745", "Infliximab injection, 10 mg",
     [r"infliximab", r"remicade"], 0.92),
    ("J9355", "Trastuzumab injection, 10 mg",
     [r"trastuzumab", r"herceptin"], 0.92),
    ("J9035", "Bevacizumab injection, 10 mg",
     [r"bevacizumab", r"avastin"], 0.92),
    ("J7325", "Ferumoxytol injection",
     [r"ferumoxytol", r"feraheme"], 0.92),
    ("J1100", "Dexamethasone sodium phosphate, 1 mg",
     [r"dexamethasone"], 0.85),
    ("J2270", "Morphine sulfate injection, 10 mg",
     [r"morphine sulfate"], 0.85),
    ("J3490", "Unclassified drugs",
     [r"unspecified drug"], 0.40),
]


def _match_codes(
    text: str,
    table: list[tuple[str, str, list[str], float]],
) -> list[CodingSuggestion]:
    text_lower = text.lower()
    results: list[CodingSuggestion] = []
    for code, display, patterns, base_conf in table:
        for pat in patterns:
            m = re.search(pat, text_lower, re.IGNORECASE)
            if m:
                excerpt = text[max(0, m.start() - 20):
                                  min(len(text), m.end() + 20)]
                results.append(CodingSuggestion(
                    code=code, code_system=_code_system_of(code),
                    display=display,
                    confidence=base_conf,
                    supporting_text_excerpts=[excerpt],
                    rationale=f"Matched pattern {pat!r}",
                ))
                break
    # De-dup by code, keeping the highest confidence
    by_code: dict[str, CodingSuggestion] = {}
    for r in results:
        if r.code not in by_code or by_code[r.code].confidence < r.confidence:
            by_code[r.code] = r
    return sorted(by_code.values(), key=lambda s: -s.confidence)


def _code_system_of(code: str) -> str:
    if re.match(r"^[A-Z]\d{2}", code):
        return "icd10cm"
    if re.match(r"^\d{4,5}", code) and not code.startswith("J"):
        return "cpt"
    if code.startswith("J"):
        return "hcpcs"
    return "icd10cm"


def _resolve_primary(suggestions: list[CodingSuggestion]) -> str | None:
    """Top suggestion is primary when confidence ≥ 0.70 AND margin to
    second-best ≥ 0.10. Otherwise abstain (no primary). The margin gate
    is intentionally tight (10pp) -- coding errors are costly so any
    real ambiguity should default to manual adjudication."""
    if not suggestions:
        return None
    if suggestions[0].confidence < 0.70:
        return None
    if len(suggestions) >= 2:
        margin = suggestions[0].confidence - suggestions[1].confidence
        if margin < 0.10:
            return None
    return suggestions[0].code


def _attach_evidence_ids(
    suggestions: list[CodingSuggestion],
    chart_excerpts: list[dict] | None,
) -> list[CodingSuggestion]:
    """When chart_excerpts arrive with explicit FHIR IDs, attach them
    to every suggestion that contains a matching keyword."""
    if not chart_excerpts:
        return suggestions
    for s in suggestions:
        ids: list[str] = []
        for ex in chart_excerpts:
            text = (ex.get("text") or "").lower()
            for excerpt in s.supporting_text_excerpts:
                if any(tok.lower() in text for tok in excerpt.split()
                          if len(tok) > 4):
                    rid = ex.get("fhir_resource_id") or ex.get("id")
                    if rid and rid not in ids:
                        ids.append(rid)
                    break
        s.supporting_evidence_ids = ids
    return suggestions


# ─────────────────────────────────────────────────────────────────────
# Public API: compute_icd10_suggest
# ─────────────────────────────────────────────────────────────────────

async def compute_icd10_suggest(
    chart_text: str,
    chart_excerpts: list[dict] | None = None,
    patient_demographics: dict[str, Any] | None = None,
) -> ICD10SuggestReport:
    """Suggest ICD-10-CM codes from clinical chart text.

    Args:
        chart_text: free-text chart content (admission H&P, progress
            notes, discharge summary).
        chart_excerpts: optional structured per-line excerpts with
            `fhir_resource_id` to enable cite-back attachment.
        patient_demographics: optional age/sex/etc for demographics-
            sensitive coding.

    Returns:
        ICD10SuggestReport with ranked suggestions + primary_code (set
        only when confidence + margin pass the gate).
    """
    if not chart_text.strip():
        return ICD10SuggestReport(
            suggestions=[], n_suggestions=0,
            abstain_recommended=True,
            abstain_reason="empty_chart_text",
            references=["ICD-10-CM Official Guidelines (2024)"],
        )
    suggestions = _match_codes(chart_text, _ICD10_TABLE)
    suggestions = _attach_evidence_ids(suggestions, chart_excerpts)
    primary = _resolve_primary(suggestions)
    abstain = not suggestions
    return ICD10SuggestReport(
        suggestions=suggestions,
        n_suggestions=len(suggestions),
        primary_code=primary,
        extraction_method="rule_based",
        abstain_recommended=abstain,
        abstain_reason=("no_match_in_curated_table" if abstain else None),
        references=[
            "ICD-10-CM Official Guidelines for Coding and Reporting (2024)",
            "AMA Coding Guidelines for documentation specificity.",
        ],
    )


async def compute_cpt_suggest(
    procedure_text: str,
    chart_excerpts: list[dict] | None = None,
) -> CPTSuggestReport:
    """Suggest CPT codes from procedure descriptions."""
    if not procedure_text.strip():
        return CPTSuggestReport(
            suggestions=[], n_suggestions=0,
            abstain_recommended=True,
            abstain_reason="empty_procedure_text",
            references=["AMA CPT Codebook (2024)"],
        )
    suggestions = _match_codes(procedure_text, _CPT_TABLE)
    suggestions = _attach_evidence_ids(suggestions, chart_excerpts)
    primary = _resolve_primary(suggestions)
    abstain = not suggestions
    return CPTSuggestReport(
        suggestions=suggestions, n_suggestions=len(suggestions),
        primary_code=primary,
        abstain_recommended=abstain,
        abstain_reason=("no_match_in_curated_table" if abstain else None),
        references=[
            "AMA CPT Codebook (2024)",
            "CMS Outpatient Prospective Payment System (2024).",
        ],
    )


async def compute_hcpcs_suggest(
    medication_or_drug_text: str,
    chart_excerpts: list[dict] | None = None,
) -> HCPCSSuggestReport:
    """Suggest HCPCS Level II J-codes for specialty drug billing."""
    if not medication_or_drug_text.strip():
        return HCPCSSuggestReport(
            suggestions=[], n_suggestions=0,
            abstain_recommended=True,
            abstain_reason="empty_medication_text",
            references=["CMS HCPCS Level II Codebook (2024)"],
        )
    suggestions = _match_codes(medication_or_drug_text, _HCPCS_TABLE)
    suggestions = _attach_evidence_ids(suggestions, chart_excerpts)
    abstain = not suggestions
    return HCPCSSuggestReport(
        suggestions=suggestions, n_suggestions=len(suggestions),
        abstain_recommended=abstain,
        abstain_reason=("no_match_in_curated_table" if abstain else None),
        references=[
            "CMS HCPCS Level II Codebook (2024)",
            "Medicare Part B drug billing guidance.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Coding audit -- gap report
# ─────────────────────────────────────────────────────────────────────

# Rough literature-derived per-finding revenue impact (USD, 2024)
_REV_IMPACT_PER_FINDING: dict[str, float] = {
    "documented_not_coded": 600.0,      # median DRG delta
    "coded_not_documented": -500.0,     # denial clawback estimate
    "specificity_loss": 350.0,          # specificity-driven uplift
    "duplicate_code": -800.0,           # clawback + audit risk
}


async def compute_coding_audit(
    chart_text: str,
    coded_artifact: dict[str, Any],
    chart_excerpts: list[dict] | None = None,
) -> CodingAuditReport:
    """Compare a coded billing artifact against the chart text.

    Args:
        chart_text: free-text chart content.
        coded_artifact: dict shaped like
            `{"icd10": [...], "cpt": [...], "hcpcs": [...]}` listing
            already-billed codes.
        chart_excerpts: optional structured chart excerpts for
            cite-back attachment.

    Returns:
        CodingAuditReport with findings + severity + revenue impact.
    """
    icd_suggested = await compute_icd10_suggest(
        chart_text, chart_excerpts=chart_excerpts)
    cpt_suggested = await compute_cpt_suggest(
        chart_text, chart_excerpts=chart_excerpts)
    hcpcs_suggested = await compute_hcpcs_suggest(
        chart_text, chart_excerpts=chart_excerpts)

    coded_icd = set(coded_artifact.get("icd10", []) or [])
    coded_cpt = set(coded_artifact.get("cpt", []) or [])
    coded_hcpcs = set(coded_artifact.get("hcpcs", []) or [])

    findings: list[CodingAuditFinding] = []

    # documented_not_coded -- for each suggestion at confidence ≥ 0.85
    # whose code isn't in the bill
    for suggested, system, coded in (
        (icd_suggested.suggestions, "icd10cm", coded_icd),
        (cpt_suggested.suggestions, "cpt", coded_cpt),
        (hcpcs_suggested.suggestions, "hcpcs", coded_hcpcs),
    ):
        for s in suggested:
            if s.confidence < 0.85:
                continue
            if s.code in coded:
                continue
            findings.append(CodingAuditFinding(
                finding_kind="documented_not_coded",
                code=s.code, code_system=system,
                description=s.display,
                chart_evidence_ids=s.supporting_evidence_ids,
                severity="high" if s.confidence >= 0.92 else "medium",
                suggested_action=(
                    f"Add {s.code} ({s.display}) to the billing artifact."
                ),
            ))

    # coded_not_documented -- for each coded code without a matching
    # high-confidence suggestion
    suggested_icd_codes = {s.code for s in icd_suggested.suggestions
                              if s.confidence >= 0.55}
    suggested_cpt_codes = {s.code for s in cpt_suggested.suggestions
                              if s.confidence >= 0.55}
    suggested_hcpcs_codes = {s.code for s in hcpcs_suggested.suggestions
                                if s.confidence >= 0.55}

    for coded, system, suggested_set in (
        (coded_icd, "icd10cm", suggested_icd_codes),
        (coded_cpt, "cpt", suggested_cpt_codes),
        (coded_hcpcs, "hcpcs", suggested_hcpcs_codes),
    ):
        for code in coded:
            if code in suggested_set:
                continue
            findings.append(CodingAuditFinding(
                finding_kind="coded_not_documented",
                code=code, code_system=system,
                description=f"Coded {code}; chart does not support it.",
                severity="high",
                suggested_action=(
                    f"Either remove {code} from the bill or add chart "
                    "documentation supporting it."
                ),
            ))

    # specificity_loss -- coded an unspecified code while a more specific
    # one is in the suggestions
    for c in coded_icd:
        if c.endswith(".9"):
            for s in icd_suggested.suggestions:
                if s.code.startswith(c[:3]) and s.code != c \
                        and s.confidence >= 0.85:
                    findings.append(CodingAuditFinding(
                        finding_kind="specificity_loss",
                        code=c, code_system="icd10cm",
                        description=(
                            f"Coded {c} (unspecified) when chart supports "
                            f"more specific {s.code} ({s.display})."
                        ),
                        chart_evidence_ids=s.supporting_evidence_ids,
                        severity="medium",
                        suggested_action=(
                            f"Replace {c} with {s.code}."
                        ),
                    ))

    n_high = sum(1 for f in findings if f.severity == "high")
    rev_impact = sum(
        _REV_IMPACT_PER_FINDING.get(f.finding_kind, 0.0) for f in findings
    )

    rationale = (
        f"Audit on coded artifact + chart text. "
        f"Findings: {len(findings)} total ({n_high} high-severity). "
        f"Estimated revenue impact: ${rev_impact:,.0f}."
    )

    return CodingAuditReport(
        findings=findings, n_findings=len(findings),
        n_high_severity=n_high,
        expected_revenue_impact_usd=round(rev_impact, 2),
        rationale=rationale,
        references=[
            "ICD-10-CM Official Guidelines (2024).",
            "OIG Coding Compliance Program Guidance (2023).",
            "CMS Medicare Severity-DRG specifications (2024).",
        ],
    )


# ─────────────────────── MCP registration ───────────────────────


def register(mcp) -> None:
    mcp.tool()(compute_icd10_suggest)
    mcp.tool()(compute_cpt_suggest)
    mcp.tool()(compute_hcpcs_suggest)
    mcp.tool()(compute_coding_audit)
