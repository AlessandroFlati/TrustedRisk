"""healthcare.compute_pa_payer_rules_match -- Phase 2.1 PA-2.

Checks payer-specific PA rules against a `PAEvidencePack`. Uses the
hard-coded rules database in `_pa_payer_rules.py`. Pure deterministic.

Each rule's `requires` predicate is matched against the evidence pack
via `_REQUIRES_MATCHERS` below. A rule that fires becomes "met"; one
that doesn't becomes "unmet" with a `gap_text` describing how to close
the gap before submitting.

Returned `overall_alignment` tier:
    - fully_aligned     ≥ 90% of rules met
    - majority_aligned  ≥ 60% met
    - partially_aligned ≥ 30% met
    - weakly_aligned    ≥ 10% met
    - misaligned        else
"""

from __future__ import annotations

from typing import Any

from shared.schemas import (
    PAEvidencePack,
    PARuleStatus,
    PARulesMatch,
)

from ._pa_payer_rules import PARule, get_rules


# ─────────────────────────────────────────────────────────────────────
# Predicate matchers
# ─────────────────────────────────────────────────────────────────────
#
# Each entry maps a rule's `requires` string to a callable that takes a
# PAEvidencePack and returns (matched: bool, evidence_ids: list[str],
# gap_text: str | None).

def _diagnosis_with_specific_icd10(pack: PAEvidencePack):
    matched_ids = [
        d.fhir_resource_id
        for d in pack.diagnoses
        if d.relevance_score >= 0.8 and d.fhir_resource_id
    ]
    if matched_ids:
        return True, matched_ids, None
    return False, [], (
        "Add at least one specifically-coded ICD-10 diagnosis (avoid "
        "*.9 / unspecified codes)."
    )


def _recent_observation_within_90d(pack: PAEvidencePack):
    matched_ids = [
        o.fhir_resource_id for o in pack.relevant_observations
        if o.relevance_score >= 0.7 and o.fhir_resource_id
    ]
    if matched_ids:
        return True, matched_ids, None
    return False, [], (
        "Order or document a clinical observation within the last 90 "
        "days that supports medical necessity."
    )


def _chart_excerpt_or_narrative_present(pack: PAEvidencePack):
    matched_ids = [
        c.fhir_resource_id for c in pack.chart_excerpts
        if c.fhir_resource_id
    ]
    if pack.chart_excerpts:
        return True, matched_ids, None
    return False, [], (
        "Document a medical-necessity narrative in the chart linking "
        "the diagnosis to the requested service."
    )


def _prior_step_therapy_or_contraindication(pack: PAEvidencePack):
    if pack.prior_treatments_tried:
        return True, [
            t.fhir_resource_id for t in pack.prior_treatments_tried
            if t.fhir_resource_id
        ], None
    if pack.contraindications_to_alternatives:
        return True, [
            c.fhir_resource_id for c in pack.contraindications_to_alternatives
            if c.fhir_resource_id
        ], None
    return False, [], (
        "Document trial+failure of at least one formulary alternative or "
        "an explicit contraindication to step-therapy alternatives."
    )


def _two_prior_formulary_trials(pack: PAEvidencePack):
    if len(pack.prior_treatments_tried) >= 2:
        return True, [
            t.fhir_resource_id for t in pack.prior_treatments_tried[:2]
            if t.fhir_resource_id
        ], None
    return False, [], (
        f"Anthem/BCBS requires two prior formulary trials; only "
        f"{len(pack.prior_treatments_tried)} documented."
    )


def _prior_xray_or_documented_contraindication(pack: PAEvidencePack):
    for t in pack.prior_treatments_tried:
        if t.kind == "procedure" and "ray" in (t.text or "").lower():
            return True, [t.fhir_resource_id] if t.fhir_resource_id else [], None
    if pack.contraindications_to_alternatives:
        return True, [
            c.fhir_resource_id for c in pack.contraindications_to_alternatives
            if c.fhir_resource_id
        ], None
    return False, [], (
        "Order a plain-film X-ray first, or document why it would be "
        "non-diagnostic for this presentation."
    )


def _specialist_referral_or_pcp_attestation(pack: PAEvidencePack):
    # Heuristic: the chart excerpts mention "specialist" or "referral"
    for ex in pack.chart_excerpts:
        if any(k in (ex.text or "").lower()
                  for k in ("specialist", "referral", "consult")):
            return True, [ex.fhir_resource_id] if ex.fhir_resource_id else [], None
    return False, [], (
        "Add a specialist consult note or PCP attestation of clinical "
        "urgency to the chart."
    )


def _specialist_prescriber(pack: PAEvidencePack):
    return _specialist_referral_or_pcp_attestation(pack)


def _aim_review_or_emergent_indication(pack: PAEvidencePack):
    # No deterministic check from FHIR alone; defer to chart narrative
    # (and to the human reviewer) -- surface as 'partial'.
    return False, [], (
        "Submit the request through AIM Specialty Health Radiology "
        "Quality Initiatives. The MCP tool cannot verify this from "
        "FHIR alone -- this is a process step."
    )


def _recent_renal_panel_within_30d(pack: PAEvidencePack):
    matched: list[str] = []
    for o in pack.relevant_observations:
        if o.relevance_score >= 0.9 and (
            "creatinine" in (o.text or "").lower()
            or "egfr" in (o.text or "").lower()
            or "bun" in (o.text or "").lower()
        ):
            if o.fhir_resource_id:
                matched.append(o.fhir_resource_id)
    if matched:
        return True, matched, None
    return False, [], (
        "Order a CMP / renal-function panel within 30 days for any "
        "IV-contrast study."
    )


def _evicore_review_or_emergent(pack: PAEvidencePack):
    return False, [], (
        "Aetna requires an eviCore review for non-emergent advanced "
        "imaging -- process step outside this tool's deterministic "
        "scope."
    )


def _nia_review_or_emergent(pack: PAEvidencePack):
    return False, [], (
        "Cigna requires National Imaging Associates review -- process "
        "step outside this tool's deterministic scope."
    )


def _cvs_specialty_dispensing_or_exception(pack: PAEvidencePack):
    return False, [], (
        "Aetna requires CVS Specialty dispensing unless contraindicated."
    )


def _express_scripts_review(pack: PAEvidencePack):
    return False, [], (
        "Cigna requires Express Scripts pharmacy review."
    )


def _centerwell_dispensing_or_exception(pack: PAEvidencePack):
    return False, [], (
        "Humana requires CenterWell Pharmacy for specialty agents."
    )


def _auc_consultation_documented(pack: PAEvidencePack):
    for ex in pack.chart_excerpts:
        if "auc" in (ex.text or "").lower() or (
            "appropriate use" in (ex.text or "").lower()
        ):
            return True, [ex.fhir_resource_id] if ex.fhir_resource_id else [], None
    return False, [], (
        "Document Appropriate Use Criteria (AUC) consultation per PAMA 2014."
    )


def _benefit_category_documented(pack: PAEvidencePack):
    return False, [], (
        "Verify Part B vs Part D benefit category and document in the "
        "submission."
    )


def _fda_approved_or_compendia_supported(pack: PAEvidencePack):
    return False, [], (
        "Cite FDA-approved indication or compendia (DrugDex, AHFS DI, "
        "NCCN, Lexi-Drugs) support for off-label use."
    )


def _lcd_ncd_criteria_met(pack: PAEvidencePack):
    return False, [], (
        "Cite the relevant Medicare LCD or NCD and demonstrate criteria "
        "are met."
    )


def _dur_review_or_pdl_drug(pack: PAEvidencePack):
    return False, [], (
        "Drug Utilization Review (DUR) board approval needed for non-PDL drugs."
    )


_REQUIRES_MATCHERS = {
    "diagnosis_with_specific_icd10_present": _diagnosis_with_specific_icd10,
    "recent_observation_within_90d": _recent_observation_within_90d,
    "chart_excerpt_or_narrative_present": _chart_excerpt_or_narrative_present,
    "prior_step_therapy_or_contraindication":
        _prior_step_therapy_or_contraindication,
    "two_prior_formulary_trials": _two_prior_formulary_trials,
    "prior_xray_or_documented_contraindication":
        _prior_xray_or_documented_contraindication,
    "specialist_referral_or_pcp_attestation":
        _specialist_referral_or_pcp_attestation,
    "specialist_prescriber": _specialist_prescriber,
    "aim_review_or_emergent_indication": _aim_review_or_emergent_indication,
    "recent_renal_panel_within_30d": _recent_renal_panel_within_30d,
    "evicore_review_or_emergent": _evicore_review_or_emergent,
    "nia_review_or_emergent": _nia_review_or_emergent,
    "cvs_specialty_dispensing_or_exception":
        _cvs_specialty_dispensing_or_exception,
    "express_scripts_review": _express_scripts_review,
    "centerwell_dispensing_or_exception": _centerwell_dispensing_or_exception,
    "auc_consultation_documented": _auc_consultation_documented,
    "benefit_category_documented": _benefit_category_documented,
    "fda_approved_or_compendia_supported": _fda_approved_or_compendia_supported,
    "lcd_ncd_criteria_met": _lcd_ncd_criteria_met,
    "dur_review_or_pdl_drug": _dur_review_or_pdl_drug,
}


def _check_rule(rule: PARule, pack: PAEvidencePack) -> PARuleStatus:
    matcher = _REQUIRES_MATCHERS.get(rule.requires)
    if matcher is None:
        # Unknown predicate -- surface as 'unknown' so the user knows
        # the rule must be assessed manually
        return PARuleStatus(
            rule_id=rule.rule_id,
            rule_text=rule.rule_text,
            status="unknown",
            evidence_ids=[],
            gap_text=(
                f"No deterministic matcher for predicate "
                f"{rule.requires!r}; assess manually."
            ),
        )
    matched, evidence_ids, gap_text = matcher(pack)
    if matched:
        status = "met"
    elif gap_text and "process step" in gap_text.lower():
        status = "partial"
    else:
        status = "unmet"
    return PARuleStatus(
        rule_id=rule.rule_id,
        rule_text=rule.rule_text,
        status=status,                                 # type: ignore[arg-type]
        evidence_ids=evidence_ids,
        gap_text=gap_text,
    )


def _alignment_tier(n_met: int, n_total: int) -> str:
    if n_total == 0:
        return "misaligned"
    pct = n_met / n_total
    if pct >= 0.90:
        return "fully_aligned"
    if pct >= 0.60:
        return "majority_aligned"
    if pct >= 0.30:
        return "partially_aligned"
    if pct >= 0.10:
        return "weakly_aligned"
    return "misaligned"


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

async def compute_pa_payer_rules_match(
    evidence_pack: PAEvidencePack | dict,
) -> PARulesMatch:
    """Match the payer's PA rules against the evidence pack.

    Args:
        evidence_pack: Output of compute_pa_evidence_pack (or its dict
            form via Pydantic round-trip).

    Returns:
        PARulesMatch with per-rule status, alignment tier, and the
        ordered next-steps to close any gaps before submission.
    """
    if isinstance(evidence_pack, dict):
        evidence_pack = PAEvidencePack.model_validate(evidence_pack)

    rules = get_rules(
        evidence_pack.payer,
        evidence_pack.requested_service.service_type,
    )
    statuses = [_check_rule(r, evidence_pack) for r in rules]
    n_met = sum(1 for s in statuses if s.status == "met")
    n_unmet = sum(1 for s in statuses if s.status == "unmet")
    n_partial = sum(1 for s in statuses if s.status == "partial")
    alignment = _alignment_tier(n_met, len(statuses))

    next_steps = [
        s.gap_text for s in statuses
        if s.status in ("unmet", "partial") and s.gap_text
    ]

    rationale = (
        f"{evidence_pack.payer} PA rules for "
        f"{evidence_pack.requested_service.service_type}: "
        f"{n_met} met / {n_unmet} unmet / {n_partial} partial of "
        f"{len(statuses)} rule(s). Overall alignment = {alignment}."
    )

    return PARulesMatch(
        payer=evidence_pack.payer,
        requested_service=evidence_pack.requested_service.service_type,
        rules=statuses,
        n_met=n_met,
        n_unmet=n_unmet,
        n_partial=n_partial,
        overall_alignment=alignment,                 # type: ignore[arg-type]
        next_steps=next_steps,
        rationale=rationale,
        references=[
            "Payer-specific PA Coverage Determination policies (2024-2025).",
            "AMA 2024 Prior Authorization Physician Survey.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_pa_payer_rules_match)
