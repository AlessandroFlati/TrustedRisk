"""Payer-specific Prior Authorization rules database (Phase 2.1).

Hard-coded snapshot of the top US payers' PA criteria. Each entry maps
a (payer, service_type) pair to a list of structured rules. Each rule
is matchable against a `PAEvidencePack`: the rule fires when its
`requires` condition is satisfied by some evidence item in the pack.

Sources:
- AMA 2024 *Prior Authorization Physician Survey*
- KFF (Kaiser Family Foundation) *Prior Authorization Tracker* (2025)
- Medicare Claims Processing Manual ch. 30 (medical necessity criteria)
- Medicaid Drug Utilization Review (DUR) program guidance
- Each payer's published Coverage Determination policies (2024-2025
  snapshot, hard-coded -- re-fit on production deployment as policies
  drift)

This is a representative subset -- production deployments must keep this
table refreshed against the payers' policy bulletins quarterly.

The `generic` fallback codifies AHRQ-style medical-necessity criteria
that apply when no payer-specific rule matches. It's intentionally
conservative: when in doubt, the rules-match output flags `unmet` so
the letter draft addresses the gap explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PARule:
    """A single PA requirement.

    `requires` describes what the evidence pack must contain for this
    rule to be met. The matching logic is in
    `mcp_server.tools.pa_payer_rules_match`.
    """
    rule_id: str
    rule_text: str
    requires: str
    weight: float = 1.0   # multiplier for the evidence-strength score


# Service-type-agnostic rules apply across every PA request for a payer.
_GENERIC_RULES: list[PARule] = [
    PARule(
        rule_id="generic.dx_specificity",
        rule_text=(
            "Diagnosis must be coded to ICD-10 specificity (no unspecified codes)."
        ),
        requires="diagnosis_with_specific_icd10_present",
    ),
    PARule(
        rule_id="generic.evidence_recency",
        rule_text=(
            "Supporting clinical observations must be ≤90 days old at "
            "the time of submission."
        ),
        requires="recent_observation_within_90d",
    ),
    PARule(
        rule_id="generic.medical_necessity_narrative",
        rule_text=(
            "Chart must contain a medical-necessity narrative tying "
            "the requested service to the diagnosis (AHRQ-style)."
        ),
        requires="chart_excerpt_or_narrative_present",
    ),
]


# Per-(payer, service_type) overrides + additions
PAYER_SERVICE_RULES: dict[
    tuple[str, str], list[PARule]
] = {
    # ─── UnitedHealth ──────────────────────────────────────────────────
    ("unitedhealth", "imaging_advanced"): [
        PARule(
            "uhc.img.exhaust_xray",
            "Plain-film X-ray must be tried first when clinically appropriate "
            "(MSK indications).",
            "prior_xray_or_documented_contraindication",
        ),
        PARule(
            "uhc.img.specialist_referral",
            "Advanced imaging requires specialist referral or PCP attestation "
            "of clinical urgency.",
            "specialist_referral_or_pcp_attestation",
        ),
    ],
    ("unitedhealth", "specialty_drug"): [
        PARule(
            "uhc.drug.step_therapy",
            "Step therapy: at least 1 generic / preferred-formulary "
            "alternative tried for ≥30 days unless contraindicated.",
            "prior_step_therapy_or_contraindication",
            weight=2.0,
        ),
        PARule(
            "uhc.drug.specialist_prescriber",
            "Specialty drug must be prescribed by an in-network specialist.",
            "specialist_prescriber",
        ),
    ],
    # ─── Anthem / BCBS ────────────────────────────────────────────────
    ("anthem_bcbs", "imaging_advanced"): [
        PARule(
            "bcbs.img.aim_specialty",
            "AIM Specialty Health Radiology Quality Initiatives review "
            "required for non-emergent advanced imaging.",
            "aim_review_or_emergent_indication",
        ),
        PARule(
            "bcbs.img.recent_labs",
            "Recent CMP / renal labs ≤30 days for IV-contrast studies.",
            "recent_renal_panel_within_30d",
        ),
    ],
    ("anthem_bcbs", "specialty_drug"): [
        PARule(
            "bcbs.drug.formulary_alternatives",
            "Documentation of trial/failure of at least 2 formulary "
            "alternatives.",
            "two_prior_formulary_trials",
            weight=2.0,
        ),
    ],
    # ─── Aetna ─────────────────────────────────────────────────────────
    ("aetna", "imaging_advanced"): [
        PARule(
            "aetna.img.evicore",
            "eviCore healthcare review for advanced imaging.",
            "evicore_review_or_emergent",
        ),
    ],
    ("aetna", "specialty_drug"): [
        PARule(
            "aetna.drug.cvs_specialty",
            "Dispense via CVS Specialty unless clinically contraindicated.",
            "cvs_specialty_dispensing_or_exception",
        ),
        PARule(
            "aetna.drug.step_therapy",
            "Step therapy: 1+ generic alternatives tried unless contraindicated.",
            "prior_step_therapy_or_contraindication",
            weight=1.5,
        ),
    ],
    # ─── Cigna ────────────────────────────────────────────────────────
    ("cigna", "imaging_advanced"): [
        PARule(
            "cigna.img.medsolutions",
            "Medical Solutions / National Imaging Associates review.",
            "nia_review_or_emergent",
        ),
    ],
    ("cigna", "specialty_drug"): [
        PARule(
            "cigna.drug.express_scripts",
            "Express Scripts pharmacy benefit review.",
            "express_scripts_review",
        ),
        PARule(
            "cigna.drug.step_therapy",
            "Step therapy: 30-day trial of formulary alternative.",
            "prior_step_therapy_or_contraindication",
            weight=1.5,
        ),
    ],
    # ─── Humana ───────────────────────────────────────────────────────
    ("humana", "specialty_drug"): [
        PARule(
            "humana.drug.specialty_pharmacy",
            "Dispense via CenterWell Pharmacy for specialty agents.",
            "centerwell_dispensing_or_exception",
        ),
    ],
    # ─── Medicare ─────────────────────────────────────────────────────
    ("medicare", "imaging_advanced"): [
        PARule(
            "medicare.img.appropriate_use_criteria",
            "Appropriate Use Criteria (AUC) consultation per PAMA 2014 "
            "for advanced diagnostic imaging.",
            "auc_consultation_documented",
            weight=2.0,
        ),
    ],
    ("medicare", "specialty_drug"): [
        PARule(
            "medicare.drug.part_b_b_vs_d",
            "Verify benefit category -- Part B vs Part D coverage.",
            "benefit_category_documented",
        ),
        PARule(
            "medicare.drug.medical_necessity",
            "FDA-approved indication or compendia-supported off-label use.",
            "fda_approved_or_compendia_supported",
            weight=2.0,
        ),
    ],
    ("medicare", "elective_procedure"): [
        PARule(
            "medicare.proc.lcd_ncd",
            "Local Coverage Determination (LCD) or National Coverage "
            "Determination (NCD) criteria met.",
            "lcd_ncd_criteria_met",
            weight=2.0,
        ),
    ],
    # ─── Medicaid ─────────────────────────────────────────────────────
    ("medicaid", "specialty_drug"): [
        PARule(
            "medicaid.drug.dur",
            "Drug Utilization Review (DUR) board approval for non-PDL drugs.",
            "dur_review_or_pdl_drug",
        ),
    ],
}


def get_rules(payer: str, service_type: str) -> list[PARule]:
    """Return generic rules + payer-specific rules for the (payer,
    service_type) tuple. Generic rules apply to every PA request."""
    specific = PAYER_SERVICE_RULES.get((payer, service_type), [])
    return list(_GENERIC_RULES) + list(specific)


# ─────────────────────────────────────────────────────────────────────
# Payer denial-rate prior (literature-derived, per AMA 2024 + KFF 2025)
# Probability of approval on first submission, before any evidence pack
# refinement. Updated per service type.
# ─────────────────────────────────────────────────────────────────────

PAYER_BASE_APPROVAL_RATE: dict[tuple[str, str], float] = {
    ("unitedhealth", "imaging_advanced"): 0.69,
    ("unitedhealth", "specialty_drug"): 0.55,
    ("anthem_bcbs", "imaging_advanced"): 0.74,
    ("anthem_bcbs", "specialty_drug"): 0.61,
    ("aetna", "imaging_advanced"): 0.77,
    ("aetna", "specialty_drug"): 0.65,
    ("cigna", "imaging_advanced"): 0.71,
    ("cigna", "specialty_drug"): 0.58,
    ("humana", "specialty_drug"): 0.70,
    ("humana", "imaging_advanced"): 0.78,
    ("medicare", "imaging_advanced"): 0.91,
    ("medicare", "specialty_drug"): 0.83,
    ("medicare", "elective_procedure"): 0.86,
    ("medicaid", "specialty_drug"): 0.75,
    ("medicaid", "imaging_advanced"): 0.82,
}

# Generic / unknown payer fallback
GENERIC_BASE_APPROVAL_RATE: float = 0.65
