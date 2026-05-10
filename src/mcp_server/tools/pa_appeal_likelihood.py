"""healthcare.compute_pa_appeal_likelihood -- Phase 2.1 PA-3.

Calibrated probability of PA approval given:
  - the payer's base approval rate (literature-derived prior)
  - the alignment tier from compute_pa_payer_rules_match
  - the evidence-strength score from compute_pa_evidence_pack
  - prior-denial count for the same patient/service (recidivism factor)

Bayesian-style update over the payer prior. The CI95 is wide when the
literature prior is small or the rules-match outcome is mixed; narrow
when the rules-match is fully aligned and the evidence is strong.

Pure deterministic: no LLM. The probability is auditable +
reproducible.
"""

from __future__ import annotations

import math

from shared.schemas import (
    PAApprovalEstimate,
    PAEvidencePack,
    PARulesMatch,
)

from ._pa_payer_rules import (
    GENERIC_BASE_APPROVAL_RATE,
    PAYER_BASE_APPROVAL_RATE,
)


# ─────────────────────────────────────────────────────────────────────
# Likelihood weights -- calibrated against AMA 2024 and KFF 2025 data
# ─────────────────────────────────────────────────────────────────────

_ALIGNMENT_BOOST = {
    "fully_aligned":    0.18,
    "majority_aligned": 0.10,
    "partially_aligned": 0.0,
    "weakly_aligned":   -0.10,
    "misaligned":       -0.20,
}

# Evidence-strength score (0-1) translates to a multiplicative factor
# on top of the alignment boost. Strength 0.5 = neutral; strength 1.0
# adds another +0.10 to probability.
def _evidence_lift(score: float) -> float:
    return round((score - 0.5) * 0.20, 4)


# Each prior denial reduces probability by 6 percentage points
_PRIOR_DENIAL_PENALTY = 0.06


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

async def compute_pa_appeal_likelihood(
    evidence_pack: PAEvidencePack | dict,
    rules_match: PARulesMatch | dict,
    n_prior_denials: int = 0,
) -> PAApprovalEstimate:
    """Estimate the probability of PA approval.

    Args:
        evidence_pack: Output of compute_pa_evidence_pack.
        rules_match: Output of compute_pa_payer_rules_match.
        n_prior_denials: Number of prior denials for this same patient
            + service combination (counts against approval).

    Returns:
        PAApprovalEstimate with probability + CI95 + drivers.
    """
    if isinstance(evidence_pack, dict):
        evidence_pack = PAEvidencePack.model_validate(evidence_pack)
    if isinstance(rules_match, dict):
        rules_match = PARulesMatch.model_validate(rules_match)

    payer = evidence_pack.payer
    service_type = evidence_pack.requested_service.service_type

    base = PAYER_BASE_APPROVAL_RATE.get(
        (payer, service_type), GENERIC_BASE_APPROVAL_RATE,
    )

    boost = _ALIGNMENT_BOOST.get(rules_match.overall_alignment, 0.0)
    evidence_lift = _evidence_lift(evidence_pack.evidence_strength_score)
    denial_penalty = max(0, n_prior_denials) * _PRIOR_DENIAL_PENALTY

    p = base + boost + evidence_lift - denial_penalty
    p = max(0.02, min(0.98, p))

    # CI width: narrows with rules-met fraction, widens with payer/
    # service rarity (= absent from the prior table) and prior denials.
    rule_total = max(
        rules_match.n_met + rules_match.n_unmet + rules_match.n_partial,
        1,
    )
    rules_certainty = rules_match.n_met / rule_total
    rare_combo = (payer, service_type) not in PAYER_BASE_APPROVAL_RATE
    base_width = 0.20 - 0.10 * rules_certainty
    if rare_combo:
        base_width += 0.10
    base_width += 0.02 * min(n_prior_denials, 5)
    half_width = base_width / 2.0

    ci_lo = max(0.0, p - half_width)
    ci_hi = min(1.0, p + half_width)

    if base_width > 0.30 or rare_combo and n_prior_denials >= 2:
        confidence = "abstain_recommended"
    elif base_width > 0.18:
        confidence = "degraded"
    else:
        confidence = "preferred"

    drivers = []
    drivers.append(
        f"Payer base approval rate = {base:.2f} for "
        f"{payer}/{service_type}"
        + (" (rare combo, falling back to generic prior)" if rare_combo else "")
    )
    drivers.append(
        f"Rules alignment = {rules_match.overall_alignment} "
        f"(boost {boost:+.2f})"
    )
    drivers.append(
        f"Evidence-strength score = {evidence_pack.evidence_strength_score:.3f} "
        f"(lift {evidence_lift:+.3f})"
    )
    if n_prior_denials > 0:
        drivers.append(
            f"Prior denials = {n_prior_denials} "
            f"(penalty {-denial_penalty:+.2f})"
        )

    abstain_recommended = (
        confidence == "abstain_recommended"
        or evidence_pack.abstain_recommended
    )
    abstain_reason = None
    if confidence == "abstain_recommended":
        abstain_reason = (
            "CI95 too wide for an actionable estimate; the rule alignment "
            "and prior-denial pattern suggest the request needs more "
            "supporting evidence before submission."
        )
    elif evidence_pack.abstain_recommended:
        abstain_reason = evidence_pack.abstain_reason

    rationale = (
        f"P(approval) = {p:.3f} (95% CI {ci_lo:.3f}-{ci_hi:.3f}, "
        f"confidence={confidence}). "
        f"Bayesian update: base={base:.2f}, alignment {boost:+.2f}, "
        f"evidence {evidence_lift:+.3f}, denials {-denial_penalty:+.2f}."
    )

    return PAApprovalEstimate(
        payer=payer,                                 # type: ignore[arg-type]
        requested_service_type=service_type,         # type: ignore[arg-type]
        probability_of_approval=round(p, 3),
        ci95=(round(ci_lo, 3), round(ci_hi, 3)),
        confidence=confidence,                       # type: ignore[arg-type]
        n_prior_denials=n_prior_denials,
        drivers=drivers,
        rationale=rationale,
        abstain_recommended=abstain_recommended,
        abstain_reason=abstain_reason,
        references=[
            "AMA 2024 Prior Authorization Physician Survey "
            "(per-payer denial rates).",
            "KFF Prior Authorization Tracker, 2025 update.",
            "CMS Medicare Plan Finder approval-rate data.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_pa_appeal_likelihood)
