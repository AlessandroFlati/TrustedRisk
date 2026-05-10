"""healthcare.compute_expected_value_of_intervention -- IMPACT-1.

Health-economics evaluation of a proposed clinical intervention against a
baseline event probability. The tool consumes either an explicit baseline
risk or a `RiskEstimate` produced by another tool (e.g. compute_readmission_risk),
and an `InterventionEvidence` object encoding the published effect estimate
and cost.

Outputs an ICER-style cost-effectiveness report (NNT, expected events
avoided, total cost, $/event-avoided, $/QALY) plus a categorical decision
following the standard CEA ladder (cost-saving -> cost-effective -> not),
modulated by the evidence grade so weak-evidence inputs surface as
`uncertain_evidence` rather than a false-confident verdict.

References (also embedded in the schema):
  Sanders GD, et al. Second Panel CEA recommendations. JAMA 2016;316:1093.
  Neumann PJ, et al. The $50k-per-QALY threshold revisited. NEJM 2014;371:796.
"""

from __future__ import annotations

from typing import Any

from shared.schemas import (
    CostEffectivenessReport,
    InterventionEvidence,
    RiskEstimate,
)


def _resolve_baseline(
    baseline_event_probability: float | None,
    risk_estimate: RiskEstimate | None,
) -> float:
    if baseline_event_probability is not None:
        if not (0.0 <= baseline_event_probability <= 1.0):
            raise ValueError(
                "baseline_event_probability must be in [0, 1], got "
                f"{baseline_event_probability}"
            )
        return float(baseline_event_probability)
    if risk_estimate is not None:
        return float(risk_estimate.probability_mean)
    raise ValueError(
        "Either baseline_event_probability or risk_estimate must be provided."
    )


def _resolve_arr(intervention: InterventionEvidence,
                  baseline_p: float) -> float:
    """Return absolute risk reduction (ARR), preferring the explicit ARR
    field when both ARR and RRR are provided."""
    if intervention.absolute_risk_reduction is not None:
        arr = float(intervention.absolute_risk_reduction)
    elif intervention.relative_risk_reduction is not None:
        # ARR = baseline × RRR (RRR is the proportional reduction)
        arr = float(intervention.relative_risk_reduction) * baseline_p
    else:
        raise ValueError(
            "InterventionEvidence must specify either relative_risk_reduction "
            "or absolute_risk_reduction."
        )
    # Clamp to [0, baseline_p] -- ARR cannot exceed the baseline probability
    return max(0.0, min(arr, baseline_p))


def _classify_decision(
    evidence_grade: str,
    net_cost_total: float,
    expected_events_avoided: float,
    cost_per_qaly: float | None,
    wtp_threshold: float,
) -> tuple[str, str]:
    """Return (decision, rationale_clause)."""
    # Weak evidence -- block a confident verdict regardless of point estimates
    if evidence_grade in ("C", "D", "expert_opinion"):
        return (
            "uncertain_evidence",
            f"evidence_grade={evidence_grade} is too weak to support a "
            "cost-effectiveness verdict; treat point estimates as illustrative.",
        )
    if expected_events_avoided <= 0:
        return (
            "dominated",
            "Effect estimate yields zero events avoided after clamping; the "
            "intervention is dominated by 'no action'.",
        )
    if net_cost_total < 0:
        return (
            "cost_saving",
            f"Avoided event costs (${abs(net_cost_total):,.0f}) exceed total "
            "intervention cost -- the intervention is cost-saving.",
        )
    if cost_per_qaly is None:
        # No QALY data -- fall back to "informational" categorization
        # by reporting cost-effective only when the per-event cost is
        # below the WTP threshold (a coarser proxy).
        return (
            "cost_effective" if net_cost_total / expected_events_avoided
                                  < wtp_threshold else "not_cost_effective",
            "QALY gain not provided -- using cost-per-event-avoided "
            f"vs WTP {wtp_threshold:,.0f}/QALY as a proxy.",
        )
    if cost_per_qaly < wtp_threshold:
        return (
            "cost_effective",
            f"ICER ${cost_per_qaly:,.0f}/QALY is below the WTP threshold of "
            f"${wtp_threshold:,.0f}/QALY.",
        )
    return (
        "not_cost_effective",
        f"ICER ${cost_per_qaly:,.0f}/QALY is at or above the WTP threshold of "
        f"${wtp_threshold:,.0f}/QALY.",
    )


async def compute_expected_value_of_intervention(
    intervention: InterventionEvidence | dict[str, Any],
    baseline_event_probability: float | None = None,
    risk_estimate: RiskEstimate | dict[str, Any] | None = None,
    cohort_size: int = 100,
    avoided_event_cost_usd: float = 14_000.0,
    wtp_threshold_per_qaly_usd: float = 100_000.0,
) -> CostEffectivenessReport:
    """Compute the expected economic value of applying `intervention` to a cohort.

    Args:
        intervention: evidence-based effect estimate + cost (an
            InterventionEvidence or its dict equivalent).
        baseline_event_probability: per-patient baseline probability of the
            adverse event the intervention prevents. Required if `risk_estimate`
            is not provided.
        risk_estimate: alternative input -- pulls `probability_mean` as the
            baseline. Useful for chaining after compute_readmission_risk.
        cohort_size: number of patients receiving the intervention. Default 100
            so that absolute totals scale to a familiar denominator.
        avoided_event_cost_usd: cost per avoided event (USD). Default 14000 ≈
            average direct cost of a 30-day readmission in US Medicare data
            (HCUP 2018, AHRQ Statistical Brief #248).
        wtp_threshold_per_qaly_usd: willingness-to-pay threshold for ICER
            classification. Default 100000 -- common benchmark in the US,
            between the 50k legacy and 150k contemporary Sanders recommendations.

    Returns:
        CostEffectivenessReport with ARR, NNT, expected events avoided, totals,
        ICER (when QALY supplied), decision, and rationale.
    """
    # Coerce dict inputs to schema instances (MCP serializes inputs as dicts)
    if isinstance(intervention, dict):
        intervention = InterventionEvidence.model_validate(intervention)
    if isinstance(risk_estimate, dict):
        risk_estimate = RiskEstimate.model_validate(risk_estimate)

    if cohort_size <= 0:
        raise ValueError("cohort_size must be a positive integer.")
    if avoided_event_cost_usd < 0:
        raise ValueError("avoided_event_cost_usd must be non-negative.")
    if wtp_threshold_per_qaly_usd < 0:
        raise ValueError("wtp_threshold_per_qaly_usd must be non-negative.")

    baseline_p = _resolve_baseline(baseline_event_probability, risk_estimate)
    arr = _resolve_arr(intervention, baseline_p)

    expected_events_avoided = cohort_size * arr
    intervention_cost_total = cohort_size * float(intervention.cost_per_patient_usd)
    avoided_event_cost_total = expected_events_avoided * float(avoided_event_cost_usd)
    net_cost_total = intervention_cost_total - avoided_event_cost_total

    nnt: float | None = None
    if arr > 0:
        nnt = 1.0 / arr

    cost_per_event_avoided: float | None = None
    if expected_events_avoided > 0:
        cost_per_event_avoided = intervention_cost_total / expected_events_avoided

    cost_per_qaly: float | None = None
    if intervention.qaly_gained_per_avoided_event is not None:
        total_qaly = expected_events_avoided * float(
            intervention.qaly_gained_per_avoided_event)
        if total_qaly > 0:
            cost_per_qaly = net_cost_total / total_qaly

    decision, decision_clause = _classify_decision(
        evidence_grade=intervention.evidence_grade,
        net_cost_total=net_cost_total,
        expected_events_avoided=expected_events_avoided,
        cost_per_qaly=cost_per_qaly,
        wtp_threshold=wtp_threshold_per_qaly_usd,
    )

    abstain = False
    abstain_reason: str | None = None
    if intervention.evidence_grade in ("D", "expert_opinion"):
        abstain = True
        abstain_reason = (
            "Effect estimate is based on weak evidence -- defer to clinician "
            "judgement rather than acting on the modeled cost-effectiveness."
        )

    rationale = (
        f"{intervention.name}: baseline event probability {baseline_p:.3f}, "
        f"ARR {arr:.3f} -> NNT {nnt:.1f}." if nnt is not None else
        f"{intervention.name}: baseline event probability {baseline_p:.3f}, "
        "ARR 0 (no expected effect)."
    )
    rationale += f" Over {cohort_size} patients: "
    rationale += (
        f"expected events avoided {expected_events_avoided:.2f}, "
        f"intervention cost ${intervention_cost_total:,.0f}, "
        f"avoided event cost ${avoided_event_cost_total:,.0f}, "
        f"net ${net_cost_total:,.0f}. "
    )
    rationale += decision_clause

    return CostEffectivenessReport(
        intervention_name=intervention.name,
        cohort_size=cohort_size,
        baseline_event_probability=baseline_p,
        absolute_risk_reduction=arr,
        number_needed_to_treat=nnt,
        expected_events_avoided=expected_events_avoided,
        intervention_cost_total_usd=intervention_cost_total,
        avoided_event_cost_total_usd=avoided_event_cost_total,
        net_cost_total_usd=net_cost_total,
        cost_per_event_avoided_usd=cost_per_event_avoided,
        cost_per_qaly_usd=cost_per_qaly,
        wtp_threshold_per_qaly_usd=wtp_threshold_per_qaly_usd,
        decision=decision,  # type: ignore[arg-type]
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_expected_value_of_intervention)
