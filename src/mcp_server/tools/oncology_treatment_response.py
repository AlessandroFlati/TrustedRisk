"""healthcare.compute_oncology_treatment_response -- RECIST 1.1 categorical response.

Inputs:
  target_lesions: list of {lesion_id, location, baseline_mm, current_mm}
  new_lesions_present: bool
  non_target_progression: bool

Output: TumorResponse with sum-change %, overall_response category
(CR/PR/SD/PD/NE), and a clinical decision implication (continue / dose
escalation / switch / palliative / repeat imaging).

RECIST 1.1 thresholds (Eisenhauer et al. 2009):
  CR -- disappearance of all lesions
  PR -- ≥30% decrease in sum of longest diameters
  PD -- ≥20% increase in sum (and ≥5mm absolute increase) OR new lesions
        OR unequivocal non-target progression
  SD -- neither PR nor PD
"""

from __future__ import annotations

from typing import Any

from shared.schemas import TargetLesion, TumorResponse


# ─────────────────────────────────────────────────────────────────────
# RECIST 1.1 classifier
# ─────────────────────────────────────────────────────────────────────

def _classify_recist(
    baseline_sum: float,
    current_sum: float,
    new_lesions: bool,
    non_target_pd: bool,
    all_lesions_disappeared: bool,
) -> str:
    if all_lesions_disappeared and not new_lesions and not non_target_pd:
        return "complete_response"
    if new_lesions or non_target_pd:
        return "progressive_disease"
    if baseline_sum <= 0:
        return "not_evaluable"
    pct_change = (current_sum - baseline_sum) / baseline_sum * 100.0
    abs_change_mm = current_sum - baseline_sum
    if pct_change <= -30.0:
        return "partial_response"
    if pct_change >= 20.0 and abs_change_mm >= 5.0:
        return "progressive_disease"
    return "stable_disease"


# ─────────────────────────────────────────────────────────────────────
# Decision implication (NCCN-style heuristic)
# ─────────────────────────────────────────────────────────────────────

def _decision_implication(response: str, prior_responses: list[str] | None
                            ) -> str:
    """Map response -> next clinical step.

    Considers prior responses to detect 'tolerating but not responding' ->
    'continue' (in early SD with good tolerance) vs 'switch' (in late SD
    after multiple cycles)."""
    prior = prior_responses or []
    if response == "complete_response":
        return "continue_current_therapy"
    if response == "partial_response":
        return "continue_current_therapy"
    if response == "stable_disease":
        # Repeated SD over many cycles -> consider switch
        if prior.count("stable_disease") >= 2:
            return "consider_dose_escalation"
        return "continue_current_therapy"
    if response == "progressive_disease":
        if prior.count("progressive_disease") >= 1:
            return "transition_palliative"
        return "switch_therapy"
    return "abstain_repeat_imaging"


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_oncology_treatment_response(
    target_lesions: list[dict[str, Any]],
    new_lesions_present: bool = False,
    non_target_progression: bool = False,
    prior_response_categories: list[str] | None = None,
    patient_id: str | None = None,
) -> TumorResponse:
    """Compute RECIST 1.1 response.

    Args:
        target_lesions: list of dicts with keys lesion_id, location,
            baseline_longest_diameter_mm, current_longest_diameter_mm.
        new_lesions_present: any new lesion appeared since baseline.
        non_target_progression: unequivocal progression in non-target lesions.
        prior_response_categories: optional sequence of past assessments
            (chronological), used to refine the decision implication.

    Returns:
        TumorResponse with overall_response + decision_implication.
    """
    parsed_lesions: list[TargetLesion] = []
    for L in target_lesions or []:
        try:
            parsed_lesions.append(TargetLesion.model_validate(L))
        except Exception:
            continue

    baseline_sum = sum(L.baseline_longest_diameter_mm for L in parsed_lesions)
    current_sum = sum(L.current_longest_diameter_mm for L in parsed_lesions)
    sum_change_pct = 0.0
    if baseline_sum > 0:
        sum_change_pct = (current_sum - baseline_sum) / baseline_sum * 100.0

    all_disappeared = (
        len(parsed_lesions) > 0
        and all(L.current_longest_diameter_mm == 0 for L in parsed_lesions)
    )

    response = _classify_recist(
        baseline_sum, current_sum, new_lesions_present,
        non_target_progression, all_disappeared,
    )

    decision = _decision_implication(response, prior_response_categories)

    abstain = response == "not_evaluable"
    abstain_reason = "no_evaluable_target_lesions" if abstain else None

    rationale = _build_rationale(
        parsed_lesions, baseline_sum, current_sum, sum_change_pct,
        new_lesions_present, non_target_progression, response, decision,
    )

    return TumorResponse(
        patient_id=patient_id,
        target_lesions=parsed_lesions,
        baseline_sum_mm=baseline_sum,
        current_sum_mm=current_sum,
        sum_change_pct=sum_change_pct,
        new_lesions_present=new_lesions_present,
        non_target_progression=non_target_progression,
        overall_response=response,  # type: ignore[arg-type]
        decision_implication=decision,  # type: ignore[arg-type]
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(lesions: list[TargetLesion], baseline: float,
                      current: float, pct: float, new_les: bool,
                      non_target_pd: bool, response: str, decision: str) -> str:
    parts = [f"{len(lesions)} target lesion(s): baseline ΣLD = {baseline:.1f} mm, "
             f"current ΣLD = {current:.1f} mm (Δ {pct:+.1f}%)."]
    if new_les:
        parts.append("New lesion(s) present.")
    if non_target_pd:
        parts.append("Unequivocal progression of non-target lesions.")
    parts.append(f"RECIST 1.1 classification: {response}.")
    parts.append(f"Decision implication: {decision}.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_oncology_treatment_response)
