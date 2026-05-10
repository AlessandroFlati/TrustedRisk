"""healthcare.compute_medication_adherence_predictor -- LIB-2.

Estimates 30-day post-discharge medication-adherence probability from
regimen complexity + demographic factors, surfaces the top contributing
factors, and recommends targeted interventions.

Model (deterministic, citation-tracked):
  - MRCI-lite regimen complexity (George 2004): pills/day + distinct dose
    times + side-effect burden + non-pill formulations.
  - Demographics + access (Osterberg/Blaschke 2005): age, polypharmacy
    threshold, insurance type, social-support proxy.
  - Logistic combination -> 30d adherence probability.

The model is a heuristic -- the value is in the structured intervention
list it produces, not the precise probability. Production deployments
should re-fit on local outcomes.
"""

from __future__ import annotations

import math
from typing import Any

from shared.schemas import (
    AdherenceFactor,
    AdherencePredictionReport,
)


_HIGH_BURDEN_DRUG_CLASSES = {
    "anticoagulant_vka", "insulin", "biguanide", "loop_diuretic",
    "antipsychotic", "ssri", "ace_inhibitor",
}


def _pills_per_day(med: dict) -> int:
    freq = (med.get("frequency") or "").lower()
    # Heuristic frequency parser
    if "qid" in freq or "4x" in freq or "four times" in freq:
        return 4
    if "tid" in freq or "3x" in freq or "three times" in freq:
        return 3
    if "bid" in freq or "2x" in freq or "twice" in freq:
        return 2
    if "qd" in freq or "daily" in freq or "once" in freq:
        return 1
    if "qhs" in freq or "bedtime" in freq:
        return 1
    if "weekly" in freq:
        return 0  # rough -- < 1 pill/day
    if "prn" in freq:
        return 0
    return 1


def _distinct_dose_times(meds: list[dict]) -> int:
    times: set[str] = set()
    for m in meds:
        f = (m.get("frequency") or "").lower()
        if "bid" in f or "twice" in f:
            times.update({"morning", "evening"})
        elif "tid" in f:
            times.update({"morning", "afternoon", "evening"})
        elif "qid" in f:
            times.update({"morning", "noon", "afternoon", "evening"})
        elif "qhs" in f or "bedtime" in f:
            times.add("bedtime")
        elif "qam" in f:
            times.add("morning")
        else:
            times.add("morning")
    return len(times)


def _regimen_complexity_index(meds: list[dict]) -> tuple[int, int, float]:
    """Return (pills_per_day, distinct_dose_times, complexity_index).

    Simplified MRCI: complexity = sum_per_med(formulation_weight + dose_freq_weight) +
    coordination_weight (distinct dose times).
    """
    if not meds:
        return 0, 0, 0.0

    total_pills_per_day = 0
    formulation_score = 0.0
    n_high_burden_class = 0
    for m in meds:
        ppd = _pills_per_day(m)
        total_pills_per_day += ppd
        formulation_score += 1.0
        # Tablet vs injection vs inhaler -- non-pill formulations harder
        form = (m.get("formulation") or "").lower()
        if "inject" in form or "subq" in form or "im" in form:
            formulation_score += 1.5
        elif "inhal" in form or "nebul" in form:
            formulation_score += 0.5
        cls = (m.get("drug_class") or "").lower()
        if cls in _HIGH_BURDEN_DRUG_CLASSES:
            n_high_burden_class += 1

    n_dose_times = _distinct_dose_times(meds)
    coordination = float(n_dose_times) * 0.5
    burden = float(n_high_burden_class) * 0.4
    complexity = formulation_score + coordination + burden \
        + total_pills_per_day * 0.2
    return total_pills_per_day, n_dose_times, round(complexity, 2)


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


async def compute_medication_adherence_predictor(
    medications: list[dict],
    patient_age: int | None = None,
    insurance_type: str | None = None,
    has_caregiver: bool = False,
    prior_adherence_known: bool | None = None,
) -> AdherencePredictionReport:
    """Estimate 30-day medication-adherence probability + interventions.

    Args:
        medications: list of dicts with `name`, `drug_class`, `frequency`,
            optionally `formulation`.
        patient_age: drives age-related risk (Osterberg-Blaschke 2005).
        insurance_type: medicare / medicaid / commercial / self_pay.
        has_caregiver: caregiver presence boosts adherence.
        prior_adherence_known: when True, prior non-adherence dominates
            the prediction (history is the strongest predictor).

    Returns:
        AdherencePredictionReport with probability + factors + intervention list.
    """
    if not isinstance(medications, list):
        raise ValueError("medications must be a list of dicts.")

    # Special case: no meds -> no adherence concern
    if not medications:
        return AdherencePredictionReport(
            n_medications=0, pills_per_day=0, distinct_dose_times=0,
            regimen_complexity_index=0.0,
            adherence_30d_probability=1.0,
            risk_tier="low",
            contributing_factors=[],
            interventions_recommended=[
                "No medications prescribed at discharge -- no adherence "
                "intervention needed."
            ],
            rationale="Empty medication list -- adherence not applicable.",
        )

    pills, n_doses, complexity = _regimen_complexity_index(medications)

    factors: list[AdherenceFactor] = []
    score = 0.0
    base = 1.0   # log-odds baseline -> ~73% adherence

    # Polypharmacy threshold (Osterberg 2005: ≥ 5 meds)
    n_meds = len(medications)
    if n_meds >= 10:
        factors.append(AdherenceFactor(
            name="polypharmacy_severe",
            direction="negative", weight=-1.2,
            detail=f"{n_meds} active medications"))
        score -= 1.2
    elif n_meds >= 5:
        factors.append(AdherenceFactor(
            name="polypharmacy_moderate",
            direction="negative", weight=-0.6,
            detail=f"{n_meds} active medications"))
        score -= 0.6

    # Distinct dose times
    if n_doses >= 4:
        factors.append(AdherenceFactor(
            name="dose_schedule_complexity",
            direction="negative", weight=-0.7,
            detail=f"{n_doses} distinct dose times per day"))
        score -= 0.7
    elif n_doses >= 3:
        factors.append(AdherenceFactor(
            name="dose_schedule_moderate",
            direction="negative", weight=-0.3,
            detail=f"{n_doses} distinct dose times per day"))
        score -= 0.3

    # Age
    if patient_age is not None:
        if patient_age >= 75:
            factors.append(AdherenceFactor(
                name="advanced_age",
                direction="negative", weight=-0.4,
                detail=f"Age {patient_age}; cognitive + dexterity factors"))
            score -= 0.4
        elif patient_age <= 30:
            factors.append(AdherenceFactor(
                name="young_age",
                direction="negative", weight=-0.3,
                detail="Age ≤ 30; lower routine adherence baseline"))
            score -= 0.3

    # Insurance
    if insurance_type == "medicaid" or insurance_type == "self_pay":
        factors.append(AdherenceFactor(
            name="financial_barrier",
            direction="negative", weight=-0.5,
            detail=f"Insurance: {insurance_type} -- copay barriers"))
        score -= 0.5

    # Caregiver
    if has_caregiver:
        factors.append(AdherenceFactor(
            name="caregiver_present",
            direction="positive", weight=0.6,
            detail="Caregiver supervision documented"))
        score += 0.6

    # Prior history (dominant predictor)
    if prior_adherence_known is False:
        factors.append(AdherenceFactor(
            name="prior_non_adherence",
            direction="negative", weight=-1.5,
            detail="Documented history of non-adherence -- strongest "
                      "predictor (Osterberg 2005)"))
        score -= 1.5
    elif prior_adherence_known is True:
        factors.append(AdherenceFactor(
            name="prior_adherence",
            direction="positive", weight=0.4,
            detail="Documented history of adherence"))
        score += 0.4

    # High-burden drug-class share
    n_high_burden = sum(
        1 for m in medications
        if (m.get("drug_class") or "").lower() in _HIGH_BURDEN_DRUG_CLASSES
    )
    if n_high_burden >= 2:
        factors.append(AdherenceFactor(
            name="high_burden_drug_classes",
            direction="negative", weight=-0.4,
            detail=f"{n_high_burden} high-burden classes "
                      "(VKA, insulin, biguanide, etc.)"))
        score -= 0.4

    prob = _logistic(base + score)
    if prob >= 0.75:
        tier = "low"
    elif prob >= 0.55:
        tier = "moderate"
    else:
        tier = "high"

    # Targeted interventions
    interventions: list[str] = []
    if pills >= 5 or n_doses >= 3:
        interventions.append(
            "Pill organizer or blister-pack dispensing (reduces "
            "complexity-driven errors)."
        )
    if n_doses >= 4:
        interventions.append(
            "Daily dose simplification: ask prescribers about "
            "extended-release / combination formulations."
        )
    if patient_age is not None and patient_age >= 75:
        interventions.append(
            "Brown-bag medication review at first follow-up; consider "
            "geriatric pharmacy consult."
        )
    if insurance_type in ("medicaid", "self_pay"):
        interventions.append(
            "Generic substitution review + 340B / patient-assistance "
            "program enrollment."
        )
    if not has_caregiver and tier in ("moderate", "high"):
        interventions.append(
            "Referral to home-health aide or pharmacy adherence-monitoring "
            "service."
        )
    if prior_adherence_known is False:
        interventions.append(
            "Motivational interviewing at follow-up; identify root cause "
            "(side effects vs forgetting vs cost)."
        )
    if not interventions:
        interventions.append(
            "Standard discharge counseling; no targeted intervention "
            "indicated."
        )

    rationale = (
        f"Complexity index {complexity}, {pills} pills/day across "
        f"{n_doses} dose times, {n_meds} total meds. "
        f"Predicted 30-day adherence: {prob:.2f} ({tier}-risk tier)."
    )

    return AdherencePredictionReport(
        n_medications=n_meds,
        pills_per_day=pills,
        distinct_dose_times=n_doses,
        regimen_complexity_index=complexity,
        adherence_30d_probability=round(prob, 4),
        risk_tier=tier,                    # type: ignore[arg-type]
        contributing_factors=factors,
        interventions_recommended=interventions,
        rationale=rationale,
    )


def register(mcp) -> None:
    mcp.tool()(compute_medication_adherence_predictor)
