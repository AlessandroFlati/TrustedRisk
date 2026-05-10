"""healthcare.compute_inpatient_glycemic_control -- basal-bolus titration.

ADA Standards 2024 + Endocrine Society 2022 inpatient targets:
  Critically ill (ICU)              : 140-180 mg/dL
  Non-critical ward (most patients) : 100-180 mg/dL (BG before meals/HS)

Titration rule of thumb:
  - Average BG > 180 mg/dL with no hypoglycemia -> ↑ basal 10-20%
  - Average BG > 240 mg/dL -> ↑ basal 20%, intensify correctional scale
  - Any hypoglycemia (<70) -> ↓ basal 10-20%
  - Severe hypoglycemia (<54) -> ↓ basal 20-30%, evaluate why
  - Recurrent hypoglycemia -> switch from sliding-scale-only to basal-bolus

Hypoglycemia risk factors (each ↑risk):
  age ≥75, eGFR <60, recent NPO / changed diet, heart failure, sulfonylurea
  use, sepsis, glucocorticoid recent taper, malnutrition.
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import InpatientGlycemicPlan


# ─────────────────────────────────────────────────────────────────────
# Hypoglycemia risk
# ─────────────────────────────────────────────────────────────────────

def _hypoglycemia_risk(factors: dict[str, Any]) -> str:
    score = 0
    age = factors.get("age")
    if isinstance(age, (int, float)) and age >= 75:
        score += 1
    egfr = factors.get("egfr_ml_min")
    if isinstance(egfr, (int, float)) and egfr < 60:
        score += 1
    if factors.get("recent_npo_or_diet_change"):
        score += 1
    if factors.get("heart_failure"):
        score += 1
    if factors.get("sulfonylurea_use"):
        score += 1
    if factors.get("sepsis"):
        score += 1
    if factors.get("recent_steroid_taper"):
        score += 1
    if factors.get("malnutrition"):
        score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_inpatient_glycemic_control(
    is_icu: bool = False,
    average_glucose_24h: float | None = None,
    n_hypoglycemic_episodes_24h: int = 0,
    n_severe_hyperglycemic_episodes_24h: int = 0,
    current_regimen: str = "sliding_scale_only",
    current_basal_total_units: float = 0.0,
    risk_factors: dict[str, Any] | None = None,
    patient_id: str | None = None,
) -> InpatientGlycemicPlan:
    """Recommend a basal-bolus titration step + correctional-scale adjustment.

    Args:
        is_icu: True -> target 140-180; False -> 100-180.
        average_glucose_24h: rolling average over 24h (mg/dL).
        n_hypoglycemic_episodes_24h: # BG <70 in last 24h.
        n_severe_hyperglycemic_episodes_24h: # BG >300 in last 24h.
        current_regimen: free-text of current insulin regimen.
        current_basal_total_units: current total basal dose in units.
        risk_factors: dict of hypoglycemia risk factors (see `_hypoglycemia_risk`).

    Returns:
        InpatientGlycemicPlan.
    """
    target = (140, 180) if is_icu else (100, 180)
    risk = _hypoglycemia_risk(risk_factors or {})

    abstain = False
    abstain_reason: str | None = None
    basal_change_pct = 0.0
    correctional_change: str | None = None
    rec_regimen = current_regimen

    if n_hypoglycemic_episodes_24h >= 1:
        # Any hypoglycemia trumps hyperglycemia -- pull basal down
        if n_hypoglycemic_episodes_24h >= 2:
            basal_change_pct = -25.0
            correctional_change = "switch to LESS aggressive correctional scale"
            rec_regimen = (
                "Reduce basal insulin by 25% from "
                f"{current_basal_total_units} U total. Reassess in 12h. "
                f"If hypoglycemia recurs, hold all sliding-scale doses and "
                f"defer to endocrine consult."
            )
        else:
            basal_change_pct = -15.0
            rec_regimen = (
                "Reduce basal insulin by 10-20% (recommend 15%) from "
                f"{current_basal_total_units} U total."
            )
    elif (average_glucose_24h is not None and average_glucose_24h > 240
            and n_severe_hyperglycemic_episodes_24h >= 1):
        basal_change_pct = 25.0
        correctional_change = "intensify correctional scale (move to next-tier scale)"
        rec_regimen = (
            f"Increase basal by 25% from {current_basal_total_units} U total. "
            f"Move from medium to high correctional scale. Add prandial insulin "
            f"(0.05-0.1 U/kg/meal) if not already on basal-bolus."
        )
    elif (average_glucose_24h is not None and average_glucose_24h > target[1]):
        basal_change_pct = 15.0
        rec_regimen = (
            f"Increase basal by 10-20% (recommend 15%) from "
            f"{current_basal_total_units} U total."
        )
    elif (average_glucose_24h is not None and average_glucose_24h < target[0]):
        basal_change_pct = -10.0
        rec_regimen = (
            f"Decrease basal by 10% from {current_basal_total_units} U total. "
            f"Mean glucose below target range -- over-treatment risk."
        )
    else:
        rec_regimen = (
            f"Maintain current regimen ({current_regimen}). Mean glucose "
            f"in target {target[0]}-{target[1]} mg/dL."
        )

    if "sliding_scale_only" in current_regimen.lower() and basal_change_pct > 0:
        rec_regimen = (
            "TRANSITION from sliding-scale-only to BASAL-BOLUS regimen. "
            "Sliding-scale-only is associated with increased glycemic "
            "variability + hypoglycemia per ADA 2024. " + rec_regimen
        )

    # If high hypoglycemia risk, narrow target band
    if risk == "high":
        target = (140, 180)
        abstain_reason = (
            "high hypoglycemia risk: defer titration up; consider endocrine "
            "consult before changing basal dose."
        )
        if basal_change_pct > 0:
            abstain = True

    rationale = _build_rationale(target, average_glucose_24h,
                                   n_hypoglycemic_episodes_24h,
                                   n_severe_hyperglycemic_episodes_24h,
                                   basal_change_pct, risk, rec_regimen)

    return InpatientGlycemicPlan(
        patient_id=patient_id,
        target_range_mg_dl=target,
        average_glucose_24h=average_glucose_24h,
        n_hypoglycemic_episodes_24h=n_hypoglycemic_episodes_24h,
        n_severe_hyperglycemic_episodes_24h=n_severe_hyperglycemic_episodes_24h,
        current_regimen=current_regimen,
        recommended_regimen=rec_regimen,
        basal_dose_change_pct=basal_change_pct,
        correctional_scale_change=correctional_change,
        hypoglycemia_risk=risk,  # type: ignore[arg-type]
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(target: tuple[int, int], avg: float | None, hypo: int,
                       hyper: int, change: float, risk: str, regimen: str) -> str:
    parts = [
        f"Target range {target[0]}-{target[1]} mg/dL.",
        f"24h: avg={avg if avg is not None else 'n/a'}, {hypo} hypo episode(s), "
        f"{hyper} severe hyper episode(s).",
        f"Hypoglycemia risk: {risk}.",
        f"Basal change: {change:+.0f}%.",
    ]
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_inpatient_glycemic_control)
