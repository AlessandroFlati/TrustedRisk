"""healthcare.compute_contrast_safety_check -- pre-imaging contrast gate.

Per ACR Manual on Contrast Media (2024) + ACR/NKF consensus 2020 (Davenport
et al.):
  - Iodinated IV contrast: AKI risk increases at eGFR <30, but for AKI alone
    the risk is now considered SMALLER than previously thought (the term has
    shifted from "contrast-induced nephropathy" to "post-contrast AKI" or
    "contrast-associated AKI" -- many cases are coincidental).
  - Metformin: hold for 48h after iodinated contrast IF eGFR <60 OR if AKI
    develops; otherwise no hold needed.
  - Iodine "allergy" is a misnomer -- it's a contrast-media reaction. Prior
    severe reaction -> premedicate with corticosteroids + diphenhydramine
    (12-13h, 6-7h, 1h before scan + 50 mg PO/IV).
  - Gadolinium: NSF risk extremely low for Group II agents (gadobutrol,
    gadoteridol, gadoterate); avoid Group I (gadodiamide, gadopentetate,
    gadoversetamide) at eGFR <30.
  - Pregnancy: iodinated contrast crosses placenta but no documented harm;
    use only when benefit outweighs risk. Gadolinium NOT recommended in
    pregnancy.
"""

from __future__ import annotations

from typing import Any

from shared.schemas import ContrastSafetyReport


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_contrast_safety_check(
    contrast_type: str,
    egfr_ml_min: float | None = None,
    on_metformin: bool = False,
    iodine_contrast_prior_severe_reaction: bool = False,
    iodine_contrast_prior_mild_reaction: bool = False,
    pregnant: bool = False,
    patient_id: str | None = None,
) -> ContrastSafetyReport:
    """Decide whether to proceed with a contrast study and what mitigations.

    Args:
        contrast_type: iodinated_iv / gadolinium_iv / oral / no_contrast.
        egfr_ml_min: most recent eGFR. Drives nephropathy + Mg hold + NSF.
        on_metformin: drives metformin hold logic.
        iodine_contrast_prior_severe_reaction: prior anaphylaxis-like.
            -> premedicate.
        iodine_contrast_prior_mild_reaction: prior mild urticaria.
            -> premedicate (less aggressive).
        pregnant: -> caution / abstain for gadolinium.

    Returns:
        ContrastSafetyReport with proceed_with_contrast + mitigation steps.
    """
    ct = contrast_type.strip().lower()
    if ct not in ("iodinated_iv", "gadolinium_iv", "oral", "no_contrast"):
        ct = "iodinated_iv"

    # CIN / contrast-associated AKI risk
    if ct == "iodinated_iv":
        if egfr_ml_min is None:
            ci_risk = "moderate"  # unknown -- be cautious
        elif egfr_ml_min >= 60:
            ci_risk = "low"
        elif egfr_ml_min >= 45:
            ci_risk = "moderate"
        elif egfr_ml_min >= 30:
            ci_risk = "high"
        else:
            ci_risk = "contraindicated"
    else:
        ci_risk = "low"

    # NSF risk for gadolinium
    if ct == "gadolinium_iv":
        if egfr_ml_min is None or egfr_ml_min >= 30:
            nsf = "low"
        elif egfr_ml_min >= 15:
            nsf = "moderate"
        else:
            nsf = "contraindicated"
    else:
        nsf = "n_a"

    # Metformin hold logic
    metformin_hold = False
    metformin_hold_h = 0
    if on_metformin and ct == "iodinated_iv":
        if egfr_ml_min is None or egfr_ml_min < 60:
            metformin_hold = True
            metformin_hold_h = 48

    # Premedication
    pre_med = bool(iodine_contrast_prior_severe_reaction
                     or iodine_contrast_prior_mild_reaction)

    # Pre-hydration
    pre_hydrate = ct == "iodinated_iv" and egfr_ml_min is not None and egfr_ml_min < 45

    # Pregnancy caution
    pregnancy_caution = pregnant and ct in ("iodinated_iv", "gadolinium_iv")

    # Final proceed decision
    if ct == "no_contrast":
        proceed = True
    elif ci_risk == "contraindicated" or nsf == "contraindicated":
        proceed = False
    elif pregnant and ct == "gadolinium_iv":
        proceed = False
    else:
        proceed = True

    abstain = (not proceed)
    abstain_reason = None
    if abstain:
        if ci_risk == "contraindicated":
            abstain_reason = (f"eGFR {egfr_ml_min} mL/min -- iodinated contrast "
                                f"contraindicated below 30 except in life-threatening indication")
        elif nsf == "contraindicated":
            abstain_reason = (f"eGFR {egfr_ml_min} mL/min -- gadolinium contraindicated "
                                f"below 15 (NSF risk)")
        elif pregnant:
            abstain_reason = "gadolinium NOT recommended in pregnancy"

    rationale = _build_rationale(ct, egfr_ml_min, ci_risk, nsf, metformin_hold,
                                   pre_med, pre_hydrate, pregnancy_caution,
                                   proceed)

    return ContrastSafetyReport(
        patient_id=patient_id,
        contrast_type=ct,  # type: ignore[arg-type]
        egfr_ml_min=egfr_ml_min,
        contrast_induced_nephropathy_risk=ci_risk,  # type: ignore[arg-type]
        nsf_risk_for_gadolinium=nsf,  # type: ignore[arg-type]
        metformin_hold_recommended=metformin_hold,
        metformin_hold_duration_hours=metformin_hold_h,
        iodine_allergy_documented=(iodine_contrast_prior_severe_reaction
                                       or iodine_contrast_prior_mild_reaction),
        premedication_recommended=pre_med,
        pregnancy_caution=pregnancy_caution,
        pre_hydration_recommended=pre_hydrate,
        proceed_with_contrast=proceed,
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(ct: str, egfr: float | None, ci: str, nsf: str,
                       met_hold: bool, premed: bool, prehyd: bool,
                       preg_caution: bool, proceed: bool) -> str:
    parts = [f"Contrast type: {ct}, eGFR: {egfr if egfr is not None else 'unknown'} mL/min."]
    parts.append(f"Contrast-associated AKI risk: {ci}.")
    if ct == "gadolinium_iv":
        parts.append(f"NSF risk: {nsf}.")
    if met_hold:
        parts.append(f"Metformin: HOLD for 48h post-contrast (eGFR < 60).")
    if premed:
        parts.append(
            "Premedication regimen: prednisone 50 mg PO at 13h + 7h + 1h "
            "before scan, plus diphenhydramine 50 mg IV/PO 1h before."
        )
    if prehyd:
        parts.append(
            "Pre-hydration recommended (0.9% NS 1 mL/kg/h × 12h pre + post)."
        )
    if preg_caution:
        parts.append("Pregnancy: weigh benefit/risk; gadolinium not recommended.")
    parts.append(f"Proceed with contrast: {'YES' if proceed else 'NO'}.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_contrast_safety_check)
