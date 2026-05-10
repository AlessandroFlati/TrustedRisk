"""healthcare.compute_imaging_appropriateness -- ACR criteria + Choosing Wisely.

Per ACR Appropriateness Criteria, each clinical scenario maps to a ranked
list of imaging modalities with appropriateness rating 1-9:
  1-3  usually NOT appropriate
  4-6  may be appropriate
  7-9  usually appropriate

This tool implements the most-frequent ED scenarios as a curated lookup:
  - chest_pain_low_risk
  - chest_pain_acs_workup
  - acute_abdominal_pain
  - acute_lbp_no_red_flag
  - acute_headache_thunderclap
  - suspected_pe_low_risk
  - suspected_pe_high_risk
  - acute_stroke
  - blunt_abdominal_trauma
  - pediatric_appendicitis

Other scenarios return abstain_recommended=true.

Radiation dose estimates (ICRP 103 effective dose, mSv):
  CXR ~0.1, CT chest ~7, CT chest+abdomen+pelvis ~14, CT-PA ~15,
  V/Q ~2.5, CT head ~2, MRI 0, US 0, MRA brain 0, X-ray spine ~1.5
"""

from __future__ import annotations

from typing import Any

from shared.schemas import ImagingOption, ImagingAppropriateness


# ─────────────────────────────────────────────────────────────────────
# Curated scenario -> modality table
# ─────────────────────────────────────────────────────────────────────

_SCENARIO_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "chest_pain_acs_workup": [
        {"modality": "ECG (NOT imaging -- preferred initial step)",
         "rating": 9, "dose": 0.0, "contrast": False, "cost": "low"},
        {"modality": "CXR portable", "rating": 7, "dose": 0.1,
         "contrast": False, "cost": "low"},
        {"modality": "CT coronary angiography",
         "rating": 7, "dose": 5.0, "contrast": True, "cost": "high"},
        {"modality": "Coronary angiography (cath)",
         "rating": 9, "dose": 7.0, "contrast": True, "cost": "high"},
        {"modality": "CT pulmonary angiography",
         "rating": 4, "dose": 15.0, "contrast": True, "cost": "high"},
    ],
    "suspected_pe_low_risk": [
        {"modality": "D-dimer (lab -- preferred first step)",
         "rating": 9, "dose": 0.0, "contrast": False, "cost": "low"},
        {"modality": "CT pulmonary angiography",
         "rating": 4, "dose": 15.0, "contrast": True, "cost": "high"},
        {"modality": "V/Q scan", "rating": 5, "dose": 2.5,
         "contrast": False, "cost": "medium"},
    ],
    "suspected_pe_high_risk": [
        {"modality": "CT pulmonary angiography",
         "rating": 9, "dose": 15.0, "contrast": True, "cost": "high"},
        {"modality": "V/Q scan (if CT-PA contraindicated)",
         "rating": 7, "dose": 2.5, "contrast": False, "cost": "medium"},
        {"modality": "Lower-extremity venous duplex",
         "rating": 6, "dose": 0.0, "contrast": False, "cost": "low"},
    ],
    "acute_abdominal_pain": [
        {"modality": "CT abdomen+pelvis with IV contrast",
         "rating": 8, "dose": 14.0, "contrast": True, "cost": "high"},
        {"modality": "Ultrasound RUQ (if biliary suspected)",
         "rating": 8, "dose": 0.0, "contrast": False, "cost": "low"},
        {"modality": "Plain abdominal X-ray (acute series)",
         "rating": 4, "dose": 0.7, "contrast": False, "cost": "low"},
        {"modality": "MRI abdomen (if pregnant or contrast-contraindicated)",
         "rating": 6, "dose": 0.0, "contrast": False, "cost": "high"},
    ],
    "acute_lbp_no_red_flag": [
        {"modality": "No imaging -- supportive care + reassessment in 6 weeks",
         "rating": 9, "dose": 0.0, "contrast": False, "cost": "low"},
        {"modality": "Lumbar spine X-ray", "rating": 3, "dose": 1.5,
         "contrast": False, "cost": "low"},
        {"modality": "MRI lumbar spine", "rating": 2,
         "dose": 0.0, "contrast": False, "cost": "high"},
        {"modality": "CT lumbar spine", "rating": 2, "dose": 6.0,
         "contrast": False, "cost": "high"},
    ],
    "acute_headache_thunderclap": [
        {"modality": "Non-contrast CT head", "rating": 9, "dose": 2.0,
         "contrast": False, "cost": "medium"},
        {"modality": "CT angiography head + neck",
         "rating": 8, "dose": 6.0, "contrast": True, "cost": "high"},
        {"modality": "Lumbar puncture (if CT negative)",
         "rating": 8, "dose": 0.0, "contrast": False, "cost": "medium"},
        {"modality": "MRI brain", "rating": 5, "dose": 0.0,
         "contrast": False, "cost": "high"},
    ],
    "acute_stroke": [
        {"modality": "Non-contrast CT head (rule out hemorrhage)",
         "rating": 9, "dose": 2.0, "contrast": False, "cost": "medium"},
        {"modality": "CT angiography head + neck",
         "rating": 9, "dose": 6.0, "contrast": True, "cost": "high"},
        {"modality": "CT perfusion (for late-window EVT decision)",
         "rating": 8, "dose": 5.0, "contrast": True, "cost": "high"},
        {"modality": "MRI brain with diffusion",
         "rating": 7, "dose": 0.0, "contrast": False, "cost": "high"},
    ],
    "blunt_abdominal_trauma": [
        {"modality": "FAST exam (bedside ultrasound)",
         "rating": 9, "dose": 0.0, "contrast": False, "cost": "low"},
        {"modality": "CT abdomen+pelvis with IV contrast",
         "rating": 9, "dose": 14.0, "contrast": True, "cost": "high"},
        {"modality": "Diagnostic peritoneal lavage (rare)",
         "rating": 3, "dose": 0.0, "contrast": False, "cost": "low"},
    ],
    "pediatric_appendicitis": [
        {"modality": "Ultrasound (graded compression)",
         "rating": 9, "dose": 0.0, "contrast": False, "cost": "low"},
        {"modality": "MRI (if US equivocal)",
         "rating": 8, "dose": 0.0, "contrast": False, "cost": "high"},
        {"modality": "CT abdomen+pelvis",
         "rating": 5, "dose": 14.0, "contrast": True, "cost": "high"},
    ],
    "chest_pain_low_risk": [
        {"modality": "ECG", "rating": 9, "dose": 0.0,
         "contrast": False, "cost": "low"},
        {"modality": "CXR", "rating": 6, "dose": 0.1,
         "contrast": False, "cost": "low"},
        {"modality": "CT coronary angiography (if HEART borderline)",
         "rating": 6, "dose": 5.0, "contrast": True, "cost": "high"},
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_imaging_appropriateness(
    clinical_scenario: str,
    patient_age: int | None = None,
    patient_pregnant: bool = False,
    cumulative_radiation_msv_last_12mo: float = 0.0,
    patient_id: str | None = None,
) -> ImagingAppropriateness:
    """Rank imaging options for a clinical scenario.

    Args:
        clinical_scenario: a slug from `_SCENARIO_OPTIONS` keys.
        patient_age: years; <18 raises pediatric ALARA caution.
        patient_pregnant: pregnancy -> bias against CT/X-ray.
        cumulative_radiation_msv_last_12mo: prior radiation in last 12 months.

    Returns:
        ImagingAppropriateness with ranked options + ALARA / radiation cautions.
    """
    scenario = clinical_scenario.strip().lower()
    options_data = _SCENARIO_OPTIONS.get(scenario)

    if options_data is None:
        return ImagingAppropriateness(
            patient_id=patient_id,
            clinical_scenario=clinical_scenario,
            options=[],
            top_pick_modality=None,
            cumulative_radiation_caution=False,
            pediatric_alara_caution=False,
            rationale=(
                f"Scenario {clinical_scenario!r} not in the bundled ACR "
                f"Appropriateness Criteria subset. Defer to ACR online lookup "
                f"(https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria)."
            ),
            abstain_recommended=True,
            abstain_reason="scenario_not_in_curated_database",
        )

    pediatric_alara = patient_age is not None and patient_age < 18
    cum_caution = cumulative_radiation_msv_last_12mo > 50

    options: list[ImagingOption] = []
    for i, o in enumerate(options_data, 1):
        rating = o["rating"]
        # Pregnancy + iodinated contrast or significant radiation -> downgrade
        if patient_pregnant:
            if o.get("contrast") or o["dose"] > 1.0:
                rating = max(1, rating - 4)
        if pediatric_alara and o["dose"] >= 5.0:
            rating = max(1, rating - 2)
        options.append(ImagingOption(
            modality=o["modality"],
            appropriateness_rating=rating,
            radiation_dose_msv=o["dose"],
            iv_contrast_required=o["contrast"],
            rank=i,
            rationale=(f"ACR rating {rating}/9 "
                        f"({'usually appropriate' if rating >= 7 else 'may be appropriate' if rating >= 4 else 'usually NOT appropriate'})."),
            relative_cost_tier=o["cost"],  # type: ignore[arg-type]
        ))

    # Re-rank by adjusted appropriateness rating (descending)
    options.sort(key=lambda o: o.appropriateness_rating, reverse=True)
    for i, opt in enumerate(options, 1):
        opt.rank = i
    top = options[0] if options and options[0].appropriateness_rating >= 7 else None

    rationale = _build_rationale(scenario, options, top, pediatric_alara,
                                   cum_caution, patient_pregnant,
                                   cumulative_radiation_msv_last_12mo)

    return ImagingAppropriateness(
        patient_id=patient_id,
        clinical_scenario=clinical_scenario,
        options=options,
        top_pick_modality=top.modality if top else None,
        cumulative_radiation_caution=cum_caution,
        pediatric_alara_caution=pediatric_alara,
        rationale=rationale,
    )


def _build_rationale(scenario: str, options: list[ImagingOption],
                       top: ImagingOption | None, pediatric: bool,
                       cum: bool, pregnant: bool, cum_msv: float) -> str:
    parts = [f"Scenario: {scenario}.",
             f"{len(options)} options ranked."]
    if top:
        parts.append(f"Top pick: {top.modality} (rating {top.appropriateness_rating}/9, "
                      f"{top.radiation_dose_msv} mSv, contrast={top.iv_contrast_required}).")
    else:
        parts.append("No usually-appropriate option (rating ≥7) for this scenario.")
    if pregnant:
        parts.append("Pregnancy: ratings adjusted down for contrast / radiation modalities.")
    if pediatric:
        parts.append("Pediatric ALARA: ratings adjusted down for high-radiation modalities (≥5 mSv).")
    if cum:
        parts.append(f"Cumulative radiation last 12 months: {cum_msv} mSv > 50 -- discuss "
                      f"non-radiation alternatives (US, MRI) when feasible.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_imaging_appropriateness)
