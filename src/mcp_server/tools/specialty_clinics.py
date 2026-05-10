"""healthcare.compute_lesion_triage / compute_diabetic_retinopathy_severity /
compute_pft_interpretation
-- Phase 13.4 H1 specialty-clinic bundle (dermatology, ophthalmology,
pulmonology).

Pure-deterministic clinical triage tools for outpatient specialty
referrals. Every threshold cited inline.

References:
- Friedman RJ et al. ABCDE rule of melanoma. CA Cancer J Clin 1985.
- MacKie RM. 7-point checklist for melanoma. BMJ 1989.
- Wilkinson CP et al. ETDRS DR severity scale. Ophthalmology 2003.
- GOLD 2024 Report -- COPD strategy.
- Nathan RA et al. Asthma Control Test. JACI 2004;113:59-65.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Lightweight inline schemas
# ─────────────────────────────────────────────────────────────────────


class LesionTriageReport(BaseModel):
    abcde_score: int = Field(ge=0, le=5)
    seven_point_score: int = Field(ge=0, le=11)
    suspicion_tier: Literal[
        "benign", "monitor", "biopsy_recommended", "urgent_referral",
    ]
    biopsy_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class DRSeverityReport(BaseModel):
    severity_class: Literal[
        "no_dr", "mild_npdr", "moderate_npdr",
        "severe_npdr", "pdr",
    ]
    macular_oedema_present: bool
    referral_urgency: Literal["routine", "soon", "urgent"]
    rationale: str
    references: list[str] = Field(default_factory=list)


class PFTInterpretationReport(BaseModel):
    fev1_fvc_ratio: float
    fev1_pct_predicted: float
    pattern: Literal[
        "normal", "obstructive", "restrictive",
        "mixed", "indeterminate",
    ]
    gold_stage: Literal["none", "GOLD_1", "GOLD_2", "GOLD_3", "GOLD_4"]
    asthma_control_test_score: int | None = None
    asthma_control_tier: Literal[
        "well_controlled", "not_well_controlled",
        "very_poorly_controlled", "not_assessed",
    ]
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────


async def compute_lesion_triage(
    *,
    asymmetry: bool = False,
    border_irregularity: bool = False,
    color_variegated: bool = False,
    diameter_mm: float = 0.0,
    evolving_or_changing: bool = False,
    seven_point_atypical_features: int = 0,
    history_uv_exposure: bool = False,
    family_history_melanoma: bool = False,
) -> LesionTriageReport:
    """Skin-lesion triage combining the ABCDE rule (Friedman 1985) +
    7-point checklist (MacKie 1989). Returns a 4-tier suspicion +
    biopsy recommendation.

    Args:
        asymmetry / border_irregularity / color_variegated /
        evolving_or_changing: ABCDE binary flags.
        diameter_mm: lesion diameter; >= 6 mm is the 'D' criterion.
        seven_point_atypical_features: count of 7-point checklist
            atypical features (irregular pigment, lateral growth, etc.)
        history_uv_exposure / family_history_melanoma: risk modifiers.
    """
    abcde = sum([
        int(asymmetry), int(border_irregularity),
        int(color_variegated), int(diameter_mm >= 6.0),
        int(evolving_or_changing),
    ])
    seven = max(0, min(11, int(seven_point_atypical_features)))
    risk_modifier = int(history_uv_exposure) + int(family_history_melanoma)

    composite = abcde + (seven // 2) + risk_modifier
    if composite >= 5 or seven >= 3:
        tier, biopsy, urgency = "urgent_referral", True, "Refer to dermatology within 2 weeks"
    elif abcde >= 3 or seven >= 2:
        tier, biopsy, urgency = "biopsy_recommended", True, "Excisional biopsy recommended"
    elif abcde >= 1 or seven >= 1 or risk_modifier >= 2:
        tier, biopsy, urgency = "monitor", False, "Photo + 3-month follow-up"
    else:
        tier, biopsy, urgency = "benign", False, "Routine reassurance"

    rationale = (
        f"ABCDE score {abcde}/5; 7-point atypical {seven}; "
        f"risk modifiers {risk_modifier}; tier = {tier}; {urgency}."
    )
    return LesionTriageReport(
        abcde_score=abcde, seven_point_score=seven,
        suspicion_tier=tier,                              # type: ignore[arg-type]
        biopsy_recommended=biopsy, rationale=rationale,
        references=[
            "Friedman RJ et al. ABCDE rule. CA Cancer J Clin 1985.",
            "MacKie RM. 7-point checklist. BMJ 1989.",
        ],
    )


async def compute_diabetic_retinopathy_severity(
    *,
    microaneurysms: bool = False,
    retinal_hemorrhages_count: int = 0,    # in 4 quadrants
    venous_beading: bool = False,
    intraretinal_microvascular_abnormalities: bool = False,
    neovascularization: bool = False,
    vitreous_hemorrhage: bool = False,
    macular_thickening: bool = False,
) -> DRSeverityReport:
    """Diabetic retinopathy severity per ETDRS / Wilkinson 2003.

    Returns the canonical 5-tier classification + macular oedema flag
    + referral urgency. Inputs are clinical-exam findings; the tool
    composes them into the standard severity rubric.
    """
    if neovascularization or vitreous_hemorrhage:
        cls = "pdr"
        urgency = "urgent"
    elif (retinal_hemorrhages_count >= 20 or venous_beading
              or intraretinal_microvascular_abnormalities):
        cls = "severe_npdr"
        urgency = "urgent"
    elif retinal_hemorrhages_count >= 5:
        cls = "moderate_npdr"
        urgency = "soon"
    elif microaneurysms or retinal_hemorrhages_count >= 1:
        cls = "mild_npdr"
        urgency = "soon"
    else:
        cls = "no_dr"
        urgency = "routine"

    rationale = (
        f"DR class = {cls}; macular oedema = {macular_thickening}; "
        f"referral urgency = {urgency}."
    )
    return DRSeverityReport(
        severity_class=cls,                              # type: ignore[arg-type]
        macular_oedema_present=macular_thickening,
        referral_urgency=urgency,                        # type: ignore[arg-type]
        rationale=rationale,
        references=[
            "Wilkinson CP et al. ETDRS scale. Ophthalmology 2003.",
            "ICO Diabetic Eye Care Guidelines (2024).",
        ],
    )


async def compute_pft_interpretation(
    *,
    fev1_l: float,
    fvc_l: float,
    fev1_pct_predicted: float,
    asthma_control_test_score: int | None = None,
) -> PFTInterpretationReport:
    """Pulmonary function test interpretation: FEV1/FVC ratio + GOLD
    COPD staging + (optional) Asthma Control Test scoring.

    Args:
        fev1_l / fvc_l: post-bronchodilator absolute volumes (litres).
        fev1_pct_predicted: % predicted FEV1.
        asthma_control_test_score: optional ACT (5-25).
    """
    if fvc_l <= 0:
        raise ValueError("fvc_l must be > 0")
    ratio = fev1_l / fvc_l
    if ratio >= 0.70:
        if fev1_pct_predicted < 80:
            pattern = "restrictive"
        else:
            pattern = "normal"
    else:
        pattern = "obstructive"

    gold = "none"
    if pattern == "obstructive":
        if fev1_pct_predicted >= 80:
            gold = "GOLD_1"
        elif fev1_pct_predicted >= 50:
            gold = "GOLD_2"
        elif fev1_pct_predicted >= 30:
            gold = "GOLD_3"
        else:
            gold = "GOLD_4"

    if asthma_control_test_score is None:
        act_tier = "not_assessed"
    elif asthma_control_test_score >= 20:
        act_tier = "well_controlled"
    elif asthma_control_test_score >= 16:
        act_tier = "not_well_controlled"
    else:
        act_tier = "very_poorly_controlled"

    rationale = (
        f"FEV1/FVC = {ratio:.2f}; FEV1 % predicted "
        f"{fev1_pct_predicted:.1f}%; pattern = {pattern}; GOLD = {gold}; "
        f"ACT tier = {act_tier}."
    )
    return PFTInterpretationReport(
        fev1_fvc_ratio=round(ratio, 3),
        fev1_pct_predicted=fev1_pct_predicted,
        pattern=pattern,                                  # type: ignore[arg-type]
        gold_stage=gold,                                  # type: ignore[arg-type]
        asthma_control_test_score=asthma_control_test_score,
        asthma_control_tier=act_tier,                     # type: ignore[arg-type]
        rationale=rationale,
        references=[
            "GOLD 2024 Report -- COPD diagnosis + staging.",
            "Nathan RA et al. Asthma Control Test. JACI 2004;113:59-65.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_lesion_triage)
    mcp.tool()(compute_diabetic_retinopathy_severity)
    mcp.tool()(compute_pft_interpretation)
