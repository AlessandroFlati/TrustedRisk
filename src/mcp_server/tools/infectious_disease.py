"""healthcare.compute_qsofa_score / compute_lactate_clearance /
compute_hiv_management_tier / compute_tb_risk_screen
-- Phase 14.3 K1 infectious-disease depth bundle.

References:
- Singer M et al. Sepsis-3. JAMA 2016;315(8):762-774.
- Nguyen HB et al. Lactate clearance. Crit Care Med 2004;32:1637-42.
- DHHS HIV management guidelines 2024.
- WHO Tuberculosis report 2024 + IGRA risk stratification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QSOFAReport(BaseModel):
    score: int = Field(ge=0, le=3)
    sepsis_likely: bool
    icu_screening_indicated: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class LactateClearanceReport(BaseModel):
    initial_lactate_mmol_l: float
    repeat_lactate_mmol_l: float
    clearance_pct: float
    target_met: bool = Field(
        description="≥ 10% clearance over 6h is the Sepsis-3 target.",
    )
    rationale: str
    references: list[str] = Field(default_factory=list)


class HIVTierReport(BaseModel):
    cd4_count: int
    viral_load_copies_ml: float
    tier: Literal[
        "viral_suppression_stable",
        "viral_suppression_immune_recovery",
        "active_viraemia_immune_compromise",
        "AIDS_defining_immune_failure",
    ]
    opportunistic_prophylaxis_indicated: list[str]
    art_change_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class TBRiskReport(BaseModel):
    risk_tier: Literal["low", "intermediate", "high"]
    recommended_test: Literal["none", "tst_or_igra", "igra", "imaging+sputum"]
    treatment_consideration: str
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# qSOFA
# ─────────────────────────────────────────────────────────────────────


async def compute_qsofa_score(
    *,
    altered_mentation_gcs_lt_15: bool = False,
    respiratory_rate_ge_22: bool = False,
    systolic_bp_le_100: bool = False,
) -> QSOFAReport:
    """Sepsis-3 qSOFA (Singer 2016): score ≥ 2 -> sepsis likely."""
    s = sum([
        altered_mentation_gcs_lt_15, respiratory_rate_ge_22,
        systolic_bp_le_100,
    ])
    likely = s >= 2
    icu = likely
    rationale = (
        f"qSOFA {s}/3; sepsis likely = {likely}; "
        f"ICU screening = {icu}."
    )
    return QSOFAReport(
        score=s, sepsis_likely=likely,
        icu_screening_indicated=icu,
        rationale=rationale,
        references=["Singer M et al. JAMA 2016;315(8):762-774."],
    )


# ─────────────────────────────────────────────────────────────────────
# Lactate clearance
# ─────────────────────────────────────────────────────────────────────


async def compute_lactate_clearance(
    *,
    initial_lactate_mmol_l: float,
    repeat_lactate_mmol_l: float,
) -> LactateClearanceReport:
    """Lactate clearance over a 6h window (Nguyen 2004)."""
    if initial_lactate_mmol_l <= 0:
        raise ValueError("initial_lactate_mmol_l must be > 0")
    clearance = (
        (initial_lactate_mmol_l - repeat_lactate_mmol_l)
        / initial_lactate_mmol_l * 100.0
    )
    target = clearance >= 10.0
    rationale = (
        f"Initial lactate {initial_lactate_mmol_l:.2f} -> "
        f"{repeat_lactate_mmol_l:.2f} mmol/L; clearance "
        f"{clearance:.1f}%; target ≥ 10% met = {target}."
    )
    return LactateClearanceReport(
        initial_lactate_mmol_l=initial_lactate_mmol_l,
        repeat_lactate_mmol_l=repeat_lactate_mmol_l,
        clearance_pct=round(clearance, 2),
        target_met=target,
        rationale=rationale,
        references=["Nguyen HB et al. Crit Care Med 2004;32:1637-42."],
    )


# ─────────────────────────────────────────────────────────────────────
# HIV management tier
# ─────────────────────────────────────────────────────────────────────


async def compute_hiv_management_tier(
    *,
    cd4_count: int,
    viral_load_copies_ml: float,
    on_art: bool = True,
) -> HIVTierReport:
    """DHHS-style HIV management tier from CD4 + VL."""
    if cd4_count < 0 or viral_load_copies_ml < 0:
        raise ValueError("counts must be ≥ 0")
    if cd4_count < 200:
        if viral_load_copies_ml > 1000 or not on_art:
            tier = "AIDS_defining_immune_failure"
        else:
            tier = "active_viraemia_immune_compromise"
    elif cd4_count < 350:
        if viral_load_copies_ml > 200:
            tier = "active_viraemia_immune_compromise"
        else:
            tier = "viral_suppression_immune_recovery"
    else:
        if viral_load_copies_ml > 50:
            tier = "active_viraemia_immune_compromise"
        else:
            tier = "viral_suppression_stable"

    prophylaxis: list[str] = []
    if cd4_count < 200:
        prophylaxis.append("TMP-SMX for PJP prophylaxis")
    if cd4_count < 50:
        prophylaxis.append("Azithromycin for MAC prophylaxis")
    if cd4_count < 100:
        prophylaxis.append("TMP-SMX for toxoplasmosis prophylaxis "
                                "(Toxoplasma seropositive)")

    art_change = (
        viral_load_copies_ml > 200 and on_art
    ) or (cd4_count < 200 and not on_art)

    rationale = (
        f"CD4 {cd4_count} cells/mm³; VL "
        f"{viral_load_copies_ml:.0f} copies/mL; "
        f"on ART = {on_art}; tier = {tier}."
    )
    return HIVTierReport(
        cd4_count=cd4_count,
        viral_load_copies_ml=viral_load_copies_ml,
        tier=tier,                                        # type: ignore[arg-type]
        opportunistic_prophylaxis_indicated=prophylaxis,
        art_change_recommended=art_change,
        rationale=rationale,
        references=["DHHS HIV Management Guidelines 2024."],
    )


# ─────────────────────────────────────────────────────────────────────
# TB risk screen
# ─────────────────────────────────────────────────────────────────────


async def compute_tb_risk_screen(
    *,
    high_burden_country_residence_or_travel: bool = False,
    close_contact_with_active_tb: bool = False,
    immunocompromised_HIV_or_TNF_inhibitor: bool = False,
    chronic_renal_failure_dialysis: bool = False,
    homeless_or_incarceration_history: bool = False,
    healthcare_worker: bool = False,
    cough_gt_3_weeks_with_constitutional_symptoms: bool = False,
) -> TBRiskReport:
    """Combine WHO + CDC LTBI risk stratification + active-TB suspicion."""
    risk_factors = sum([
        high_burden_country_residence_or_travel,
        close_contact_with_active_tb,
        immunocompromised_HIV_or_TNF_inhibitor,
        chronic_renal_failure_dialysis,
        homeless_or_incarceration_history,
        healthcare_worker,
    ])
    if cough_gt_3_weeks_with_constitutional_symptoms:
        tier = "high"
        test = "imaging+sputum"
        treatment = "Empiric airborne isolation + workup; do NOT delay."
    elif risk_factors >= 2 or close_contact_with_active_tb \
            or immunocompromised_HIV_or_TNF_inhibitor:
        tier = "intermediate"
        test = "igra"
        treatment = "If positive -> 9-month INH or 4-month rifampin."
    elif risk_factors == 1:
        tier = "low"
        test = "tst_or_igra"
        treatment = "Per CDC: discuss LTBI treatment."
    else:
        tier = "low"
        test = "none"
        treatment = "No screening indicated."
    rationale = (
        f"{risk_factors} LTBI risk factor(s); "
        f"active-TB suspicion = "
        f"{cough_gt_3_weeks_with_constitutional_symptoms}; "
        f"tier = {tier}; recommended test = {test}."
    )
    return TBRiskReport(
        risk_tier=tier,                                   # type: ignore[arg-type]
        recommended_test=test,                            # type: ignore[arg-type]
        treatment_consideration=treatment,
        rationale=rationale,
        references=[
            "WHO TB Report 2024.",
            "CDC LTBI Treatment Regimens 2020.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_qsofa_score)
    mcp.tool()(compute_lactate_clearance)
    mcp.tool()(compute_hiv_management_tier)
    mcp.tool()(compute_tb_risk_screen)
