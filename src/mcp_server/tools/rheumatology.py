"""healthcare.compute_das28_rheumatoid_arthritis / compute_asdas_axspa /
compute_acr_eular_ra_classification
-- Phase 14.1 K6 rheumatology bundle.

References:
- Prevoo MLL et al. DAS28. Arthritis Rheum 1995;38:44-48.
- Lukas C et al. ASDAS. Ann Rheum Dis 2009;68:18-24.
- Aletaha D et al. ACR/EULAR 2010 RA classification. Ann Rheum Dis
  2010;69:1580-1588.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field


class DAS28Report(BaseModel):
    das28_score: float = Field(ge=0.0, le=10.0)
    activity_tier: Literal[
        "remission", "low", "moderate", "high",
    ]
    biologic_eligibility_threshold_met: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class ASDASReport(BaseModel):
    asdas_score: float = Field(ge=0.0, le=10.0)
    activity_tier: Literal[
        "inactive", "moderate", "high", "very_high",
    ]
    biologic_step_up_indicated: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class ACREULARReport(BaseModel):
    joint_points: int = Field(ge=0, le=5)
    serology_points: int = Field(ge=0, le=3)
    acute_phase_points: int = Field(ge=0, le=1)
    duration_points: int = Field(ge=0, le=1)
    total_score: int = Field(ge=0, le=10)
    classification_met: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# DAS28
# ─────────────────────────────────────────────────────────────────────


async def compute_das28_rheumatoid_arthritis(
    *,
    tender_joint_count_28: int,
    swollen_joint_count_28: int,
    esr_mm_per_hour: float,
    patient_global_assessment_vas_100: float,
) -> DAS28Report:
    """DAS28-ESR (Prevoo 1995). Inputs are 28-joint counts + ESR + PGA.

    Score = 0.56·√TJC + 0.28·√SJC + 0.70·ln(ESR) + 0.014·PGA
    Tier: <2.6 remission; 2.6-3.2 low; 3.2-5.1 moderate; >5.1 high.
    Biologic step-up typically considered at score > 3.2 with
    inadequate response to ≥ 1 conventional DMARD.
    """
    if tender_joint_count_28 < 0 or tender_joint_count_28 > 28:
        raise ValueError("tender_joint_count_28 must be 0-28")
    if swollen_joint_count_28 < 0 or swollen_joint_count_28 > 28:
        raise ValueError("swollen_joint_count_28 must be 0-28")
    if esr_mm_per_hour <= 0:
        raise ValueError("esr_mm_per_hour must be > 0")
    if not 0 <= patient_global_assessment_vas_100 <= 100:
        raise ValueError("patient_global_assessment_vas_100 must be 0-100")

    score = (
        0.56 * math.sqrt(tender_joint_count_28)
        + 0.28 * math.sqrt(swollen_joint_count_28)
        + 0.70 * math.log(esr_mm_per_hour)
        + 0.014 * patient_global_assessment_vas_100
    )
    if score < 2.6:
        tier = "remission"
    elif score < 3.2:
        tier = "low"
    elif score < 5.1:
        tier = "moderate"
    else:
        tier = "high"
    biologic = score > 3.2

    rationale = (
        f"DAS28-ESR {score:.2f}; activity = {tier}; "
        f"biologic step-up threshold met = {biologic}."
    )
    return DAS28Report(
        das28_score=round(score, 2),
        activity_tier=tier,                              # type: ignore[arg-type]
        biologic_eligibility_threshold_met=biologic,
        rationale=rationale,
        references=["Prevoo MLL et al. Arthritis Rheum 1995;38:44-48."],
    )


# ─────────────────────────────────────────────────────────────────────
# ASDAS
# ─────────────────────────────────────────────────────────────────────


async def compute_asdas_axspa(
    *,
    back_pain_vas_0_10: float,
    duration_morning_stiffness_vas_0_10: float,
    patient_global_vas_0_10: float,
    peripheral_pain_swelling_vas_0_10: float,
    crp_mg_l: float,
) -> ASDASReport:
    """ASDAS-CRP for axial spondyloarthritis (Lukas 2009).

    Score = 0.12·BackPain + 0.06·MorningStiff + 0.11·PtGlobal
            + 0.07·PeripheralPain + 0.58·ln(CRP+1)
    Tier: <1.3 inactive; 1.3-2.1 moderate; 2.1-3.5 high; >3.5 very high.
    """
    for v in (back_pain_vas_0_10, duration_morning_stiffness_vas_0_10,
                  patient_global_vas_0_10,
                  peripheral_pain_swelling_vas_0_10):
        if not 0 <= v <= 10:
            raise ValueError("VAS inputs must be 0-10")
    if crp_mg_l < 0:
        raise ValueError("crp_mg_l must be ≥ 0")

    score = (
        0.12 * back_pain_vas_0_10
        + 0.06 * duration_morning_stiffness_vas_0_10
        + 0.11 * patient_global_vas_0_10
        + 0.07 * peripheral_pain_swelling_vas_0_10
        + 0.58 * math.log(crp_mg_l + 1)
    )
    if score < 1.3:
        tier = "inactive"
    elif score < 2.1:
        tier = "moderate"
    elif score < 3.5:
        tier = "high"
    else:
        tier = "very_high"
    step_up = score >= 2.1
    rationale = (
        f"ASDAS-CRP {score:.2f}; tier = {tier}; "
        f"biologic step-up = {step_up}."
    )
    return ASDASReport(
        asdas_score=round(score, 2),
        activity_tier=tier,                              # type: ignore[arg-type]
        biologic_step_up_indicated=step_up,
        rationale=rationale,
        references=["Lukas C et al. Ann Rheum Dis 2009;68:18-24."],
    )


# ─────────────────────────────────────────────────────────────────────
# ACR/EULAR 2010 RA classification
# ─────────────────────────────────────────────────────────────────────


def _joint_points(small_joints: int, large_joints: int) -> int:
    """ACR/EULAR Table 1 -- joint involvement points (0-5)."""
    if small_joints > 10 or large_joints + small_joints > 10:
        return 5
    if small_joints >= 4:
        return 3
    if small_joints >= 1:
        return 2
    if large_joints >= 2:
        return 1
    return 0


async def compute_acr_eular_ra_classification(
    *,
    n_small_joints_involved: int,
    n_large_joints_involved: int,
    rf_or_acpa_positive: bool,
    rf_or_acpa_high_titre: bool,
    elevated_crp_or_esr: bool,
    symptom_duration_weeks_ge_6: bool,
) -> ACREULARReport:
    """ACR/EULAR 2010 RA classification (Aletaha 2010).

    Score components (max 10):
      A. Joint involvement: 0-5.
      B. Serology (RF/ACPA): 0 negative; 2 low-positive; 3 high-positive.
      C. Acute-phase reactants (CRP/ESR): 1 if elevated.
      D. Symptom duration ≥ 6 weeks: 1.
    Threshold: ≥ 6 -> definite RA.
    """
    a = _joint_points(n_small_joints_involved, n_large_joints_involved)
    if rf_or_acpa_high_titre:
        b = 3
    elif rf_or_acpa_positive:
        b = 2
    else:
        b = 0
    c = 1 if elevated_crp_or_esr else 0
    d = 1 if symptom_duration_weeks_ge_6 else 0
    total = a + b + c + d
    met = total >= 6
    rationale = (
        f"Joint {a} + serology {b} + APR {c} + duration {d} = {total}/10; "
        f"RA classification met = {met}."
    )
    return ACREULARReport(
        joint_points=a, serology_points=b,
        acute_phase_points=c, duration_points=d,
        total_score=total, classification_met=met,
        rationale=rationale,
        references=[
            "Aletaha D et al. ACR/EULAR. Ann Rheum Dis 2010;69:1580-1588.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_das28_rheumatoid_arthritis)
    mcp.tool()(compute_asdas_axspa)
    mcp.tool()(compute_acr_eular_ra_classification)
