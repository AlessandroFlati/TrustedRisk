"""healthcare.compute_rcri_cardiac_risk / compute_ariscat_pulmonary_risk /
compute_caprini_vte_risk
-- Phase 14.2 K4 surgery / peri-operative risk bundle.

References:
- Lee TH et al. Revised Cardiac Risk Index. Circulation 1999;100:1043-9.
- Canet J et al. ARISCAT score. Anesthesiology 2010;113(6):1338-1350.
- Caprini JA. Risk-assessment for VTE. Semin Thromb Hemost 1991;17 Suppl 3:304-12.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RCRIReport(BaseModel):
    score: int = Field(ge=0, le=6)
    cardiac_risk_pct: float = Field(ge=0.0, le=20.0)
    risk_tier: Literal["low", "moderate", "high"]
    cardiac_consult_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class ARISCATReport(BaseModel):
    score: int = Field(ge=0, le=123)
    pulmonary_risk_pct: float = Field(ge=0.0, le=80.0)
    risk_tier: Literal["low", "intermediate", "high"]
    rationale: str
    references: list[str] = Field(default_factory=list)


class CapriniReport(BaseModel):
    score: int = Field(ge=0)
    vte_risk_tier: Literal[
        "very_low", "low", "moderate", "high", "highest",
    ]
    prophylaxis_recommendation: str
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# RCRI
# ─────────────────────────────────────────────────────────────────────


_RCRI_RISK = {0: 0.4, 1: 0.9, 2: 6.6, 3: 11.0, 4: 11.0, 5: 11.0, 6: 11.0}


async def compute_rcri_cardiac_risk(
    *,
    high_risk_surgery: bool = False,
    history_ischemic_heart_disease: bool = False,
    history_congestive_heart_failure: bool = False,
    history_cerebrovascular_disease: bool = False,
    insulin_dependent_diabetes: bool = False,
    creatinine_gt_2_mg_dl: bool = False,
) -> RCRIReport:
    """Lee 1999 RCRI: 6 binary criteria; major-adverse-cardiac-event
    risk increases with score."""
    s = sum([
        high_risk_surgery, history_ischemic_heart_disease,
        history_congestive_heart_failure,
        history_cerebrovascular_disease,
        insulin_dependent_diabetes, creatinine_gt_2_mg_dl,
    ])
    risk = _RCRI_RISK[s]
    if s >= 3:
        tier = "high"
    elif s == 2:
        tier = "moderate"
    else:
        tier = "low"
    consult = s >= 2
    rationale = (
        f"RCRI {s}/6; MACE risk {risk:.1f}%; tier = {tier}; "
        f"cardiac consult = {consult}."
    )
    return RCRIReport(
        score=s, cardiac_risk_pct=risk,
        risk_tier=tier,                                   # type: ignore[arg-type]
        cardiac_consult_recommended=consult,
        rationale=rationale,
        references=["Lee TH et al. Circulation 1999;100:1043-9."],
    )


# ─────────────────────────────────────────────────────────────────────
# ARISCAT
# ─────────────────────────────────────────────────────────────────────


def _ariscat_age(age: int) -> int:
    if age <= 50: return 0
    if age <= 80: return 3
    return 16


def _ariscat_spo2(spo2: float) -> int:
    if spo2 >= 96: return 0
    if spo2 >= 91: return 8
    return 24


def _ariscat_incision(incision: str) -> int:
    incision = (incision or "").lower()
    if "intrathoracic" in incision: return 24
    if "upper abdominal" in incision: return 15
    if "peripheral" in incision: return 0
    return 0


def _ariscat_duration(hours: float) -> int:
    if hours < 2: return 0
    if hours <= 3: return 16
    return 23


async def compute_ariscat_pulmonary_risk(
    *,
    age: int,
    preop_spo2_pct: float,
    surgical_incision_site: str,
    surgical_duration_hours: float,
    respiratory_infection_last_month: bool = False,
    preop_anemia_hgb_lt_10: bool = False,
    emergency_surgery: bool = False,
) -> ARISCATReport:
    """Canet 2010 ARISCAT score for postoperative pulmonary
    complications."""
    s = (
        _ariscat_age(age)
        + _ariscat_spo2(preop_spo2_pct)
        + _ariscat_incision(surgical_incision_site)
        + _ariscat_duration(surgical_duration_hours)
        + (17 if respiratory_infection_last_month else 0)
        + (11 if preop_anemia_hgb_lt_10 else 0)
        + (8 if emergency_surgery else 0)
    )
    if s < 26:
        tier, risk = "low", 1.6
    elif s < 45:
        tier, risk = "intermediate", 13.3
    else:
        tier, risk = "high", 42.1
    rationale = (
        f"ARISCAT {s}; pulmonary complication risk {risk:.1f}%; "
        f"tier = {tier}."
    )
    return ARISCATReport(
        score=s, pulmonary_risk_pct=risk,
        risk_tier=tier,                                   # type: ignore[arg-type]
        rationale=rationale,
        references=["Canet J et al. Anesthesiology 2010;113:1338-1350."],
    )


# ─────────────────────────────────────────────────────────────────────
# Caprini VTE risk
# ─────────────────────────────────────────────────────────────────────


async def compute_caprini_vte_risk(
    *,
    age: int,
    bmi_gt_25: bool = False,
    minor_surgery_planned: bool = False,
    major_surgery_planned: bool = False,
    laparoscopic_surgery_lt_45_min: bool = False,
    history_vte: bool = False,
    family_history_vte: bool = False,
    factor_v_leiden_or_thrombophilia: bool = False,
    active_malignancy: bool = False,
    confined_to_bed_gt_72h: bool = False,
    central_venous_access: bool = False,
    sepsis_lt_1_month: bool = False,
    serious_lung_disease: bool = False,
    pregnant_or_postpartum: bool = False,
    oral_contraceptive_or_hrt: bool = False,
) -> CapriniReport:
    """Caprini VTE risk-assessment model (the ACCP-9 version)."""
    s = 0
    # Age
    if age >= 75: s += 3
    elif age >= 61: s += 2
    elif age >= 41: s += 1
    # 1-point items
    s += sum([
        bmi_gt_25, minor_surgery_planned,
        laparoscopic_surgery_lt_45_min,
        oral_contraceptive_or_hrt, pregnant_or_postpartum,
        serious_lung_disease,
    ])
    # 2-point items
    if major_surgery_planned: s += 2
    if confined_to_bed_gt_72h: s += 2
    if central_venous_access: s += 2
    if active_malignancy: s += 2
    # 3-point items
    if history_vte: s += 3
    if family_history_vte: s += 3
    if factor_v_leiden_or_thrombophilia: s += 3
    if sepsis_lt_1_month: s += 1

    if s == 0:
        tier = "very_low"
        rec = "Early ambulation only; no pharmacologic prophylaxis."
    elif s <= 1:
        tier = "low"
        rec = "Mechanical prophylaxis (IPC) recommended."
    elif s <= 2:
        tier = "moderate"
        rec = "Pharmacologic prophylaxis (LMWH) preferred."
    elif s <= 4:
        tier = "high"
        rec = "Pharmacologic + mechanical prophylaxis."
    else:
        tier = "highest"
        rec = "Combined prophylaxis with extended duration ≥ 4 weeks."
    rationale = (
        f"Caprini score {s}; VTE risk tier = {tier}; "
        f"recommendation: {rec}"
    )
    return CapriniReport(
        score=s,
        vte_risk_tier=tier,                              # type: ignore[arg-type]
        prophylaxis_recommendation=rec,
        rationale=rationale,
        references=[
            "Caprini JA. Semin Thromb Hemost 1991;17 Suppl 3:304-12.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_rcri_cardiac_risk)
    mcp.tool()(compute_ariscat_pulmonary_risk)
    mcp.tool()(compute_caprini_vte_risk)
