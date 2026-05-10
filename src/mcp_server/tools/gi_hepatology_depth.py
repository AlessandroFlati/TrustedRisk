"""healthcare.compute_maddrey_alcoholic_hepatitis / compute_fib4_liver_fibrosis /
compute_glasgow_blatchford_ugib / compute_rome_iv_ibs
-- Phase 14.4 K2 GI / hepatology depth bundle.

References:
- Maddrey WC. Discriminant function. Gastroenterology 1978;75:193-9.
- Sterling RK et al. FIB-4. Hepatology 2006;43:1317-25.
- Blatchford O et al. Lancet 2000;356:1318-21.
- Lacy BE et al. Rome IV criteria. Gastroenterology 2016;150:1393-1407.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field


class MaddreyReport(BaseModel):
    discriminant_function: float
    severity_tier: Literal["mild", "moderate", "severe"]
    corticosteroid_indicated: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class FIB4Report(BaseModel):
    fib4_index: float
    fibrosis_stage_estimate: Literal[
        "F0_F1_minimal", "F1_F2_intermediate",
        "F3_F4_advanced",
    ]
    biopsy_or_elastography_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class GBSReport(BaseModel):
    score: int = Field(ge=0, le=23)
    risk_tier: Literal["very_low", "moderate", "high"]
    inpatient_admission_required: bool
    endoscopy_within_24h: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class RomeIVReport(BaseModel):
    criteria_met: bool
    subtype: Literal[
        "IBS_C", "IBS_D", "IBS_M", "IBS_U", "not_IBS",
    ]
    duration_criterion_met: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Maddrey discriminant function
# ─────────────────────────────────────────────────────────────────────


async def compute_maddrey_alcoholic_hepatitis(
    *,
    patient_pt_seconds: float,
    control_pt_seconds: float,
    serum_bilirubin_mg_dl: float,
) -> MaddreyReport:
    """Maddrey 1978 DF = 4.6·(PT_pt - PT_ctl) + bilirubin.
    DF ≥ 32 -> severe; corticosteroids reduce 30-day mortality."""
    if control_pt_seconds <= 0:
        raise ValueError("control_pt_seconds must be > 0")
    df = 4.6 * (patient_pt_seconds - control_pt_seconds) \
        + serum_bilirubin_mg_dl
    if df >= 32:
        tier = "severe"
        steroid = True
    elif df >= 20:
        tier = "moderate"
        steroid = False
    else:
        tier = "mild"
        steroid = False
    rationale = (
        f"DF = {df:.1f}; tier = {tier}; corticosteroid indicated = {steroid}."
    )
    return MaddreyReport(
        discriminant_function=round(df, 1),
        severity_tier=tier,                              # type: ignore[arg-type]
        corticosteroid_indicated=steroid,
        rationale=rationale,
        references=["Maddrey WC. Gastroenterology 1978;75:193-9."],
    )


# ─────────────────────────────────────────────────────────────────────
# FIB-4
# ─────────────────────────────────────────────────────────────────────


async def compute_fib4_liver_fibrosis(
    *,
    age: int,
    ast_iu_l: float,
    alt_iu_l: float,
    platelets_thousands_per_uL: float,
) -> FIB4Report:
    """FIB-4 = (age × AST) / (PLT × √ALT). Sterling 2006 cut-offs:
    < 1.45 -> F0/F1; > 3.25 -> F3/F4."""
    if age < 18:
        raise ValueError("FIB-4 not validated < 18y")
    if alt_iu_l <= 0 or platelets_thousands_per_uL <= 0:
        raise ValueError("ALT and PLT must be > 0")
    fib4 = (age * ast_iu_l) / (
        platelets_thousands_per_uL * math.sqrt(alt_iu_l)
    )
    if fib4 < 1.45:
        stage = "F0_F1_minimal"
        biopsy = False
    elif fib4 <= 3.25:
        stage = "F1_F2_intermediate"
        biopsy = True
    else:
        stage = "F3_F4_advanced"
        biopsy = True
    rationale = (
        f"FIB-4 = {fib4:.2f}; stage = {stage}; "
        f"biopsy/elastography = {biopsy}."
    )
    return FIB4Report(
        fib4_index=round(fib4, 2),
        fibrosis_stage_estimate=stage,                   # type: ignore[arg-type]
        biopsy_or_elastography_recommended=biopsy,
        rationale=rationale,
        references=["Sterling RK et al. Hepatology 2006;43:1317-25."],
    )


# ─────────────────────────────────────────────────────────────────────
# Glasgow-Blatchford
# ─────────────────────────────────────────────────────────────────────


def _gbs_urea(urea_mmol: float) -> int:
    if urea_mmol >= 25: return 6
    if urea_mmol >= 10: return 4
    if urea_mmol >= 8: return 3
    if urea_mmol >= 6.5: return 2
    return 0


def _gbs_hgb_male(hgb: float) -> int:
    if hgb < 10: return 6
    if hgb < 12: return 3
    if hgb < 13: return 1
    return 0


def _gbs_hgb_female(hgb: float) -> int:
    if hgb < 10: return 6
    if hgb < 12: return 1
    return 0


def _gbs_sbp(sbp: float) -> int:
    if sbp < 90: return 3
    if sbp < 100: return 2
    if sbp < 110: return 1
    return 0


async def compute_glasgow_blatchford_ugib(
    *,
    blood_urea_mmol_l: float,
    hemoglobin_g_dl: float,
    sex: str,
    systolic_bp_mmHg: float,
    pulse_ge_100: bool = False,
    melena: bool = False,
    syncope: bool = False,
    hepatic_disease: bool = False,
    cardiac_failure: bool = False,
) -> GBSReport:
    """Glasgow-Blatchford for upper-GI bleed (Blatchford 2000).

    Score 0 -> very-low risk, dischargeable from ED.
    """
    s = (
        _gbs_urea(blood_urea_mmol_l)
        + (_gbs_hgb_female(hemoglobin_g_dl)
              if sex.lower().startswith("f")
              else _gbs_hgb_male(hemoglobin_g_dl))
        + _gbs_sbp(systolic_bp_mmHg)
        + (1 if pulse_ge_100 else 0)
        + (1 if melena else 0)
        + (2 if syncope else 0)
        + (2 if hepatic_disease else 0)
        + (2 if cardiac_failure else 0)
    )
    if s == 0:
        tier = "very_low"
    elif s <= 6:
        tier = "moderate"
    else:
        tier = "high"
    admit = s > 0
    endoscopy = s >= 7
    rationale = (
        f"GBS {s}; risk tier = {tier}; admit = {admit}; "
        f"endoscopy ≤ 24h = {endoscopy}."
    )
    return GBSReport(
        score=s, risk_tier=tier,                          # type: ignore[arg-type]
        inpatient_admission_required=admit,
        endoscopy_within_24h=endoscopy,
        rationale=rationale,
        references=["Blatchford O et al. Lancet 2000;356:1318-21."],
    )


# ─────────────────────────────────────────────────────────────────────
# Rome IV IBS
# ─────────────────────────────────────────────────────────────────────


async def compute_rome_iv_ibs(
    *,
    abdominal_pain_days_per_week: int,
    related_to_defecation: bool,
    associated_with_change_in_stool_frequency: bool,
    associated_with_change_in_stool_form: bool,
    symptom_duration_months: int,
    bowel_pattern_predominant: str = "mixed",   # constipation/diarrhoea/mixed/unsubtyped
) -> RomeIVReport:
    """Rome IV criteria for IBS (Lacy 2016).

    Required: ≥ 1 day/week pain on average over the last 3 months,
    associated with ≥ 2 of: defecation, change in frequency, change
    in form. Symptoms must have started > 6 months prior.
    """
    pain_ok = abdominal_pain_days_per_week >= 1
    associations = sum([
        related_to_defecation,
        associated_with_change_in_stool_frequency,
        associated_with_change_in_stool_form,
    ])
    duration_ok = symptom_duration_months >= 6
    criteria_met = pain_ok and associations >= 2 and duration_ok

    if criteria_met:
        bp = bowel_pattern_predominant.lower()
        if "constipation" in bp:
            sub = "IBS_C"
        elif "diarrhoea" in bp or "diarrhea" in bp:
            sub = "IBS_D"
        elif "mixed" in bp:
            sub = "IBS_M"
        else:
            sub = "IBS_U"
    else:
        sub = "not_IBS"
    rationale = (
        f"Pain ≥1 day/week = {pain_ok}; "
        f"{associations}/3 associations; duration ≥ 6 months = {duration_ok}; "
        f"Rome IV criteria met = {criteria_met}; subtype = {sub}."
    )
    return RomeIVReport(
        criteria_met=criteria_met, subtype=sub,           # type: ignore[arg-type]
        duration_criterion_met=duration_ok,
        rationale=rationale,
        references=["Lacy BE et al. Gastroenterology 2016;150:1393-1407."],
    )


def register(mcp) -> None:
    mcp.tool()(compute_maddrey_alcoholic_hepatitis)
    mcp.tool()(compute_fib4_liver_fibrosis)
    mcp.tool()(compute_glasgow_blatchford_ugib)
    mcp.tool()(compute_rome_iv_ibs)
