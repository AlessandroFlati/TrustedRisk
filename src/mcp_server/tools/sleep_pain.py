"""healthcare.compute_epworth_sleepiness_scale / compute_stop_bang_osa_screen
/ compute_dn4_neuropathic_pain
-- Phase 13.8 H7 sleep medicine + pain management bundle.

References:
- Johns MW. Epworth Sleepiness Scale. Sleep 1991;14(6):540-545.
- Chung F et al. STOP-BANG questionnaire. Anesthesiology 2008;108:812-21.
- Bouhassira D et al. DN4. Pain 2005;114:29-36.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EpworthReport(BaseModel):
    score: int = Field(ge=0, le=24)
    severity: Literal[
        "normal", "mild_excessive", "moderate_excessive", "severe_excessive",
    ]
    sleep_study_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class STOPBANGReport(BaseModel):
    score: int = Field(ge=0, le=8)
    risk_tier: Literal["low", "intermediate", "high"]
    polysomnography_referral_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class DN4Report(BaseModel):
    score: int = Field(ge=0, le=10)
    neuropathic_pain_likely: bool
    suggested_first_line: list[str]
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Epworth Sleepiness Scale
# ─────────────────────────────────────────────────────────────────────


async def compute_epworth_sleepiness_scale(
    *,
    sitting_reading: int = 0,
    watching_tv: int = 0,
    sitting_inactive_in_public: int = 0,
    passenger_in_car_one_hour: int = 0,
    lying_down_to_rest_afternoon: int = 0,
    sitting_and_talking_to_someone: int = 0,
    sitting_quietly_after_lunch: int = 0,
    in_car_stopped_in_traffic: int = 0,
) -> EpworthReport:
    """Epworth Sleepiness Scale (Johns 1991). Each item 0-3."""
    items = [
        sitting_reading, watching_tv, sitting_inactive_in_public,
        passenger_in_car_one_hour, lying_down_to_rest_afternoon,
        sitting_and_talking_to_someone, sitting_quietly_after_lunch,
        in_car_stopped_in_traffic,
    ]
    if any(i < 0 or i > 3 for i in items):
        raise ValueError("each Epworth item must be 0-3")
    s = sum(items)
    if s <= 10:
        sev = "normal"
        sleep_study = False
    elif s <= 12:
        sev = "mild_excessive"
        sleep_study = False
    elif s <= 15:
        sev = "moderate_excessive"
        sleep_study = True
    else:
        sev = "severe_excessive"
        sleep_study = True
    rationale = (
        f"Epworth {s}/24; severity {sev}; "
        f"sleep study referral = {sleep_study}."
    )
    return EpworthReport(
        score=s, severity=sev,                            # type: ignore[arg-type]
        sleep_study_recommended=sleep_study,
        rationale=rationale,
        references=["Johns MW. Sleep 1991;14(6):540-545."],
    )


# ─────────────────────────────────────────────────────────────────────
# STOP-BANG OSA
# ─────────────────────────────────────────────────────────────────────


async def compute_stop_bang_osa_screen(
    *,
    snoring_loudly: bool = False,
    tired_during_day: bool = False,
    observed_apnea: bool = False,
    high_blood_pressure: bool = False,
    bmi_gt_35: bool = False,
    age_gt_50: bool = False,
    neck_circumference_gt_40_cm: bool = False,
    male_sex: bool = False,
) -> STOPBANGReport:
    """STOP-BANG questionnaire (Chung 2008)."""
    s = sum([
        snoring_loudly, tired_during_day, observed_apnea,
        high_blood_pressure, bmi_gt_35, age_gt_50,
        neck_circumference_gt_40_cm, male_sex,
    ])
    if s <= 2:
        tier, refer = "low", False
    elif s <= 4:
        tier, refer = "intermediate", True
    else:
        tier, refer = "high", True
    rationale = (
        f"STOP-BANG {s}/8; risk {tier}; PSG referral = {refer}."
    )
    return STOPBANGReport(
        score=s, risk_tier=tier,                          # type: ignore[arg-type]
        polysomnography_referral_recommended=refer,
        rationale=rationale,
        references=["Chung F et al. Anesthesiology 2008;108:812-21."],
    )


# ─────────────────────────────────────────────────────────────────────
# DN4 neuropathic pain
# ─────────────────────────────────────────────────────────────────────


async def compute_dn4_neuropathic_pain(
    *,
    burning: bool = False,
    painful_cold: bool = False,
    electric_shocks: bool = False,
    tingling: bool = False,
    pins_and_needles: bool = False,
    numbness: bool = False,
    itching: bool = False,
    hypoesthesia_to_touch: bool = False,
    hypoesthesia_to_pinprick: bool = False,
    pain_provoked_by_brushing: bool = False,
) -> DN4Report:
    """DN4 (Bouhassira 2005). Score ≥ 4/10 -> neuropathic pain likely."""
    s = sum([
        burning, painful_cold, electric_shocks, tingling,
        pins_and_needles, numbness, itching,
        hypoesthesia_to_touch, hypoesthesia_to_pinprick,
        pain_provoked_by_brushing,
    ])
    likely = s >= 4
    first_line: list[str] = []
    if likely:
        first_line = [
            "Gabapentin or pregabalin (gabapentinoid)",
            "Duloxetine (SNRI) or amitriptyline (TCA)",
            "Topical lidocaine 5% patch for localised pain",
        ]
    rationale = (
        f"DN4 {s}/10; neuropathic pain likely = {likely}. "
        f"First-line: {len(first_line)} options."
    )
    return DN4Report(
        score=s, neuropathic_pain_likely=likely,
        suggested_first_line=first_line, rationale=rationale,
        references=["Bouhassira D et al. Pain 2005;114:29-36."],
    )


def register(mcp) -> None:
    mcp.tool()(compute_epworth_sleepiness_scale)
    mcp.tool()(compute_stop_bang_osa_screen)
    mcp.tool()(compute_dn4_neuropathic_pain)
