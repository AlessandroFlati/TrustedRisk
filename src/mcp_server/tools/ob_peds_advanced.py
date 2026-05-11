"""healthcare.compute_bishop_induction_score / compute_apgar_score /
compute_bell_nec_stage / compute_bilirubin_nomogram
-- Phase 14.6 K5 OB / pediatric advanced bundle.

References:
- Bishop EH. Pelvic scoring for elective induction. Obstet Gynecol 1964.
- Apgar V. NEJM 1953;249:485-90.
- Bell MJ et al. Bell stage NEC. Ann Surg 1978;187:1-7.
- Bhutani VK et al. Bilirubin nomogram. Pediatrics 1999;103:6-14.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._chart_inputs import harden_clinical_inputs


class BishopReport(BaseModel):
    score: int = Field(ge=0, le=13)
    favourable_for_induction: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class ApgarReport(BaseModel):
    score_1_min: int = Field(ge=0, le=10)
    score_5_min: int = Field(ge=0, le=10)
    severity: Literal["normal", "mildly_depressed", "severely_depressed"]
    nicu_evaluation_indicated: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class BellNECReport(BaseModel):
    stage: Literal["IA", "IB", "IIA", "IIB", "IIIA", "IIIB"]
    severity_tier: Literal["suspected", "definite", "advanced"]
    surgical_consult_required: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class BilirubinNomogramReport(BaseModel):
    age_hours: int
    total_bilirubin_mg_dl: float
    risk_zone: Literal["low", "low_intermediate", "high_intermediate", "high"]
    phototherapy_indicated: bool
    exchange_transfusion_indicated: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Bishop score (cervical ripening)
# ─────────────────────────────────────────────────────────────────────


def _bishop_dilation(cm: float) -> int:
    if cm == 0: return 0
    if cm < 3: return 1
    if cm < 5: return 2
    return 3


def _bishop_effacement(pct: float) -> int:
    if pct < 30: return 0
    if pct <= 50: return 1
    if pct <= 80: return 2
    return 3


def _bishop_station(station: int) -> int:
    """Station scale −3 to +2."""
    if station == -3: return 0
    if station == -2: return 1
    if station == -1 or station == 0: return 2
    return 3


def _bishop_consistency(c: str) -> int:
    return {"firm": 0, "medium": 1, "soft": 2}.get(c.lower(), 1)


def _bishop_position(p: str) -> int:
    return {"posterior": 0, "mid": 1, "anterior": 2}.get(p.lower(), 1)


async def compute_bishop_induction_score(
    *,
    cervical_dilation_cm: float,
    cervical_effacement_pct: float,
    fetal_station: int,
    cervical_consistency: str,    # firm / medium / soft
    cervical_position: str,        # posterior / mid / anterior
) -> BishopReport:
    """Bishop 1964 score for predicting induction success.
    Score ≥ 8 -> favourable for induction (success rate ≥ vaginal
    delivery)."""
    s = (
        _bishop_dilation(cervical_dilation_cm)
        + _bishop_effacement(cervical_effacement_pct)
        + _bishop_station(fetal_station)
        + _bishop_consistency(cervical_consistency)
        + _bishop_position(cervical_position)
    )
    fav = s >= 8
    rationale = (
        f"Bishop score {s}; favourable for induction = {fav} "
        f"(threshold ≥ 8)."
    )
    return BishopReport(
        score=s, favourable_for_induction=fav, rationale=rationale,
        references=["Bishop EH. Obstet Gynecol 1964;24:266-8."],
    )


# ─────────────────────────────────────────────────────────────────────
# Apgar score
# ─────────────────────────────────────────────────────────────────────


async def compute_apgar_score(
    *,
    one_min_appearance: int,    # 0-2
    one_min_pulse: int,
    one_min_grimace: int,
    one_min_activity: int,
    one_min_respiration: int,
    five_min_appearance: int,
    five_min_pulse: int,
    five_min_grimace: int,
    five_min_activity: int,
    five_min_respiration: int,
) -> ApgarReport:
    """5-component Apgar at 1 + 5 minutes (Apgar 1953)."""
    for v in (
        one_min_appearance, one_min_pulse, one_min_grimace,
        one_min_activity, one_min_respiration,
        five_min_appearance, five_min_pulse, five_min_grimace,
        five_min_activity, five_min_respiration,
    ):
        if v < 0 or v > 2:
            raise ValueError("each Apgar component must be 0-2")
    s1 = (
        one_min_appearance + one_min_pulse + one_min_grimace
        + one_min_activity + one_min_respiration
    )
    s5 = (
        five_min_appearance + five_min_pulse + five_min_grimace
        + five_min_activity + five_min_respiration
    )
    if s5 >= 7:
        sev = "normal"
    elif s5 >= 4:
        sev = "mildly_depressed"
    else:
        sev = "severely_depressed"
    nicu = s5 < 7 or s1 < 4
    rationale = (
        f"Apgar 1-min {s1}; 5-min {s5}; severity = {sev}; "
        f"NICU = {nicu}."
    )
    return ApgarReport(
        score_1_min=s1, score_5_min=s5,
        severity=sev,                                     # type: ignore[arg-type]
        nicu_evaluation_indicated=nicu,
        rationale=rationale,
        references=["Apgar V. NEJM 1953;249:485-90."],
    )


# ─────────────────────────────────────────────────────────────────────
# Bell NEC stage
# ─────────────────────────────────────────────────────────────────────


async def compute_bell_nec_stage(
    *,
    abdominal_distension: bool = False,
    occult_blood_in_stool: bool = False,
    radiographic_pneumatosis_intestinalis: bool = False,
    persistent_metabolic_acidosis: bool = False,
    portal_venous_gas: bool = False,
    pneumoperitoneum: bool = False,
    septic_shock_or_dic: bool = False,
) -> BellNECReport:
    """Bell 1978 staging for neonatal NEC."""
    if pneumoperitoneum or septic_shock_or_dic:
        stage = "IIIB" if septic_shock_or_dic else "IIIA"
        sev = "advanced"
    elif portal_venous_gas:
        stage = "IIB"
        sev = "definite"
    elif radiographic_pneumatosis_intestinalis or persistent_metabolic_acidosis:
        stage = "IIA"
        sev = "definite"
    elif occult_blood_in_stool:
        stage = "IB"
        sev = "suspected"
    elif abdominal_distension:
        stage = "IA"
        sev = "suspected"
    else:
        stage = "IA"
        sev = "suspected"
    surgery = stage in ("IIIA", "IIIB", "IIB")
    rationale = (
        f"Bell stage {stage} ({sev}); surgical consult = {surgery}."
    )
    return BellNECReport(
        stage=stage,                                      # type: ignore[arg-type]
        severity_tier=sev,                                # type: ignore[arg-type]
        surgical_consult_required=surgery,
        rationale=rationale,
        references=["Bell MJ et al. Ann Surg 1978;187:1-7."],
    )


# ─────────────────────────────────────────────────────────────────────
# Bilirubin nomogram (Bhutani 1999)
# ─────────────────────────────────────────────────────────────────────


def _bhutani_zone(age_hours: int, tsb: float) -> str:
    """Approximate Bhutani 1999 nomogram zones (40th / 75th / 95th
    percentile lookup)."""
    if age_hours < 24:
        if tsb < 4: return "low"
        if tsb < 6: return "low_intermediate"
        if tsb < 8: return "high_intermediate"
        return "high"
    if age_hours < 48:
        if tsb < 6: return "low"
        if tsb < 9: return "low_intermediate"
        if tsb < 12: return "high_intermediate"
        return "high"
    if age_hours < 72:
        if tsb < 8: return "low"
        if tsb < 12: return "low_intermediate"
        if tsb < 15: return "high_intermediate"
        return "high"
    if tsb < 11: return "low"
    if tsb < 14: return "low_intermediate"
    if tsb < 17: return "high_intermediate"
    return "high"


async def compute_bilirubin_nomogram(
    *,
    age_hours: int,
    total_bilirubin_mg_dl: float,
    gestational_age_weeks: int = 40,
    has_neurologic_risk_factors: bool = False,
) -> BilirubinNomogramReport:
    """Bhutani 1999 hour-specific bilirubin nomogram for term/late-
    preterm neonates. Returns risk zone + AAP 2004 phototherapy /
    exchange-transfusion indications."""
    _r, _sb, _ = await harden_clinical_inputs(
        {"gestational_age_weeks": gestational_age_weeks,
         "total_bilirubin_mg_dl": total_bilirubin_mg_dl},
        chart_derivable={"gestational_age_weeks",
                          "total_bilirubin_mg_dl"},
    )
    if _sb:
        gestational_age_weeks = _r.get("gestational_age_weeks") if _r.get("gestational_age_weeks") is not None else gestational_age_weeks
        total_bilirubin_mg_dl = _r.get("total_bilirubin_mg_dl") if _r.get("total_bilirubin_mg_dl") is not None else total_bilirubin_mg_dl
    if age_hours < 0:
        raise ValueError("age_hours must be ≥ 0")
    zone = _bhutani_zone(age_hours, total_bilirubin_mg_dl)
    photo = (
        zone in ("high_intermediate", "high")
        or (zone == "low_intermediate" and has_neurologic_risk_factors)
        or (gestational_age_weeks < 37 and zone != "low")
    )
    exchange = (
        zone == "high"
        and (total_bilirubin_mg_dl >= 20 or has_neurologic_risk_factors)
    )
    rationale = (
        f"Age {age_hours} h; TSB {total_bilirubin_mg_dl:.1f} mg/dL; "
        f"Bhutani zone = {zone}; phototherapy = {photo}; "
        f"exchange transfusion = {exchange}."
    )
    return BilirubinNomogramReport(
        age_hours=age_hours,
        total_bilirubin_mg_dl=total_bilirubin_mg_dl,
        risk_zone=zone,                                   # type: ignore[arg-type]
        phototherapy_indicated=photo,
        exchange_transfusion_indicated=exchange,
        rationale=rationale,
        references=["Bhutani VK et al. Pediatrics 1999;103:6-14."],
    )


def register(mcp) -> None:
    mcp.tool()(compute_bishop_induction_score)
    mcp.tool()(compute_apgar_score)
    mcp.tool()(compute_bell_nec_stage)
    mcp.tool()(compute_bilirubin_nomogram)
