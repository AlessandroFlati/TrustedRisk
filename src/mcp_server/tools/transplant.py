"""healthcare.compute_kdpi_kidney_donor / compute_epts_recipient_score /
compute_immunosuppression_dose_check
-- Phase 13.9 H3 transplant medicine bundle.

References:
- Rao PS et al. KDRI/KDPI. Transplantation 2009;88:231-236.
- Friedewald JJ et al. EPTS. Am J Transplant 2013;13(8):2115-23.
- Birdwell KA et al. CPIC for tacrolimus + CYP3A5. Clin Pharmacol Ther 2015.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._chart_inputs import harden_clinical_inputs


class KDPIReport(BaseModel):
    kdri_raw: float
    kdpi_pct: int = Field(ge=0, le=100)
    quality_tier: Literal["top_quartile", "middle_50", "bottom_quartile"]
    rationale: str
    references: list[str] = Field(default_factory=list)


class EPTSReport(BaseModel):
    epts_raw: float
    epts_pct: int = Field(ge=0, le=100)
    top_20_pct_eligible: bool = Field(
        description="Recipients in the EPTS top 20% are matched to "
                          "low-KDPI organs under UNOS allocation.",
    )
    rationale: str
    references: list[str] = Field(default_factory=list)


class ImmunosuppressionDoseReport(BaseModel):
    drug: str
    standard_dose_mg_per_kg_per_day: float
    adjusted_dose_mg_per_kg_per_day: float
    adjustment_reason: str
    cpic_aware: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# KDPI / KDRI
# ─────────────────────────────────────────────────────────────────────


def _kdri_score(
    age: int, height_cm: float, weight_kg: float,
    ethnicity_african_american: bool, history_of_hypertension: bool,
    history_of_diabetes: bool,
    cause_of_death_cva: bool, serum_creatinine_mg_dl: float,
    hcv_positive: bool, dcd: bool,
) -> float:
    """Simplified KDRI per Rao 2009 -- coefficients abbreviated for
    deterministic floor.

    Each non-zero factor contributes a bounded multiplier; the
    aggregate KDRI is mapped to KDPI percentile via a fixed lookup.
    """
    contrib = 0.0
    if age < 18: contrib += -0.0194 * (18 - age)
    elif age <= 50: contrib += 0.0
    elif age <= 65: contrib += 0.0128 * (age - 50)
    else: contrib += 0.0128 * 15 + 0.018 * (age - 65)

    if height_cm > 0:
        contrib += -0.0464 * ((height_cm - 170) / 10)
    if weight_kg < 80:
        contrib += -0.0199 * ((80 - weight_kg) / 5)

    if ethnicity_african_american: contrib += 0.179
    if history_of_hypertension: contrib += 0.126
    if history_of_diabetes: contrib += 0.130
    if cause_of_death_cva: contrib += 0.088
    if serum_creatinine_mg_dl > 1.5:
        contrib += 0.220 * (serum_creatinine_mg_dl - 1.5)
    if hcv_positive: contrib += 0.240
    if dcd: contrib += 0.133

    import math as _m
    return _m.exp(contrib)


def _kdri_to_kdpi(kdri: float) -> int:
    """Approximate KDPI percentile from KDRI (per OPTN 2024
    distribution statistics)."""
    if kdri < 0.6: return 5
    if kdri < 0.8: return 25
    if kdri < 1.0: return 50
    if kdri < 1.2: return 65
    if kdri < 1.5: return 80
    if kdri < 1.8: return 90
    return 99


async def compute_kdpi_kidney_donor(
    *,
    age: int,
    height_cm: float = 170,
    weight_kg: float = 75,
    ethnicity_african_american: bool = False,
    history_of_hypertension: bool = False,
    history_of_diabetes: bool = False,
    cause_of_death_cva: bool = False,
    serum_creatinine_mg_dl: float = 1.0,
    hcv_positive: bool = False,
    donation_after_circulatory_death: bool = False,
) -> KDPIReport:
    """KDRI -> KDPI for deceased-donor kidneys (Rao 2009).

    Note: this tool models the DONOR profile, not the recipient. The
    SHARP-bound chart usually represents the recipient, so chart-
    sourcing donor demographics from the recipient's bundle is wrong.
    The chart override is therefore not applied here; donor data is
    accepted from the caller as the only reasonable source.
    """
    kdri = _kdri_score(
        age=age, height_cm=height_cm, weight_kg=weight_kg,
        ethnicity_african_american=ethnicity_african_american,
        history_of_hypertension=history_of_hypertension,
        history_of_diabetes=history_of_diabetes,
        cause_of_death_cva=cause_of_death_cva,
        serum_creatinine_mg_dl=serum_creatinine_mg_dl,
        hcv_positive=hcv_positive, dcd=donation_after_circulatory_death,
    )
    kdpi = _kdri_to_kdpi(kdri)
    if kdpi <= 20: tier = "top_quartile"
    elif kdpi <= 80: tier = "middle_50"
    else: tier = "bottom_quartile"
    rationale = (
        f"KDRI {kdri:.3f} -> KDPI {kdpi}%; tier = {tier}."
    )
    return KDPIReport(
        kdri_raw=round(kdri, 3), kdpi_pct=kdpi,
        quality_tier=tier,                                # type: ignore[arg-type]
        rationale=rationale,
        references=["Rao PS et al. Transplantation 2009;88:231-236."],
    )


# ─────────────────────────────────────────────────────────────────────
# EPTS
# ─────────────────────────────────────────────────────────────────────


def _epts_raw(age: int, time_on_dialysis_years: float,
                  prior_solid_organ_transplant: bool,
                  diabetes: bool) -> float:
    """Per Friedewald 2013 simplified linear regression."""
    return (
        0.047 * max(0, age - 25)
        + 0.398 * (1 if prior_solid_organ_transplant else 0)
        - 0.237 * (max(0, age - 25)
                       * (1 if prior_solid_organ_transplant else 0))
        + 0.315 * max(0, time_on_dialysis_years)
        - 0.099 * max(0, age - 25) * max(0, time_on_dialysis_years)
        + 0.130 * (1 if diabetes else 0)
        - 0.348 * (1 if diabetes else 0) * max(0, age - 25)
    )


def _epts_pct(raw: float) -> int:
    """Empirical mapping per OPTN 2024 EPTS percentile distribution."""
    if raw < 0.5: return 10
    if raw < 1.0: return 30
    if raw < 1.5: return 55
    if raw < 2.0: return 75
    if raw < 2.5: return 90
    return 99


async def compute_epts_recipient_score(
    *,
    age: int,
    time_on_dialysis_years: float = 0.0,
    prior_solid_organ_transplant: bool = False,
    diabetes: bool = False,
) -> EPTSReport:
    """EPTS for kidney-transplant recipients (Friedewald 2013)."""
    _r, _sb, _ = await harden_clinical_inputs(
        {"age": age}, chart_derivable={"age"},
    )
    if _sb and _r.get("age") is not None:
        age = _r["age"]
    raw = _epts_raw(
        age, time_on_dialysis_years,
        prior_solid_organ_transplant, diabetes,
    )
    pct = _epts_pct(raw)
    top20 = pct <= 20
    rationale = (
        f"EPTS raw {raw:.3f} -> {pct}%; top-20% eligible = {top20}."
    )
    return EPTSReport(
        epts_raw=round(raw, 3), epts_pct=pct,
        top_20_pct_eligible=top20, rationale=rationale,
        references=["Friedewald JJ et al. Am J Transplant 2013;13(8):2115-23."],
    )


# ─────────────────────────────────────────────────────────────────────
# Immunosuppression dose check
# ─────────────────────────────────────────────────────────────────────


_IMMUNO_TABLE: dict[str, dict] = {
    "tacrolimus": {
        "standard_mg_per_kg_per_day": 0.10,
        "cypa_3a5_aware": True,
    },
    "cyclosporine": {
        "standard_mg_per_kg_per_day": 6.0,
        "cypa_3a5_aware": False,
    },
    "azathioprine": {
        "standard_mg_per_kg_per_day": 2.0,
        "tpmt_aware": True,
    },
    "mmf": {
        "standard_mg_per_kg_per_day": 14.0,
        "cypa_3a5_aware": False,
    },
    "everolimus": {
        "standard_mg_per_kg_per_day": 0.05,
        "cypa_3a5_aware": False,
    },
}


async def compute_immunosuppression_dose_check(
    *,
    drug: str,
    weight_kg: float,
    cyp3a5_phenotype: str | None = None,
    tpmt_phenotype: str | None = None,
    egfr_ml_min: float | None = None,
    on_strong_cyp3a4_inhibitor: bool = False,
) -> ImmunosuppressionDoseReport:
    """Recommend an adjusted starting dose for a transplant
    immunosuppressant given CPIC genotype + renal function +
    CYP3A4 inhibitor co-administration."""
    _r, _sb, _ = await harden_clinical_inputs(
        {"weight_kg": weight_kg, "egfr_ml_min": egfr_ml_min},
        chart_derivable={"weight_kg", "egfr_ml_min"},
    )
    if _sb:
        weight_kg = _r.get("weight_kg") if _r.get("weight_kg") is not None else weight_kg
        egfr_ml_min = _r.get("egfr_ml_min") if _r.get("egfr_ml_min") is not None else egfr_ml_min
    drug = drug.lower()
    if drug not in _IMMUNO_TABLE:
        raise ValueError(f"unknown drug: {drug}")
    spec = _IMMUNO_TABLE[drug]
    standard = float(spec["standard_mg_per_kg_per_day"])
    adjusted = standard
    reasons: list[str] = []

    if drug == "tacrolimus" and cyp3a5_phenotype:
        ph = cyp3a5_phenotype.lower()
        if "intermediate" in ph or "rapid" in ph:
            adjusted *= 1.5
            reasons.append(
                f"CYP3A5 {ph} -> 1.5x dose (CPIC tacrolimus)"
            )
    if drug == "azathioprine" and tpmt_phenotype:
        ph = tpmt_phenotype.lower()
        if "poor" in ph:
            adjusted *= 0.10
            reasons.append("TPMT poor metaboliser -> 10% standard dose")
        elif "intermediate" in ph:
            adjusted *= 0.5
            reasons.append("TPMT intermediate -> 50% standard dose")
    if egfr_ml_min is not None and egfr_ml_min < 30:
        adjusted *= 0.75
        reasons.append("eGFR < 30 -> 25% reduction")
    if on_strong_cyp3a4_inhibitor and drug in (
        "tacrolimus", "cyclosporine", "everolimus",
    ):
        adjusted *= 0.5
        reasons.append("Strong CYP3A4 inhibitor co-prescription -> halve dose")

    rationale = (
        f"{drug} standard {standard:.3f} mg/kg/day -> adjusted "
        f"{adjusted:.3f}; weight {weight_kg} kg; reasons: "
        f"{'; '.join(reasons) if reasons else 'standard dosing'}."
    )
    return ImmunosuppressionDoseReport(
        drug=drug,
        standard_dose_mg_per_kg_per_day=round(standard, 4),
        adjusted_dose_mg_per_kg_per_day=round(adjusted, 4),
        adjustment_reason="; ".join(reasons) or "standard dosing",
        cpic_aware=bool(spec.get("cypa_3a5_aware") or
                            spec.get("tpmt_aware")),
        rationale=rationale,
        references=[
            "Birdwell KA et al. CPIC tacrolimus + CYP3A5. CPT 2015.",
            "OPTN immunosuppression management protocols.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_kdpi_kidney_donor)
    mcp.tool()(compute_epts_recipient_score)
    mcp.tool()(compute_immunosuppression_dose_check)
