"""healthcare.compute_iss_myeloma_staging / compute_ipss_r_mds_score /
compute_ecog_performance_status / compute_karnofsky_performance
-- Phase 13.6 H5 hematology / oncology depth bundle.

References:
- Greipp PR et al. ISS staging for myeloma. JCO 2005;23:3412-20.
- Greenberg PL et al. IPSS-R for MDS. Blood 2012;120(12):2454-65.
- Oken MM et al. ECOG performance status. AJCO 1982;5:649-655.
- Karnofsky DA, Burchenal JH. The clinical evaluation of
  chemotherapeutic agents in cancer. Columbia University Press 1949.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ISSReport(BaseModel):
    iss_stage: Literal["I", "II", "III"]
    median_overall_survival_months: int
    rationale: str
    references: list[str] = Field(default_factory=list)


class IPSSRReport(BaseModel):
    cytogenetic_points: float = Field(ge=0.0, le=4.0)
    bm_blast_points: float = Field(ge=0.0, le=3.0)
    hb_points: float = Field(ge=0.0, le=1.5)
    plt_points: float = Field(ge=0.0, le=1.0)
    anc_points: float = Field(ge=0.0, le=0.5)
    total_score: float = Field(ge=0.0, le=10.0)
    ipss_r_category: Literal[
        "very_low", "low", "intermediate", "high", "very_high",
    ]
    median_overall_survival_years: float
    rationale: str
    references: list[str] = Field(default_factory=list)


class ECOGReport(BaseModel):
    grade: Literal[0, 1, 2, 3, 4, 5]
    description: str
    chemotherapy_eligibility: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class KarnofskyReport(BaseModel):
    score: int = Field(ge=0, le=100)
    description: str
    care_setting: Literal[
        "fully_active", "self_care_with_assistance",
        "requires_caretaker", "hospice_or_palliative",
    ]
    ecog_equivalent: int = Field(ge=0, le=5)
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# ISS multiple myeloma
# ─────────────────────────────────────────────────────────────────────


async def compute_iss_myeloma_staging(
    *,
    serum_beta2_microglobulin_mg_l: float,
    serum_albumin_g_dl: float,
) -> ISSReport:
    """ISS for myeloma -- Greipp 2005."""
    if serum_beta2_microglobulin_mg_l < 3.5 and serum_albumin_g_dl >= 3.5:
        stage, median = "I", 62
    elif serum_beta2_microglobulin_mg_l >= 5.5:
        stage, median = "III", 29
    else:
        stage, median = "II", 44
    rationale = (
        f"β₂-microglobulin {serum_beta2_microglobulin_mg_l:.2f} mg/L; "
        f"albumin {serum_albumin_g_dl:.1f} g/dL -> ISS {stage}; "
        f"median OS {median} months."
    )
    return ISSReport(
        iss_stage=stage,                                  # type: ignore[arg-type]
        median_overall_survival_months=median,
        rationale=rationale,
        references=["Greipp PR et al. JCO 2005;23:3412-20."],
    )


# ─────────────────────────────────────────────────────────────────────
# IPSS-R MDS
# ─────────────────────────────────────────────────────────────────────


def _ipss_cytogenetic(category: str) -> float:
    return {
        "very_good": 0, "good": 1, "intermediate": 2,
        "poor": 3, "very_poor": 4,
    }.get(category.lower(), 2)


def _ipss_blast(blast_pct: float) -> float:
    if blast_pct <= 2: return 0
    if blast_pct < 5: return 1
    if blast_pct <= 10: return 2
    return 3


def _ipss_hb(hb: float) -> float:
    if hb >= 10: return 0
    if hb >= 8: return 1
    return 1.5


def _ipss_plt(plt: float) -> float:
    if plt >= 100: return 0
    if plt >= 50: return 0.5
    return 1


def _ipss_anc(anc: float) -> float:
    if anc >= 0.8: return 0
    return 0.5


_IPSS_OS = [(1.5, ("very_low", 8.8)), (3.0, ("low", 5.3)),
                (4.5, ("intermediate", 3.0)),
                (6.0, ("high", 1.6)),
                (10.0, ("very_high", 0.8))]


async def compute_ipss_r_mds_score(
    *,
    cytogenetic_category: str,
    bm_blast_pct: float,
    hemoglobin_g_dl: float,
    platelets_thousands_per_uL: float,
    anc_thousands_per_uL: float,
) -> IPSSRReport:
    """IPSS-R for MDS -- Greenberg 2012."""
    cyto = _ipss_cytogenetic(cytogenetic_category)
    blast = _ipss_blast(bm_blast_pct)
    hb = _ipss_hb(hemoglobin_g_dl)
    plt = _ipss_plt(platelets_thousands_per_uL)
    anc = _ipss_anc(anc_thousands_per_uL)
    total = cyto + blast + hb + plt + anc
    cat, os_yr = "very_high", 0.8
    for thresh, (label, os) in _IPSS_OS:
        if total <= thresh:
            cat, os_yr = label, os
            break
    rationale = (
        f"IPSS-R total {total:.1f} (cyto {cyto} + blasts {blast} "
        f"+ Hb {hb} + Plt {plt} + ANC {anc}); category {cat}; "
        f"median OS ~ {os_yr:.1f} years."
    )
    return IPSSRReport(
        cytogenetic_points=cyto, bm_blast_points=blast,
        hb_points=hb, plt_points=plt, anc_points=anc,
        total_score=total,
        ipss_r_category=cat,                              # type: ignore[arg-type]
        median_overall_survival_years=os_yr,
        rationale=rationale,
        references=["Greenberg PL et al. Blood 2012;120(12):2454-65."],
    )


# ─────────────────────────────────────────────────────────────────────
# ECOG / Karnofsky
# ─────────────────────────────────────────────────────────────────────


_ECOG_TEXT = {
    0: "Fully active, no restriction",
    1: "Restricted in strenuous activity but ambulatory",
    2: "Ambulatory + capable of self-care, > 50% waking hours up",
    3: "Capable of only limited self-care, > 50% in bed/chair",
    4: "Completely disabled, no self-care",
    5: "Dead",
}


async def compute_ecog_performance_status(
    *,
    grade: int,
) -> ECOGReport:
    """ECOG PS -- Oken 1982."""
    grade = max(0, min(5, int(grade)))
    description = _ECOG_TEXT[grade]
    chemo_eligible = grade <= 2
    rationale = (
        f"ECOG {grade}: {description}. "
        f"Chemotherapy eligibility = {chemo_eligible}."
    )
    return ECOGReport(
        grade=grade,                                      # type: ignore[arg-type]
        description=description,
        chemotherapy_eligibility=chemo_eligible,
        rationale=rationale,
        references=["Oken MM et al. AJCO 1982;5:649-655."],
    )


_KARNOFSKY_TEXT = {
    100: "Normal, no complaints, no evidence of disease",
    90: "Able to carry on normal activity, minor symptoms",
    80: "Normal activity with effort",
    70: "Cares for self but unable to carry on normal activity",
    60: "Requires occasional assistance",
    50: "Requires considerable assistance + frequent care",
    40: "Disabled, requires special care + assistance",
    30: "Severely disabled, hospitalisation indicated",
    20: "Very ill, active supportive treatment necessary",
    10: "Moribund",
    0: "Dead",
}


def _karnofsky_to_ecog(score: int) -> int:
    if score >= 100: return 0
    if score >= 80: return 1
    if score >= 60: return 2
    if score >= 40: return 3
    if score >= 10: return 4
    return 5


async def compute_karnofsky_performance(
    *,
    score: int,
) -> KarnofskyReport:
    """Karnofsky Performance Status -- Karnofsky 1949."""
    score = max(0, min(100, (int(score) // 10) * 10))
    desc = _KARNOFSKY_TEXT.get(score, "Unspecified")
    if score >= 80: care = "fully_active"
    elif score >= 60: care = "self_care_with_assistance"
    elif score >= 30: care = "requires_caretaker"
    else: care = "hospice_or_palliative"
    ecog = _karnofsky_to_ecog(score)
    rationale = f"Karnofsky {score}: {desc}; ECOG-equivalent {ecog}."
    return KarnofskyReport(
        score=score, description=desc,
        care_setting=care,                                # type: ignore[arg-type]
        ecog_equivalent=ecog, rationale=rationale,
        references=["Karnofsky DA, Burchenal JH. 1949."],
    )


def register(mcp) -> None:
    mcp.tool()(compute_iss_myeloma_staging)
    mcp.tool()(compute_ipss_r_mds_score)
    mcp.tool()(compute_ecog_performance_status)
    mcp.tool()(compute_karnofsky_performance)
