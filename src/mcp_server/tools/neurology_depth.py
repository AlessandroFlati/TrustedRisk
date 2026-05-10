"""healthcare.compute_hunt_hess_sah / compute_ich_score /
compute_modified_rankin / compute_hauser_ambulation_index
-- Phase 14.5 K3 neurology depth bundle.

References:
- Hunt WE, Hess RM. SAH grade. J Neurosurg 1968;28(1):14-20.
- Hemphill JC et al. ICH score. Stroke 2001;32(4):891-7.
- Rankin J. Mod. Rankin scale. Scott Med J 1957;2(5):200-15.
- Hauser SL et al. Ambulation index. Trial in MS 1983.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HuntHessReport(BaseModel):
    grade: Literal[1, 2, 3, 4, 5]
    description: str
    surgical_eligibility: Literal[
        "good_candidate", "guarded", "poor_candidate",
    ]
    estimated_mortality_pct: float
    rationale: str
    references: list[str] = Field(default_factory=list)


class ICHScoreReport(BaseModel):
    score: int = Field(ge=0, le=6)
    estimated_30d_mortality_pct: float = Field(ge=0.0, le=100.0)
    icu_admission_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class ModifiedRankinReport(BaseModel):
    grade: Literal[0, 1, 2, 3, 4, 5, 6]
    description: str
    independent_living: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class HauserAmbulationReport(BaseModel):
    grade: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    description: str
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Hunt-Hess SAH
# ─────────────────────────────────────────────────────────────────────


_HUNT_HESS = {
    1: ("Asymptomatic or mild headache",     "good_candidate", 11.0),
    2: ("Moderate-severe headache, no deficit", "good_candidate", 26.0),
    3: ("Drowsy with mild focal deficit",    "guarded",         37.0),
    4: ("Stupor with hemiparesis",           "poor_candidate",  71.0),
    5: ("Deep coma, decerebrate posturing",  "poor_candidate",  100.0),
}


async def compute_hunt_hess_sah(
    *,
    grade: int,
) -> HuntHessReport:
    """Hunt-Hess grading for subarachnoid haemorrhage (1968)."""
    grade = max(1, min(5, int(grade)))
    desc, elig, mort = _HUNT_HESS[grade]
    rationale = (
        f"Hunt-Hess grade {grade}: {desc}; surgical eligibility = {elig}; "
        f"estimated mortality {mort:.0f}%."
    )
    return HuntHessReport(
        grade=grade,                                      # type: ignore[arg-type]
        description=desc,
        surgical_eligibility=elig,                        # type: ignore[arg-type]
        estimated_mortality_pct=mort,
        rationale=rationale,
        references=["Hunt WE, Hess RM. J Neurosurg 1968;28:14-20."],
    )


# ─────────────────────────────────────────────────────────────────────
# ICH score
# ─────────────────────────────────────────────────────────────────────


_ICH_30D_MORT = {
    0: 0.0, 1: 13.0, 2: 26.0, 3: 72.0, 4: 97.0, 5: 100.0, 6: 100.0,
}


async def compute_ich_score(
    *,
    glasgow_coma_scale: int,
    ich_volume_ml: float,
    intraventricular_hemorrhage: bool = False,
    infratentorial_origin: bool = False,
    age_ge_80: bool = False,
) -> ICHScoreReport:
    """Hemphill 2001 ICH score (0-6)."""
    if glasgow_coma_scale < 3 or glasgow_coma_scale > 15:
        raise ValueError("GCS must be 3-15")
    if 13 <= glasgow_coma_scale <= 15:
        gcs_pts = 0
    elif 5 <= glasgow_coma_scale <= 12:
        gcs_pts = 1
    else:
        gcs_pts = 2
    s = (
        gcs_pts
        + (1 if ich_volume_ml >= 30 else 0)
        + (1 if intraventricular_hemorrhage else 0)
        + (1 if infratentorial_origin else 0)
        + (1 if age_ge_80 else 0)
    )
    s = min(6, s)
    mortality = _ICH_30D_MORT[s]
    icu = s >= 1
    rationale = (
        f"ICH score {s}; GCS pts {gcs_pts}; "
        f"30-d mortality {mortality:.0f}%; ICU = {icu}."
    )
    return ICHScoreReport(
        score=s, estimated_30d_mortality_pct=mortality,
        icu_admission_recommended=icu,
        rationale=rationale,
        references=["Hemphill JC et al. Stroke 2001;32:891-7."],
    )


# ─────────────────────────────────────────────────────────────────────
# Modified Rankin Scale
# ─────────────────────────────────────────────────────────────────────


_MRS_TEXT = {
    0: ("No symptoms", True),
    1: ("No significant disability despite symptoms", True),
    2: ("Slight disability -- independent in ADLs", True),
    3: ("Moderate disability -- needs some help, walks unaided", False),
    4: ("Moderately severe disability -- unable to walk unassisted", False),
    5: ("Severe disability -- bedridden, needs constant care", False),
    6: ("Death", False),
}


async def compute_modified_rankin(*, grade: int) -> ModifiedRankinReport:
    """Modified Rankin Scale (1957)."""
    grade = max(0, min(6, int(grade)))
    desc, indep = _MRS_TEXT[grade]
    rationale = f"mRS grade {grade}: {desc}; independent living = {indep}."
    return ModifiedRankinReport(
        grade=grade,                                      # type: ignore[arg-type]
        description=desc, independent_living=indep,
        rationale=rationale,
        references=["Rankin J. Scott Med J 1957;2(5):200-15."],
    )


# ─────────────────────────────────────────────────────────────────────
# Hauser ambulation index
# ─────────────────────────────────────────────────────────────────────


_HAUSER_TEXT = {
    0: "Asymptomatic, fully active",
    1: "Walks normally, mild fatigue at sport / heavy work",
    2: "Walks abnormally but does not require aid",
    3: "Requires unilateral support to walk 25 ft (≤ 20 s)",
    4: "Requires bilateral support to walk 25 ft (≤ 20 s)",
    5: "Walks 25 ft with bilateral support (> 20 s)",
    6: "Walks > 25 ft with bilateral support; sometimes wheelchair",
    7: "Wheelchair-bound, walks several steps with help",
    8: "Wheelchair-bound, unable to take steps",
    9: "Bedridden",
}


async def compute_hauser_ambulation_index(
    *,
    grade: int,
) -> HauserAmbulationReport:
    """Hauser Ambulation Index (1983) -- used in MS / neuromuscular trials."""
    grade = max(0, min(9, int(grade)))
    desc = _HAUSER_TEXT[grade]
    rationale = f"Hauser grade {grade}: {desc}."
    return HauserAmbulationReport(
        grade=grade,                                      # type: ignore[arg-type]
        description=desc, rationale=rationale,
        references=["Hauser SL et al. NEJM 1983;308:173-180."],
    )


def register(mcp) -> None:
    mcp.tool()(compute_hunt_hess_sah)
    mcp.tool()(compute_ich_score)
    mcp.tool()(compute_modified_rankin)
    mcp.tool()(compute_hauser_ambulation_index)
