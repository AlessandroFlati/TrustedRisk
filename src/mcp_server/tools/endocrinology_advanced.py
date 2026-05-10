"""healthcare.compute_thyroid_management / compute_adrenal_insufficiency_workup
/ compute_hypocalcemia_severity
-- Phase 13.7 H6 endocrinology depth bundle.

References:
- Jonklaas J et al. ATA Hypothyroidism Guidelines. Thyroid 2014;24(12):1670-1751.
- Bornstein SR et al. Endocrine Society 2016 -- Adrenal Insufficiency Diagnosis.
- Cooper MS, Gittoes NJ. BMJ 2008;336:1298-302 (hypocalcaemia).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ThyroidReport(BaseModel):
    pattern: Literal[
        "euthyroid", "subclinical_hypothyroid", "overt_hypothyroid",
        "subclinical_hyperthyroid", "overt_hyperthyroid",
        "central_hypothyroid", "indeterminate",
    ]
    levothyroxine_dose_change_mcg: int
    follow_up_weeks: int
    rationale: str
    references: list[str] = Field(default_factory=list)


class AdrenalReport(BaseModel):
    morning_cortisol_ug_dl: float
    cortisol_after_acth_stim_ug_dl: float | None
    diagnosis_tier: Literal[
        "rule_out", "indeterminate_repeat",
        "primary_AI", "secondary_AI",
    ]
    glucocorticoid_replacement_recommended: bool
    fludrocortisone_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


class HypocalcemiaReport(BaseModel):
    corrected_calcium_mg_dl: float
    severity: Literal["mild", "moderate", "severe", "critical"]
    iv_calcium_indicated: bool
    cardiac_monitoring_indicated: bool
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Thyroid management
# ─────────────────────────────────────────────────────────────────────


async def compute_thyroid_management(
    *,
    tsh_mU_L: float,
    free_t4_ng_dl: float | None = None,
    current_levothyroxine_dose_mcg: int = 0,
    on_levothyroxine: bool = False,
) -> ThyroidReport:
    """Categorise thyroid function + suggest a levothyroxine dose
    change consistent with ATA 2014 ranges (TSH target 0.5-4.5 mU/L)."""
    if tsh_mU_L < 0.1 and free_t4_ng_dl is not None and free_t4_ng_dl > 1.7:
        pattern = "overt_hyperthyroid"
        delta, follow = -25 if on_levothyroxine else 0, 6
    elif tsh_mU_L < 0.5:
        pattern = "subclinical_hyperthyroid"
        delta, follow = -25 if on_levothyroxine else 0, 6
    elif tsh_mU_L > 10 and (free_t4_ng_dl is None or free_t4_ng_dl < 0.8):
        pattern = "overt_hypothyroid"
        delta, follow = 25, 6
    elif tsh_mU_L > 4.5:
        pattern = "subclinical_hypothyroid"
        delta, follow = 25 if on_levothyroxine else 0, 8
    elif (tsh_mU_L < 1.0 and free_t4_ng_dl is not None
              and free_t4_ng_dl < 0.8):
        pattern = "central_hypothyroid"
        delta, follow = 25, 6
    else:
        pattern = "euthyroid"
        delta, follow = 0, 26
    rationale = (
        f"TSH {tsh_mU_L:.2f} mU/L; free T4 "
        f"{free_t4_ng_dl if free_t4_ng_dl is not None else 'n/a'}; "
        f"pattern {pattern}; "
        f"levothyroxine Δ {delta:+d} mcg; follow-up {follow} weeks."
    )
    return ThyroidReport(
        pattern=pattern,                                  # type: ignore[arg-type]
        levothyroxine_dose_change_mcg=delta,
        follow_up_weeks=follow,
        rationale=rationale,
        references=[
            "Jonklaas J et al. Thyroid 2014;24(12):1670-1751.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Adrenal insufficiency
# ─────────────────────────────────────────────────────────────────────


async def compute_adrenal_insufficiency_workup(
    *,
    morning_cortisol_ug_dl: float,
    cortisol_after_acth_stim_ug_dl: float | None = None,
    acth_pg_ml: float | None = None,
) -> AdrenalReport:
    """AI workup per Endocrine Society 2016 guidelines.

    Cut-offs:
      - Morning cortisol > 18 µg/dL -> rule out AI.
      - Morning cortisol < 3 µg/dL -> AI very likely.
      - 3-18 µg/dL: ACTH-stim test required; post-stim < 18 -> AI.
      - High ACTH + low cortisol -> primary AI.
      - Low ACTH + low cortisol -> secondary AI.
    """
    if morning_cortisol_ug_dl >= 18:
        tier = "rule_out"
        gluco, fludro = False, False
    elif (cortisol_after_acth_stim_ug_dl is not None
              and cortisol_after_acth_stim_ug_dl >= 18):
        tier = "rule_out"
        gluco, fludro = False, False
    elif morning_cortisol_ug_dl < 3:
        if acth_pg_ml is not None and acth_pg_ml > 100:
            tier = "primary_AI"
            gluco, fludro = True, True
        elif acth_pg_ml is not None:
            tier = "secondary_AI"
            gluco, fludro = True, False
        else:
            tier = "indeterminate_repeat"
            gluco, fludro = True, False
    else:
        tier = "indeterminate_repeat"
        gluco, fludro = False, False

    rationale = (
        f"Morning cortisol {morning_cortisol_ug_dl:.1f} µg/dL; "
        f"post-ACTH "
        f"{cortisol_after_acth_stim_ug_dl if cortisol_after_acth_stim_ug_dl is not None else 'n/a'}; "
        f"ACTH {acth_pg_ml if acth_pg_ml is not None else 'n/a'}; "
        f"diagnosis = {tier}; gluco replacement = {gluco}; "
        f"fludro = {fludro}."
    )
    return AdrenalReport(
        morning_cortisol_ug_dl=morning_cortisol_ug_dl,
        cortisol_after_acth_stim_ug_dl=cortisol_after_acth_stim_ug_dl,
        diagnosis_tier=tier,                              # type: ignore[arg-type]
        glucocorticoid_replacement_recommended=gluco,
        fludrocortisone_recommended=fludro,
        rationale=rationale,
        references=[
            "Bornstein SR et al. Endocrine Society 2016.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Hypocalcemia severity
# ─────────────────────────────────────────────────────────────────────


async def compute_hypocalcemia_severity(
    *,
    serum_calcium_mg_dl: float,
    serum_albumin_g_dl: float = 4.0,
    qtc_ms: float | None = None,
    symptomatic_tetany: bool = False,
) -> HypocalcemiaReport:
    """Albumin-corrected calcium severity tier + IV-calcium indication."""
    corrected = serum_calcium_mg_dl + 0.8 * (4.0 - serum_albumin_g_dl)
    if corrected < 6.0 or symptomatic_tetany:
        sev = "critical"
        iv = True
    elif corrected < 7.5:
        sev = "severe"
        iv = True
    elif corrected < 8.0:
        sev = "moderate"
        iv = False
    else:
        sev = "mild"
        iv = False

    cardiac = sev in ("severe", "critical") or (
        qtc_ms is not None and qtc_ms > 470
    )
    rationale = (
        f"Calcium {serum_calcium_mg_dl:.1f}; albumin "
        f"{serum_albumin_g_dl:.1f}; corrected {corrected:.2f}; "
        f"severity {sev}; IV calcium = {iv}; "
        f"cardiac monitoring = {cardiac}."
    )
    return HypocalcemiaReport(
        corrected_calcium_mg_dl=round(corrected, 2),
        severity=sev,                                     # type: ignore[arg-type]
        iv_calcium_indicated=iv,
        cardiac_monitoring_indicated=cardiac,
        rationale=rationale,
        references=[
            "Cooper MS, Gittoes NJ. BMJ 2008;336:1298-302.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_thyroid_management)
    mcp.tool()(compute_adrenal_insufficiency_workup)
    mcp.tool()(compute_hypocalcemia_severity)
