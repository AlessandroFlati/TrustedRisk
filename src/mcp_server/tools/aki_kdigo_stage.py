"""healthcare.compute_aki_kdigo_stage -- KDIGO 2012 AKI classification.

Stage criteria -- meet EITHER creatinine OR urine output:

  Stage 1: Cr ↑ ≥0.3 mg/dL within 48h
           OR Cr ↑ to 1.5-1.9× baseline within 7d
           OR UOP <0.5 mL/kg/h × 6-12h
  Stage 2: Cr ↑ to 2.0-2.9× baseline
           OR UOP <0.5 mL/kg/h × ≥12h
  Stage 3: Cr ↑ to ≥3.0× baseline
           OR Cr ≥4.0 (with acute rise of ≥0.5)
           OR initiation of RRT
           OR UOP <0.3 mL/kg/h × ≥24h OR anuria ×12h

Etiology clue (informational, not part of staging):
  pre-renal:  FENa <1% (or FEUrea <35%), responding to fluid challenge
  intrinsic:  FENa >2%, urine sediment with casts/RBCs/WBCs
  post-renal: hydronephrosis on imaging, distended bladder
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import AKIStagingReport


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_aki_kdigo_stage(
    creatinine_baseline_mg_dl: float,
    creatinine_current_mg_dl: float,
    urine_output_ml_per_kg_per_hour: float | None = None,
    urine_output_window_hours: int = 6,
    fena_pct: float | None = None,
    feurea_pct: float | None = None,
    hydronephrosis_present: bool = False,
    nephrotoxic_medications_present: bool = False,
    patient_id: str | None = None,
) -> AKIStagingReport:
    """Apply KDIGO AKI staging.

    Args:
        creatinine_baseline_mg_dl: best estimate of baseline (within 7-365d).
        creatinine_current_mg_dl: most recent value.
        urine_output_ml_per_kg_per_hour: rolling average over the past
            `urine_output_window_hours`.
        urine_output_window_hours: window of the UOP measure (6 / 12 / 24).
        fena_pct / feurea_pct: optional fractional excretion to suggest etiology.
        hydronephrosis_present: imaging finding (suggests post-renal).
        nephrotoxic_medications_present: helps drive medication review.

    Returns:
        AKIStagingReport with stage + etiology hint + consult recommendation.
    """
    if (creatinine_baseline_mg_dl is None
            or creatinine_baseline_mg_dl <= 0
            or creatinine_current_mg_dl is None
            or creatinine_current_mg_dl <= 0):
        return AKIStagingReport(
            patient_id=patient_id,
            creatinine_baseline_mg_dl=creatinine_baseline_mg_dl or 0.0,
            creatinine_current_mg_dl=creatinine_current_mg_dl or 0.0,
            creatinine_change_ratio=0.0,
            creatinine_change_absolute=0.0,
            urine_output_ml_per_kg_per_hour=urine_output_ml_per_kg_per_hour,
            aki_stage="no_aki",
            aki_etiology_clue="unclear",
            nephrotoxin_review_recommended=False,
            nephrology_consult_indicated=False,
            rationale=(
                "AKI staging abstained: serum creatinine values "
                "missing or non-positive. The 'no_aki' stage shown is "
                "a placeholder, NOT a clinical finding. A missing "
                "baseline cannot be assumed normal -- it must be "
                "obtained from prior labs (within 7-365 days)."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_creatinine: KDIGO requires a baseline and a "
                "current creatinine, both > 0 mg/dL. Without them the "
                "stage cannot be calibrated."
            ),
        )

    abs_change = creatinine_current_mg_dl - creatinine_baseline_mg_dl
    ratio = creatinine_current_mg_dl / creatinine_baseline_mg_dl

    # Default no AKI
    stage: Literal["no_aki", "stage_1", "stage_2", "stage_3"] = "no_aki"

    # Stage 3 (highest priority)
    if ratio >= 3.0:
        stage = "stage_3"
    elif (creatinine_current_mg_dl >= 4.0 and abs_change >= 0.5):
        stage = "stage_3"
    elif (urine_output_ml_per_kg_per_hour is not None
            and urine_output_ml_per_kg_per_hour < 0.3
            and urine_output_window_hours >= 24):
        stage = "stage_3"
    # Stage 2
    elif ratio >= 2.0:
        stage = "stage_2"
    elif (urine_output_ml_per_kg_per_hour is not None
            and urine_output_ml_per_kg_per_hour < 0.5
            and urine_output_window_hours >= 12):
        stage = "stage_2"
    # Stage 1
    elif abs_change >= 0.3:
        stage = "stage_1"
    elif ratio >= 1.5:
        stage = "stage_1"
    elif (urine_output_ml_per_kg_per_hour is not None
            and urine_output_ml_per_kg_per_hour < 0.5
            and urine_output_window_hours >= 6):
        stage = "stage_1"

    # Etiology clue
    etiology: Literal["pre_renal", "intrinsic", "post_renal", "unclear"] = "unclear"
    if hydronephrosis_present:
        etiology = "post_renal"
    elif fena_pct is not None and fena_pct < 1.0:
        etiology = "pre_renal"
    elif feurea_pct is not None and feurea_pct < 35.0:
        etiology = "pre_renal"
    elif fena_pct is not None and fena_pct > 2.0:
        etiology = "intrinsic"

    consult = stage in ("stage_2", "stage_3")
    nephrotox_review = (stage in ("stage_1", "stage_2", "stage_3")
                          or nephrotoxic_medications_present)

    rationale = _build_rationale(creatinine_baseline_mg_dl,
                                   creatinine_current_mg_dl, abs_change, ratio,
                                   urine_output_ml_per_kg_per_hour,
                                   urine_output_window_hours, stage, etiology,
                                   consult, nephrotox_review)

    return AKIStagingReport(
        patient_id=patient_id,
        creatinine_baseline_mg_dl=creatinine_baseline_mg_dl,
        creatinine_current_mg_dl=creatinine_current_mg_dl,
        creatinine_change_ratio=round(ratio, 2),
        creatinine_change_absolute=round(abs_change, 2),
        urine_output_ml_per_kg_per_hour=urine_output_ml_per_kg_per_hour,
        aki_stage=stage,  # type: ignore[arg-type]
        aki_etiology_clue=etiology,  # type: ignore[arg-type]
        nephrotoxin_review_recommended=nephrotox_review,
        nephrology_consult_indicated=consult,
        rationale=rationale,
    )


def _build_rationale(base: float, cur: float, abs_chg: float, ratio: float,
                       uop: float | None, uop_window: int, stage: str,
                       etiology: str, consult: bool, nephrotox: bool) -> str:
    parts = [
        f"Cr {base} -> {cur} mg/dL (Δ {abs_chg:+.2f}, ratio {ratio:.2f}×).",
    ]
    if uop is not None:
        parts.append(f"UOP {uop} mL/kg/h over last {uop_window}h.")
    parts.append(f"KDIGO stage: {stage}.")
    if etiology != "unclear":
        parts.append(f"Etiology suggestion: {etiology}.")
    if consult:
        parts.append("Nephrology consult INDICATED.")
    if nephrotox:
        parts.append(
            "Review and discontinue / dose-adjust nephrotoxins (NSAIDs, "
            "ACE-I/ARB, contrast, aminoglycosides) where feasible."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_aki_kdigo_stage)
