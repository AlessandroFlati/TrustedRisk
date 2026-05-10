"""healthcare.compute_acs_disposition_decision -- chest pain disposition.

Combines HEART score + STEMI status + dynamic troponin + high-risk features
into a single disposition recommendation:
  - cath_lab_activation_immediate (STEMI or ongoing ischemia + dynamic trop)
  - admit_telemetry_for_workup (high HEART or high-risk features)
  - ed_observation_serial_troponin (moderate HEART or single elevated trop)
  - discharge_with_outpatient_followup (low HEART + non-dynamic + no high-risk)

References:
  Gulati M et al. 2021 AHA/ACC/ASE Guideline for the Evaluation and
    Diagnosis of Chest Pain. Circulation 2021;144:e368.
"""

from __future__ import annotations

from typing import Literal

from shared.schemas import ACSDisposition


async def compute_acs_disposition_decision(
    heart_score_total: int,
    has_stemi: bool = False,
    has_dynamic_troponin: bool = False,
    ongoing_chest_pain: bool = False,
    hemodynamic_instability: bool = False,
    new_heart_failure: bool = False,
    patient_id: str | None = None,
) -> ACSDisposition:
    """Decide chest-pain disposition.

    Args:
        heart_score_total: 0-10 HEART score.
        has_stemi: ECG meets STEMI criteria (≥1 mm ST elevation in 2
            contiguous limb leads, ≥2 mm in 2 contiguous precordial, new LBBB).
        has_dynamic_troponin: serial troponin shows rise/fall pattern.
        ongoing_chest_pain: pain at rest at the time of evaluation.
        hemodynamic_instability: SBP <90, signs of cardiogenic shock.
        new_heart_failure: rales / pulmonary edema.

    Returns:
        ACSDisposition with disposition tier + recommended follow-up window.
    """
    total = max(0, min(10, int(heart_score_total)))
    if total <= 3:
        band = "low"
    elif total <= 6:
        band = "moderate"
    else:
        band = "high"

    high_risk_features = (ongoing_chest_pain or hemodynamic_instability
                            or new_heart_failure)

    disposition: Literal[
        "discharge_with_outpatient_followup",
        "ed_observation_serial_troponin",
        "admit_telemetry_for_workup",
        "cath_lab_activation_immediate",
    ]
    follow_up_hours: int

    if has_stemi or hemodynamic_instability:
        disposition = "cath_lab_activation_immediate"
        follow_up_hours = 0   # immediate -- door-to-balloon 90 min
    elif has_dynamic_troponin and (total >= 4 or high_risk_features):
        disposition = "admit_telemetry_for_workup"
        follow_up_hours = 12
    elif band == "high" or high_risk_features:
        disposition = "admit_telemetry_for_workup"
        follow_up_hours = 24
    elif band == "moderate":
        disposition = "ed_observation_serial_troponin"
        follow_up_hours = 6
    else:  # low
        disposition = "discharge_with_outpatient_followup"
        follow_up_hours = 72

    rationale = _build_rationale(total, band, has_stemi, has_dynamic_troponin,
                                   high_risk_features, disposition,
                                   follow_up_hours)

    return ACSDisposition(
        patient_id=patient_id,
        heart_score_total=total,
        risk_band=band,  # type: ignore[arg-type]
        has_stemi=has_stemi,
        has_dynamic_troponin=has_dynamic_troponin,
        has_high_risk_features=high_risk_features,
        disposition=disposition,
        recommended_followup_hours=follow_up_hours,
        rationale=rationale,
    )


def _build_rationale(total: int, band: str, stemi: bool, dynamic: bool,
                       high_risk: bool, disposition: str, hours: int) -> str:
    parts = [f"HEART score {total}/10 ({band} risk band)."]
    flags: list[str] = []
    if stemi:
        flags.append("STEMI on ECG")
    if dynamic:
        flags.append("dynamic troponin")
    if high_risk:
        flags.append("high-risk features (ongoing pain / hemodynamic / new HF)")
    if flags:
        parts.append("Modifiers: " + ", ".join(flags) + ".")
    parts.append(f"Disposition: {disposition} (follow-up within {hours}h).")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_acs_disposition_decision)
