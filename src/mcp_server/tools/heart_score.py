"""healthcare.compute_heart_score -- HEART chest-pain rule-out score.

The HEART score (Six AJ et al. 2008, Backus BE et al. 2013 validation) is
the most widely used ED risk stratification for undifferentiated chest
pain. 5 items, 0-2 each, total 0-10.

  H -- History (suspicious for ACS): 0 (slightly), 1 (moderate), 2 (highly)
  E -- ECG: 0 (normal), 1 (non-specific repolarization), 2 (significant ST
      depression / dynamic changes)
  A -- Age: 0 (<45), 1 (45-64), 2 (≥65)
  R -- Risk factors (HTN, hyperlipidemia, DM, smoking, FHx CAD, obesity, prior
      CAD/CVA/PAD): 0 (none), 1 (1-2), 2 (≥3 OR known atherosclerotic disease)
  T -- Troponin: 0 (≤normal limit), 1 (1-3× normal), 2 (>3× normal)

Risk bands (30-day MACE = death + MI + revascularization):
  0-3   low (~1.7%)
  4-6   moderate (~16.6%)
  7-10  high (~50.1%)
"""

from __future__ import annotations

from typing import Literal

from shared.schemas import HEARTScore

from ._chart_inputs import chart_abstain_reason, harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# Risk band classification
# ─────────────────────────────────────────────────────────────────────

def _classify_risk(total: int) -> tuple[str, float]:
    if total <= 3:
        return "low", 1.7
    if total <= 6:
        return "moderate", 16.6
    return "high", 50.1


# ─────────────────────────────────────────────────────────────────────
# History scoring helper
# ─────────────────────────────────────────────────────────────────────

_HISTORY_LABELS: dict[str, int] = {
    "non_suspicious": 0,
    "slightly_suspicious": 0,
    "moderately_suspicious": 1,
    "highly_suspicious": 2,
}

_ECG_LABELS: dict[str, int] = {
    "normal": 0,
    "non_specific_repolarization": 1,
    "significant_st_depression": 2,
    "dynamic_changes": 2,
    "stemi": 2,
}


def _age_points(age: int) -> int:
    if age >= 65:
        return 2
    if age >= 45:
        return 1
    return 0


def _risk_factors_points(rf_count: int, known_cad: bool) -> int:
    if known_cad:
        return 2
    if rf_count >= 3:
        return 2
    if rf_count >= 1:
        return 1
    return 0


def _troponin_points(troponin_x_uln: float) -> int:
    if troponin_x_uln > 3.0:
        return 2
    if troponin_x_uln > 1.0:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_heart_score(
    history_descriptor: str | None = None,
    ecg_descriptor: str | None = None,
    age: int | None = None,
    risk_factors_count: int = 0,
    known_atherosclerotic_disease: bool = False,
    troponin_times_uln: float | None = None,
    patient_id: str | None = None,
) -> HEARTScore:
    """Compute the HEART score for ED chest pain.

    Args:
        history_descriptor: how suspicious the chief complaint sounds for ACS.
            One of: non_suspicious / slightly_suspicious / moderately_suspicious
            / highly_suspicious. None -> abstain (cannot guess).
        ecg_descriptor: normal / non_specific_repolarization /
            significant_st_depression / dynamic_changes / stemi. None -> abstain.
        age: patient age in years. None -> abstain.
        risk_factors_count: count of HTN, hyperlipidemia, DM, smoking,
            FHx CAD, obesity (each is one).
        known_atherosclerotic_disease: prior CAD / CVA / PAD -> automatic +2
            in the R domain (overrides count).
        troponin_times_uln: initial troponin / upper-limit-of-normal ratio.
            None -> abstain (a missing troponin cannot be assumed normal).

    Returns:
        HEARTScore with each domain's points + total + risk band + estimated
        30-day MACE risk. When any of {history, ecg, age, troponin} is
        missing the report is returned with abstain_recommended=True and
        all numeric fields set to placeholder zeros -- they MUST NOT be
        used as a clinical estimate.
    """
    _resolved, _sharp_bound, _missing_chart = await harden_clinical_inputs(
        {"age": age},
        chart_derivable={"age"},
    )
    if _sharp_bound and _missing_chart:
        return HEARTScore(
            patient_id=patient_id,
            history_points=0, ecg_points=0, age_points=0,
            risk_factors_points=0, troponin_points=0,
            total_score=0,
            risk_band="low",
            estimated_30d_mace_risk_pct=0.0,
            rationale=(
                "HEART score abstained: patient age could not be "
                "resolved from the SHARP-bound patient's chart. "
                "Caller-supplied age is discarded under SHARP to "
                "prevent fabricated demographics from driving an ED "
                "disposition decision."
            ),
            abstain_recommended=True,
            abstain_reason=chart_abstain_reason(_missing_chart),
        )
    age = _resolved.get("age")
    missing = [
        f for f, v in {
            "history_descriptor": history_descriptor,
            "ecg_descriptor": ecg_descriptor,
            "age": age,
            "troponin_times_uln": troponin_times_uln,
        }.items() if v is None
    ]
    if missing:
        return HEARTScore(
            patient_id=patient_id,
            history_points=0, ecg_points=0, age_points=0,
            risk_factors_points=0, troponin_points=0,
            total_score=0,
            risk_band="low",  # placeholder
            estimated_30d_mace_risk_pct=0.0,
            rationale=(
                "HEART score abstained: missing input(s) "
                f"{missing}. The score and risk band shown are "
                "placeholder zeros -- DO NOT use them for disposition. "
                "Default values for chest-pain risk stratification "
                "would systematically under-triage patients."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_required_inputs: HEART requires explicit "
                f"values for {missing}. A missing troponin or ECG "
                "cannot be assumed normal; a missing history cannot "
                "be assumed 'moderately suspicious'."
            ),
        )

    h_pts = _HISTORY_LABELS.get(history_descriptor.strip().lower(), 1)
    e_pts = _ECG_LABELS.get(ecg_descriptor.strip().lower(), 0)
    a_pts = _age_points(int(age))
    r_pts = _risk_factors_points(int(risk_factors_count),
                                    bool(known_atherosclerotic_disease))
    t_pts = _troponin_points(float(troponin_times_uln))
    total = h_pts + e_pts + a_pts + r_pts + t_pts
    band, mace = _classify_risk(total)

    rationale = _build_rationale(h_pts, e_pts, a_pts, r_pts, t_pts,
                                   total, band, mace,
                                   history_descriptor, ecg_descriptor,
                                   age, risk_factors_count,
                                   known_atherosclerotic_disease,
                                   troponin_times_uln)

    return HEARTScore(
        patient_id=patient_id,
        history_points=h_pts,
        ecg_points=e_pts,
        age_points=a_pts,
        risk_factors_points=r_pts,
        troponin_points=t_pts,
        total_score=total,
        risk_band=band,  # type: ignore[arg-type]
        estimated_30d_mace_risk_pct=mace,
        rationale=rationale,
    )


def _build_rationale(h: int, e: int, a: int, r: int, t: int, total: int,
                       band: str, mace: float, hist: str, ecg: str, age: int,
                       rf_count: int, known_cad: bool, trop: float) -> str:
    parts = [
        f"HEART score = {total}/10 ({band} risk; ~{mace:.1f}% 30-day MACE).",
        f"H={h} (history: {hist}); E={e} (ECG: {ecg}); A={a} (age {age}); "
        f"R={r} (risk factors count={rf_count}, known CAD/CVA/PAD={known_cad}); "
        f"T={t} (troponin × ULN = {trop:.2f}).",
    ]
    if band == "low":
        parts.append("Disposition typically: discharge with outpatient followup.")
    elif band == "moderate":
        parts.append("Disposition typically: ED observation with serial troponin.")
    else:
        parts.append("Disposition typically: admission for ACS workup ± cath lab.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_heart_score)
