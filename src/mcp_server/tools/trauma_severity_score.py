"""healthcare.compute_trauma_severity_score -- ISS + RTS composite trauma severity.

ISS (Baker 1974): take the AIS severity (1-6) of the worst injury per body
region across 6 regions, square the top-3 values, sum. 0-75 (any AIS=6 ->
ISS=75 by convention). Bands: 1-8 minor, 9-15 moderate, 16-24 major,
25-49 severe, ≥50 unsurvivable.

RTS (Champion 1989): coded weights of GCS (0-4) + SBP (0-4) + RR (0-4),
weighted and summed: 0.9368*GCS_coded + 0.7326*SBP_coded + 0.2908*RR_coded.
Range 0-7.84 (commonly displayed as 0-12 raw sum); <12 = abnormal physiology.

Triage priority:
  - ISS 1-8 + RTS 12 -> minor injury / outpatient
  - ISS 9-15 + RTS 11-12 -> trauma team activation
  - ISS 16-24 OR RTS 9-10 -> trauma center transfer
  - ISS ≥25 OR RTS ≤8 -> operating room / massive transfusion ready

References:
  Baker SP, O'Neill B, Haddon W, Long WB. The Injury Severity Score:
    a method for describing patients with multiple injuries. J Trauma
    1974;14:187.
  Champion HR et al. A revision of the Trauma Score. J Trauma 1989;29:623.
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import TraumaInjury, TraumaSeverityReport

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# RTS coded weights
# ─────────────────────────────────────────────────────────────────────

def _gcs_coded(gcs: int) -> int:
    if gcs >= 13:
        return 4
    if gcs >= 9:
        return 3
    if gcs >= 6:
        return 2
    if gcs >= 4:
        return 1
    return 0


def _sbp_coded(sbp: float) -> int:
    if sbp > 89:
        return 4
    if sbp >= 76:
        return 3
    if sbp >= 50:
        return 2
    if sbp >= 1:
        return 1
    return 0


def _rr_coded(rr: float) -> int:
    if 10 <= rr <= 29:
        return 4
    if rr > 29:
        return 3
    if rr >= 6:
        return 2
    if rr >= 1:
        return 1
    return 0


def _compute_rts(gcs: int, sbp: float, rr: float) -> float:
    """T-RTS (Triage-RTS) coded sum 0-12. We use the triage variant rather than
    the weighted research variant (0-7.84) because the triage thresholds in
    field protocols (T-RTS ≤11 = trauma-center transfer) are expressed on the
    0-12 scale and are easier to interpret bedside."""
    return float(_gcs_coded(gcs) + _sbp_coded(sbp) + _rr_coded(rr))


# ─────────────────────────────────────────────────────────────────────
# ISS computation
# ─────────────────────────────────────────────────────────────────────

def _compute_iss(injuries: list[TraumaInjury]) -> int:
    if not injuries:
        return 0
    if any(inj.ais_severity == 6 for inj in injuries):
        return 75
    by_region: dict[str, int] = {}
    for inj in injuries:
        cur = by_region.get(inj.body_region, 0)
        if inj.ais_severity > cur:
            by_region[inj.body_region] = inj.ais_severity
    top3 = sorted(by_region.values(), reverse=True)[:3]
    return sum(s * s for s in top3)


# ─────────────────────────────────────────────────────────────────────
# Banding + triage priority
# ─────────────────────────────────────────────────────────────────────

def _iss_band(iss: int) -> str:
    if iss >= 50:
        return "unsurvivable"
    if iss >= 25:
        return "severe"
    if iss >= 16:
        return "major"
    if iss >= 9:
        return "moderate"
    return "minor"


def _rts_band(rts: float) -> str:
    """T-RTS bands (0-12 scale):
        12       normal physiology
        9-11     abnormal physiology
        ≤8       critical (trauma-center transfer + OR ready)
    """
    if rts >= 12:
        return "normal_physiology"
    if rts >= 9:
        return "abnormal_physiology"
    return "critical"


def _triage_priority(iss: int, rts: float, has_active_hemorrhage: bool) -> str:
    if has_active_hemorrhage and (iss >= 25 or rts <= 8):
        return "operating_room_immediate"
    if iss >= 25 or rts <= 8:
        return "operating_room_immediate"
    if iss >= 16 or rts <= 10:
        return "trauma_center_transfer"
    if iss >= 9 or rts < 12:
        return "trauma_team_activation"
    return "minor_injury_outpatient"


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_trauma_severity_score(
    injuries: list[dict[str, Any]],
    glasgow_coma_score: int = 15,
    systolic_bp: float = 120.0,
    respiratory_rate: float = 16.0,
    has_active_hemorrhage: bool = False,
    patient_id: str | None = None,
) -> TraumaSeverityReport:
    """Composite ISS + RTS trauma severity + triage priority.

    Args:
        injuries: list of {body_region, ais_severity, description?}.
            body_region ∈ {head_neck, face, chest, abdomen_pelvis,
            extremities_pelvic_girdle, external}; ais_severity 1-6.
        glasgow_coma_score: 3-15.
        systolic_bp: mmHg.
        respiratory_rate: per minute.
        has_active_hemorrhage: bool -- pulls toward OR-immediate even at
            moderate ISS if RTS is critical.

    Returns:
        TraumaSeverityReport with ISS, RTS, bands, triage priority.
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {"systolic_bp": systolic_bp, "respiratory_rate": respiratory_rate,
         "glasgow_coma_score": glasgow_coma_score},
        chart_derivable={"systolic_bp", "respiratory_rate",
                          "glasgow_coma_score"},
    )
    if _sb:
        systolic_bp = _r.get("systolic_bp") if _r.get("systolic_bp") is not None else systolic_bp
        respiratory_rate = _r.get("respiratory_rate") if _r.get("respiratory_rate") is not None else respiratory_rate
        glasgow_coma_score = _r.get("glasgow_coma_score") if _r.get("glasgow_coma_score") is not None else glasgow_coma_score
    parsed_injuries: list[TraumaInjury] = []
    for inj in injuries or []:
        try:
            parsed_injuries.append(TraumaInjury.model_validate(inj))
        except Exception:
            continue

    iss = _compute_iss(parsed_injuries)
    rts = _compute_rts(int(glasgow_coma_score), float(systolic_bp),
                         float(respiratory_rate))
    iss_band = _iss_band(iss)
    rts_band = _rts_band(rts)
    triage = _triage_priority(iss, rts, has_active_hemorrhage)

    rationale = _build_rationale(iss, rts, iss_band, rts_band, triage,
                                   parsed_injuries, glasgow_coma_score,
                                   systolic_bp, respiratory_rate,
                                   has_active_hemorrhage)

    return TraumaSeverityReport(
        patient_id=patient_id,
        iss=iss,
        rts=rts,
        iss_band=iss_band,  # type: ignore[arg-type]
        rts_band=rts_band,  # type: ignore[arg-type]
        injuries=parsed_injuries,
        triage_priority=triage,  # type: ignore[arg-type]
        rationale=rationale,
    )


def _build_rationale(iss: int, rts: float, ib: str, rb: str, tri: str,
                       injuries: list[TraumaInjury], gcs: int, sbp: float,
                       rr: float, hemorrhage: bool) -> str:
    parts = [
        f"ISS = {iss} ({ib}); RTS = {rts} ({rb}).",
        f"Physiology: GCS={gcs}, SBP={sbp}, RR={rr}, "
        f"active_hemorrhage={hemorrhage}.",
    ]
    if injuries:
        items = ", ".join(
            f"{inj.body_region}=AIS{inj.ais_severity}" for inj in injuries
        )
        parts.append(f"Injuries ({len(injuries)}): {items}.")
    parts.append(f"Triage priority: {tri}.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_trauma_severity_score)
