"""healthcare.compute_pediatric_early_warning -- Brighton/Monaghan PEWS.

Pediatric vitals are NOT just adult vitals scaled -- both normal ranges and
escalation thresholds shift with age. This tool implements PEWS with 4
age bands (0-11mo, 1-4y, 5-11y, 12-17y) and 4 sub-scores (behavior,
cardiovascular, respiratory, other-concern). Total 0-12 with 3 escalation
tiers.

References:
  Monaghan A. Detecting and managing deterioration in children. Paediatric
    Nursing 2005;17(1):32-35.
  Brighton & Sussex University Hospitals NHS PEWS Implementation Guide (2022).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from shared.schemas import (
    PEWSAgeBand,
    PEWSComponent,
    PEWSReport,
)

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# Age-band reference ranges (heart rate / respiratory rate / SBP)
# ─────────────────────────────────────────────────────────────────────

_AGE_BANDS: list[tuple[int, int, str, dict[str, tuple[int, int]]]] = [
    # (lo_months_inclusive, hi_months_exclusive, label, normals)
    (0, 12, "0-11mo",
     {"hr": (100, 160), "rr": (30, 50), "sbp": (75, 100)}),
    (12, 60, "1-4y",
     {"hr": (90, 140), "rr": (24, 40), "sbp": (80, 110)}),
    (60, 144, "5-11y",
     {"hr": (70, 120), "rr": (18, 30), "sbp": (90, 120)}),
    (144, 217, "12-17y",
     {"hr": (60, 100), "rr": (12, 20), "sbp": (100, 130)}),
]


def _age_band(age_months: int) -> tuple[str, dict[str, tuple[int, int]]]:
    for lo, hi, label, normals in _AGE_BANDS:
        if lo <= age_months < hi:
            return label, normals
    # Out-of-range default to oldest pediatric band
    return "12-17y", _AGE_BANDS[-1][3]


# ─────────────────────────────────────────────────────────────────────
# PEWS sub-scores (Monaghan)
# ─────────────────────────────────────────────────────────────────────

# Behavior: 0=playing/appropriate, 1=sleeping, 2=irritable, 3=lethargic / reduced response to pain
def _behavior_score(behavior: str) -> int:
    b = (behavior or "").strip().lower()
    if b in ("playing", "appropriate", "alert", "normal"):
        return 0
    if b in ("sleeping", "drowsy_appropriate"):
        return 1
    if b in ("irritable", "consolable_with_difficulty"):
        return 2
    if b in ("lethargic", "confused", "reduced_response_to_pain", "obtunded"):
        return 3
    return 0  # unknown defaults to 0


def _cv_score(hr: float | None, cap_refill_seconds: float | None,
              sbp: float | None, normals: dict[str, tuple[int, int]]) -> tuple[int, str]:
    """Cardiovascular: combination of HR, capillary refill, and skin/SBP."""
    pts = 0
    notes: list[str] = []

    if hr is not None:
        lo, hi = normals["hr"]
        if hr > hi + 30:
            pts = max(pts, 3)
            notes.append(f"HR {hr} > {hi + 30}/min (severe tachycardia)")
        elif hr > hi:
            pts = max(pts, 2)
            notes.append(f"HR {hr} > {hi}/min (tachycardia)")
        elif hr < lo - 20:
            pts = max(pts, 3)
            notes.append(f"HR {hr} < {lo - 20}/min (severe bradycardia)")
        elif hr < lo:
            pts = max(pts, 1)
            notes.append(f"HR {hr} < {lo}/min (mild bradycardia)")

    if cap_refill_seconds is not None:
        if cap_refill_seconds >= 4:
            pts = max(pts, 3)
            notes.append(f"capillary refill {cap_refill_seconds}s ≥ 4s")
        elif cap_refill_seconds >= 3:
            pts = max(pts, 2)
            notes.append(f"capillary refill {cap_refill_seconds}s")

    if sbp is not None:
        lo, hi = normals["sbp"]
        if sbp < lo - 20:
            pts = max(pts, 3)
            notes.append(f"SBP {sbp} < {lo - 20} mmHg (severe hypotension)")
        elif sbp < lo:
            pts = max(pts, 2)
            notes.append(f"SBP {sbp} < {lo} mmHg (hypotension)")

    if not notes and pts == 0:
        notes.append("HR, capillary refill, and SBP within age-appropriate normals.")
    return pts, "; ".join(notes)


def _resp_score(rr: float | None, spo2: float | None,
                accessory_muscle_use: bool, on_oxygen: bool,
                normals: dict[str, tuple[int, int]]) -> tuple[int, str]:
    pts = 0
    notes: list[str] = []

    if rr is not None:
        lo, hi = normals["rr"]
        if rr > hi + 20:
            pts = max(pts, 3)
            notes.append(f"RR {rr} > {hi + 20}/min (severe tachypnea)")
        elif rr > hi + 5:
            pts = max(pts, 2)
            notes.append(f"RR {rr} > {hi + 5}/min (tachypnea)")
        elif rr < lo:
            pts = max(pts, 1)
            notes.append(f"RR {rr} < {lo}/min (bradypnea)")

    if spo2 is not None:
        if spo2 < 90:
            pts = max(pts, 3)
            notes.append(f"SpO₂ {spo2}% (severe hypoxemia)")
        elif spo2 < 94:
            pts = max(pts, 2)
            notes.append(f"SpO₂ {spo2}% (hypoxemia)")

    if accessory_muscle_use:
        pts = max(pts, 2)
        notes.append("accessory muscle use / retractions")

    if on_oxygen:
        pts = max(pts, 1)
        notes.append("on supplemental oxygen")

    if not notes:
        notes.append("Respiratory parameters within normal limits.")
    return pts, "; ".join(notes)


def _other_concern_score(persistent_vomiting: bool,
                          parental_or_nurse_concern: bool) -> tuple[int, str]:
    pts = 0
    notes: list[str] = []
    if persistent_vomiting:
        pts = max(pts, 1)
        notes.append("persistent vomiting")
    if parental_or_nurse_concern:
        pts = max(pts, 2)
        notes.append("parental/nurse expressed concern (key softer-signal predictor)")
    if not notes:
        notes.append("No additional concerns reported.")
    return pts, "; ".join(notes)


# ─────────────────────────────────────────────────────────────────────
# Severity classification
# ─────────────────────────────────────────────────────────────────────

def _classify_severity(total: int, components: list[PEWSComponent]
                        ) -> tuple[str, str]:
    """PEWS escalation:
      0-2 -> low: routine 4-h observations
      3-4 -> medium: 1-h obs + bedside review
      5-6 -> high: urgent pediatric review within 30 min
      ≥7  -> high: rapid response / PICU consult
      Any single component scoring 3 -> escalate to high tier minimum.
    """
    any_three = any(c.points >= 3 for c in components)
    if total >= 7:
        return "high", "rapid_response_picu_consult"
    if total >= 5 or (any_three and total >= 3):
        return "high", "urgent_pediatric_review"
    if total >= 3:
        return "medium", "increase_to_hourly_obs"
    return "low", "routine_obs"


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_pediatric_early_warning(
    age_months: int,
    behavior: str = "appropriate",
    heart_rate: float | None = None,
    respiratory_rate: float | None = None,
    spo2: float | None = None,
    systolic_bp: float | None = None,
    capillary_refill_seconds: float | None = None,
    accessory_muscle_use: bool = False,
    on_supplemental_oxygen: bool = False,
    persistent_vomiting: bool = False,
    parental_or_nurse_concern: bool = False,
    patient_id: str | None = None,
) -> PEWSReport:
    """Compute Brighton/Monaghan Pediatric Early Warning Score.

    Args:
        age_months: 0-216 (up to 17y inclusive). Out-of-range falls back to 12-17y.
        behavior: enum-like string describing behavior. See _behavior_score for
            accepted values.
        heart_rate / respiratory_rate / spo2 / systolic_bp: per-vital values.
            Each is graded against the age-band normals.
        capillary_refill_seconds: peripheral capillary refill time.
        accessory_muscle_use: bool. Common in pediatric respiratory distress.
        on_supplemental_oxygen: bool. +1 to respiratory sub-score.
        persistent_vomiting: bool. +1 to other-concern sub-score.
        parental_or_nurse_concern: bool. +2 to other-concern sub-score (the
            softer-signal predictor that consistently outperforms isolated
            vital trends in the pediatric literature).
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {"age_months": age_months, "heart_rate": heart_rate,
         "respiratory_rate": respiratory_rate, "spo2": spo2,
         "systolic_bp": systolic_bp},
        chart_derivable={"age_months", "heart_rate", "respiratory_rate",
                          "spo2", "systolic_bp"},
    )
    if _sb:
        age_months = _r.get("age_months") if _r.get("age_months") is not None else age_months
        heart_rate = _r.get("heart_rate") if _r.get("heart_rate") is not None else heart_rate
        respiratory_rate = _r.get("respiratory_rate") if _r.get("respiratory_rate") is not None else respiratory_rate
        spo2 = _r.get("spo2") if _r.get("spo2") is not None else spo2
        systolic_bp = _r.get("systolic_bp") if _r.get("systolic_bp") is not None else systolic_bp
    # Clamp to the PEWSReport schema bounds (0-216) to avoid Pydantic
    # validation errors on adversarial inputs (negative or >18y).
    age_months = max(0, min(216, age_months))
    label, normals = _age_band(age_months)

    # If every vital is missing the PEWS becomes purely a behavior +
    # parental-concern score, which is not how the tool is meant to be
    # used. Abstain so a downstream consumer cannot present a "low
    # score" as a calibrated finding.
    all_vitals_missing = all(
        v is None for v in (heart_rate, respiratory_rate, spo2,
                                 systolic_bp, capillary_refill_seconds)
    )
    if all_vitals_missing:
        return PEWSReport(
            patient_id=patient_id,
            age_months=age_months,
            age_band=label,  # type: ignore[arg-type]
            score_total=0,
            components=[],
            severity_tier="low",
            recommended_response="routine_obs",
            age_band_reference=PEWSAgeBand(
                band_label=label,  # type: ignore[arg-type]
                hr_normal=normals["hr"],
                rr_normal=normals["rr"],
                sbp_normal=normals["sbp"],
            ),
            rationale=(
                "PEWS abstained: no vital signs supplied (all of "
                "heart_rate, respiratory_rate, spo2, systolic_bp, "
                "capillary_refill are missing). The placeholder score "
                "of 0 is NOT a calibrated finding."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_vital_signs: PEWS requires at least one of "
                "{heart_rate, respiratory_rate, spo2, systolic_bp, "
                "capillary_refill_seconds} to compute a score."
            ),
        )

    behavior_pts = _behavior_score(behavior)
    cv_pts, cv_notes = _cv_score(heart_rate, capillary_refill_seconds,
                                   systolic_bp, normals)
    resp_pts, resp_notes = _resp_score(respiratory_rate, spo2,
                                         accessory_muscle_use,
                                         on_supplemental_oxygen, normals)
    other_pts, other_notes = _other_concern_score(persistent_vomiting,
                                                    parental_or_nurse_concern)

    components = [
        PEWSComponent(component="behavior", raw_value=behavior,
                       points=behavior_pts,
                       rationale=_behavior_rationale(behavior, behavior_pts)),
        PEWSComponent(component="cardiovascular",
                       raw_value=f"HR={heart_rate}, cap_refill={capillary_refill_seconds}, SBP={systolic_bp}",
                       points=cv_pts, rationale=cv_notes),
        PEWSComponent(component="respiratory",
                       raw_value=f"RR={respiratory_rate}, SpO2={spo2}, "
                                  f"accessory={accessory_muscle_use}, O2={on_supplemental_oxygen}",
                       points=resp_pts, rationale=resp_notes),
        PEWSComponent(component="other_concern",
                       raw_value=f"vomiting={persistent_vomiting}, concern={parental_or_nurse_concern}",
                       points=other_pts, rationale=other_notes),
    ]

    total = sum(c.points for c in components)
    severity, response = _classify_severity(total, components)

    rationale = _build_rationale(total, severity, response, components, label)

    return PEWSReport(
        patient_id=patient_id,
        age_months=age_months,
        age_band=label,  # type: ignore[arg-type]
        score_total=total,
        components=components,
        severity_tier=severity,  # type: ignore[arg-type]
        recommended_response=response,  # type: ignore[arg-type]
        age_band_reference=PEWSAgeBand(
            band_label=label,  # type: ignore[arg-type]
            hr_normal=normals["hr"],
            rr_normal=normals["rr"],
            sbp_normal=normals["sbp"],
        ),
        rationale=rationale,
    )


def _behavior_rationale(behavior: str, pts: int) -> str:
    b = (behavior or "appropriate").lower()
    if pts == 0:
        return f"Behavior = {b!r} (alert / appropriate, 0 points)"
    if pts == 1:
        return f"Behavior = {b!r} (sleeping but rouseable, +1)"
    if pts == 2:
        return f"Behavior = {b!r} (irritable / hard to console, +2)"
    return f"Behavior = {b!r} (lethargic / obtunded / reduced pain response, +3)"


def _build_rationale(total: int, severity: str, response: str,
                      components: list[PEWSComponent], age_band: str) -> str:
    parts = [f"PEWS total = {total} (age band {age_band}, {severity} tier -> {response})."]
    top = sorted(components, key=lambda c: c.points, reverse=True)
    high_contribs = [c for c in top if c.points > 0]
    if high_contribs:
        items = ", ".join(f"{c.component}=+{c.points}" for c in high_contribs)
        parts.append(f"Contributing sub-scores: {items}.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_pediatric_early_warning)
