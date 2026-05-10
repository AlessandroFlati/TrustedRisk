"""healthcare.compute_maternal_early_warning -- MEOWS.

Pregnancy physiology differs materially from non-pregnant adults:
  - HR 80-100 normal in 3rd trimester (vs 60-100)
  - SBP slight drop (~5-10 mmHg) in 2nd trimester
  - DBP drops ~10 mmHg
  - RR slightly elevated due to progesterone effect (16-22 normal)

Standard NEWS2 over-flags pregnant women on baseline tachycardia. MEOWS
(CEMACH 2007 / RCOG 2011) uses pregnancy-adjusted thresholds with traffic-
light coding (green/yellow/red) and sub-tier escalation.

References:
  Singh S et al. A validation study of the CEMACH MEOWS. Anaesthesia 2012.
  RCOG Greentop Guideline 56: Maternal Collapse in Pregnancy and Puerperium.
"""

from __future__ import annotations

from typing import Literal

from shared.schemas import MEOWSReport


# ─────────────────────────────────────────────────────────────────────
# Pregnancy-adjusted vital thresholds (single-trigger color codes)
# ─────────────────────────────────────────────────────────────────────
# Each entry: (parameter, yellow_lo_lo, yellow_hi_hi, red_lo_lo, red_hi_hi)
# Where yellow is single-trigger increase obs; red is rapid response.

_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    "respiratory_rate": {
        "yellow": (10, 30),       # outside is yellow
        "red": (10, 25),          # outside red is RR<10 OR ≥25 -> escalate further only if outside red
    },
    "spo2": {
        "yellow": (95, 100),
        "red": (90, 100),
    },
    "heart_rate": {
        "yellow": (60, 110),
        "red": (50, 130),
    },
    "systolic_bp": {
        "yellow": (90, 150),
        "red": (80, 160),
    },
    "diastolic_bp": {
        "yellow": (50, 90),
        "red": (40, 100),
    },
    "temperature": {
        "yellow": (35.5, 38.0),
        "red": (35.0, 38.5),
    },
}


def _color_for_parameter(name: str, value: float) -> tuple[str, str | None]:
    """Returns (color, parameter_note). color in {green, yellow, red}."""
    th = _THRESHOLDS.get(name)
    if th is None:
        return "green", None
    yl, yh = th["yellow"]
    rl, rh = th["red"]
    if value < rl or value > rh:
        return "red", f"{name}={value} outside red threshold [{rl}, {rh}]"
    if value < yl or value > yh:
        return "yellow", f"{name}={value} outside yellow threshold [{yl}, {yh}]"
    return "green", None


def _avpu_color(avpu: str) -> tuple[str, str | None]:
    if not avpu:
        return "green", None
    a = avpu.strip().upper()[:1]
    if a == "A":
        return "green", None
    if a == "V":
        return "yellow", "AVPU=V (responds to voice -- not fully alert)"
    return "red", f"AVPU={avpu} (responds to pain or unresponsive)"


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_maternal_early_warning(
    gestational_age_weeks: int,
    pregnancy_phase: str = "antepartum",
    respiratory_rate: float | None = None,
    spo2: float | None = None,
    heart_rate: float | None = None,
    systolic_bp: float | None = None,
    diastolic_bp: float | None = None,
    temperature: float | None = None,
    avpu: str | None = None,
    proteinuria_present: bool = False,
    severe_headache_or_visual: bool = False,
    chest_pain_or_dyspnea: bool = False,
    excess_bleeding: bool = False,
    patient_id: str | None = None,
) -> MEOWSReport:
    """Compute MEOWS color + escalation tier.

    Args:
        gestational_age_weeks: 0-44.
        pregnancy_phase: antepartum / intrapartum / postpartum.
        respiratory_rate / spo2 / heart_rate / systolic_bp / diastolic_bp /
            temperature: vital values.
        avpu: A / V / P / U.
        Specific obstetric red flags (proteinuria, severe headache or
            visual disturbance, chest pain or dyspnea, excess bleeding) --
            each red flag pulls toward yellow or red.

    Returns:
        MEOWSReport with severity_tier (green / yellow / red) +
        recommended_response.
    """
    # Abstain when no vital signs and no obstetric red-flag bool was
    # supplied. The placeholder "green" tier would otherwise be reported
    # as a clinical reassurance with no data behind it.
    all_vitals_missing = all(
        v is None for v in (respiratory_rate, spo2, heart_rate,
                                  systolic_bp, diastolic_bp, temperature)
    )
    no_red_flags = not (
        proteinuria_present or severe_headache_or_visual
        or chest_pain_or_dyspnea or excess_bleeding
    )
    if all_vitals_missing and avpu is None and no_red_flags:
        return MEOWSReport(
            patient_id=patient_id,
            gestational_age_weeks=gestational_age_weeks,
            pregnancy_phase=pregnancy_phase,  # type: ignore[arg-type]
            score_total=0,
            severity_tier="green",
            recommended_response="routine_obs_4h",
            contributing_parameters=[],
            rationale=(
                "MEOWS abstained: no vitals, no AVPU, and no obstetric "
                "red flag were supplied. The placeholder 'green' tier "
                "is NOT a clinical reassurance."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_obstetric_inputs: MEOWS requires at least one "
                "of {RR, SpO2, HR, SBP, DBP, T, AVPU} or one obstetric "
                "red flag to compute a meaningful tier."
            ),
        )

    n_yellow = 0
    n_red = 0
    notes: list[str] = []

    for param, value in (
        ("respiratory_rate", respiratory_rate),
        ("spo2", spo2),
        ("heart_rate", heart_rate),
        ("systolic_bp", systolic_bp),
        ("diastolic_bp", diastolic_bp),
        ("temperature", temperature),
    ):
        if value is None:
            continue
        color, note = _color_for_parameter(param, float(value))
        if color == "yellow":
            n_yellow += 1
            if note:
                notes.append(note)
        elif color == "red":
            n_red += 1
            if note:
                notes.append(note)

    if avpu:
        c, note = _avpu_color(avpu)
        if c == "yellow":
            n_yellow += 1
            if note:
                notes.append(note)
        elif c == "red":
            n_red += 1
            if note:
                notes.append(note)

    # Obstetric-specific red flags
    if proteinuria_present:
        n_yellow += 1
        notes.append("proteinuria present (preeclampsia risk)")
    if severe_headache_or_visual:
        n_red += 1
        notes.append("severe headache or visual disturbance (preeclampsia / HELLP red flag)")
    if chest_pain_or_dyspnea:
        n_red += 1
        notes.append("chest pain or dyspnea (PE / amniotic fluid embolus / ACS red flag)")
    if excess_bleeding:
        n_red += 1
        notes.append("excess bleeding (PPH / placental abruption)")

    score = n_yellow + 3 * n_red

    if n_red >= 1 or n_yellow >= 2:
        if n_red >= 1:
            severity = "red"
            response = "obstetric_emergency_response"
        else:
            severity = "yellow"
            response = "urgent_obstetric_review"
    elif n_yellow == 1:
        severity = "yellow"
        response = "increase_to_hourly_obs"
    else:
        severity = "green"
        response = "routine_obs_4h"

    rationale = _build_rationale(gestational_age_weeks, pregnancy_phase,
                                   n_yellow, n_red, severity, response, notes)

    return MEOWSReport(
        patient_id=patient_id,
        gestational_age_weeks=gestational_age_weeks,
        pregnancy_phase=pregnancy_phase,  # type: ignore[arg-type]
        score_total=min(20, score),
        severity_tier=severity,  # type: ignore[arg-type]
        recommended_response=response,  # type: ignore[arg-type]
        contributing_parameters=notes,
        rationale=rationale,
    )


def _build_rationale(ga: int, phase: str, n_yel: int, n_red: int,
                       sev: str, resp: str, notes: list[str]) -> str:
    parts = [f"MEOWS at GA {ga} weeks ({phase}). {n_yel} yellow + {n_red} red triggers -> {sev} tier ({resp})."]
    if notes:
        parts.append("Contributing: " + "; ".join(notes[:5]))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_maternal_early_warning)
