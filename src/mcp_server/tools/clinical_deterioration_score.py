"""healthcare.compute_clinical_deterioration_score -- NEWS2 + trend.

Per the Royal College of Physicians 2017 update (NEWS2):
  https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2

Each of 7 parameters scores 0-3 points; total 0-20 (a single 3 in any
parameter alone triggers a clinical review). We extend stock NEWS2 with a
simple trend delta -- change vs the prior assessment in the input series --
that flags worsening before the absolute score crosses an escalation
threshold.

Inputs are kept transport-agnostic: the tool accepts either an explicit
list of `VitalSign` dicts (stateless mode, useful for batch ICU dashboards
or for callers that already have the structured vitals), or it falls back
to extracting Observation resources from the FHIR bundle.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from shared.schemas import (
    DeteriorationFactor,
    DeteriorationReport,
    VitalSign,
)

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


# ─────────────────────────────────────────────────────────────────────
# NEWS2 score table (RCP 2017)
# ─────────────────────────────────────────────────────────────────────
# Each entry: list of (lower_bound, upper_bound, points). Bounds are
# inclusive on the lower side, exclusive on the upper, except the open-end
# bracket. None means "open-ended" toward that side.

_NEWS2_RR: list[tuple[float | None, float | None, int]] = [
    (None, 9, 3),     # ≤ 8
    (9, 12, 1),       # 9-11
    (12, 21, 0),      # 12-20
    (21, 25, 2),      # 21-24
    (25, None, 3),    # ≥ 25
]

_NEWS2_SPO2_SCALE_1: list[tuple[float | None, float | None, int]] = [
    (None, 92, 3),    # ≤ 91
    (92, 94, 2),      # 92-93
    (94, 96, 1),      # 94-95
    (96, None, 0),    # ≥ 96
]

_NEWS2_SBP: list[tuple[float | None, float | None, int]] = [
    (None, 91, 3),    # ≤ 90
    (91, 101, 2),     # 91-100
    (101, 111, 1),    # 101-110
    (111, 220, 0),    # 111-219
    (220, None, 3),   # ≥ 220
]

_NEWS2_HR: list[tuple[float | None, float | None, int]] = [
    (None, 41, 3),    # ≤ 40
    (41, 51, 1),      # 41-50
    (51, 91, 0),      # 51-90
    (91, 111, 1),     # 91-110
    (111, 131, 2),    # 111-130
    (131, None, 3),   # ≥ 131
]

_NEWS2_TEMP: list[tuple[float | None, float | None, int]] = [
    (None, 35.1, 3),  # ≤ 35.0
    (35.1, 36.1, 1),  # 35.1-36.0
    (36.1, 38.1, 0),  # 36.1-38.0
    (38.1, 39.1, 1),  # 38.1-39.0
    (39.1, None, 2),  # ≥ 39.1
]


def _bracket_score(value: float, table: list[tuple[float | None, float | None, int]]) -> int:
    for lo, hi, points in table:
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return points
    return 0


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_clinical_deterioration_score(
    vital_signs: list[dict[str, Any]] | None = None,
    prior_score: int | None = None,
    patient_id: str | None = None,
) -> DeteriorationReport:
    """Compute NEWS2 + trend on the most recent vital-sign set.

    Args:
        vital_signs: explicit list of vital observations (stateless mode).
            Each dict must have `type`, `value`, optionally `unit` and
            `observed_at`. If None, vitals are extracted from the FHIR bundle.
        prior_score: previous NEWS2 total for trend computation. None ->
            trend_direction = "unknown".
        patient_id: FHIR Patient ID -- required if vital_signs is None.

    Returns:
        DeteriorationReport with score, parameter breakdown, severity tier,
        recommended response, and trend (if prior_score given).
    """
    pid = patient_id
    if vital_signs is None:
        pid = await resolve_patient_id(patient_id)
        bundle = await fetch_patient_bundle(pid)
        vitals = _extract_vitals_from_bundle(bundle)
    else:
        vitals = _coerce_vitals(vital_signs)

    if not vitals:
        # No vitals -> abstain. Returning a "score 0 / low tier" report
        # would let a downstream consumer treat the absence of data as
        # a clean low-acuity finding, which is clinically dangerous.
        return DeteriorationReport(
            patient_id=pid,
            score_total=0,
            parameter_contributions=[],
            severity_tier="low",
            recommended_response="routine_12h_obs",
            trend_delta=0,
            trend_direction="unknown",
            rationale=(
                "NEWS2 abstained: no vital signs available. The absence "
                "of vitals is NOT a low-acuity finding. Recommend manual "
                "chart review before any escalation decision."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_vital_signs: NEWS2 requires at least one of "
                "{respiratory_rate, spo2, systolic_bp, heart_rate, "
                "temperature, consciousness} to compute a score. The "
                "numeric score and severity tier in this report are "
                "placeholder values, not a calibrated estimate."
            ),
        )

    # Pick the most recent value per parameter type
    latest: dict[str, VitalSign] = {}
    for v in vitals:
        if v.type not in latest or v.observed_at > latest[v.type].observed_at:
            latest[v.type] = v

    contributions: list[DeteriorationFactor] = []
    total = 0

    if "respiratory_rate" in latest:
        v = latest["respiratory_rate"]
        pts = _bracket_score(float(v.value), _NEWS2_RR)
        total += pts
        contributions.append(DeteriorationFactor(
            parameter="respiratory_rate",
            value=v.value,
            points=pts,
            rationale=_rr_rationale(float(v.value), pts),
        ))

    if "spo2" in latest:
        v = latest["spo2"]
        pts = _bracket_score(float(v.value), _NEWS2_SPO2_SCALE_1)
        total += pts
        contributions.append(DeteriorationFactor(
            parameter="spo2",
            value=v.value,
            points=pts,
            rationale=_spo2_rationale(float(v.value), pts),
        ))

    if "supplemental_oxygen" in latest:
        v = latest["supplemental_oxygen"]
        on_o2 = _is_truthy(v.value)
        pts = 2 if on_o2 else 0
        total += pts
        contributions.append(DeteriorationFactor(
            parameter="supplemental_oxygen",
            value=str(v.value),
            points=pts,
            rationale=("Patient is on supplemental O2 (+2)" if on_o2
                       else "Patient on room air (0)"),
        ))

    if "systolic_bp" in latest:
        v = latest["systolic_bp"]
        pts = _bracket_score(float(v.value), _NEWS2_SBP)
        total += pts
        contributions.append(DeteriorationFactor(
            parameter="systolic_bp",
            value=v.value,
            points=pts,
            rationale=_sbp_rationale(float(v.value), pts),
        ))

    if "heart_rate" in latest:
        v = latest["heart_rate"]
        pts = _bracket_score(float(v.value), _NEWS2_HR)
        total += pts
        contributions.append(DeteriorationFactor(
            parameter="heart_rate",
            value=v.value,
            points=pts,
            rationale=_hr_rationale(float(v.value), pts),
        ))

    if "consciousness" in latest:
        v = latest["consciousness"]
        avpu = str(v.value).strip().upper()[:1]
        pts = 0 if avpu == "A" else 3
        total += pts
        contributions.append(DeteriorationFactor(
            parameter="consciousness",
            value=str(v.value),
            points=pts,
            rationale=("Alert (A): 0 points" if avpu == "A"
                       else f"AVPU={v.value} (not Alert): +3 points"),
        ))

    if "temperature" in latest:
        v = latest["temperature"]
        pts = _bracket_score(float(v.value), _NEWS2_TEMP)
        total += pts
        contributions.append(DeteriorationFactor(
            parameter="temperature",
            value=v.value,
            points=pts,
            rationale=_temp_rationale(float(v.value), pts),
        ))

    severity, response = _classify_severity(total, contributions)
    trend_delta, trend_direction = _classify_trend(total, prior_score)
    rationale = _build_rationale(total, contributions, severity, response,
                                  trend_delta, trend_direction)

    return DeteriorationReport(
        patient_id=pid,
        score_total=total,
        parameter_contributions=contributions,
        severity_tier=severity,  # type: ignore[arg-type]
        recommended_response=response,  # type: ignore[arg-type]
        trend_delta=trend_delta,
        trend_direction=trend_direction,  # type: ignore[arg-type]
        rationale=rationale,
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _classify_severity(
    total: int, contribs: list[DeteriorationFactor],
) -> tuple[str, str]:
    """RCP NEWS2 thresholds:
        0      -> low: routine 12-h observation
        1-4    -> low_medium: minimum 4-6h observation
        5-6    -> medium: urgent ward-based response within 1h
        ≥7     -> high: emergency response (rapid response/ICU)
        any single parameter scoring 3 -> escalate to medium tier minimum.
    """
    any_three = any(c.points >= 3 for c in contribs)
    if total >= 7:
        return "high", "emergency_response"
    if total >= 5 or (any_three and total >= 3):
        return "medium", "urgent_review_1h"
    if total >= 1:
        return "low_medium", "increase_obs_4_6h"
    return "low", "routine_12h_obs"


def _classify_trend(current: int, prior: int | None) -> tuple[int, str]:
    if prior is None:
        return 0, "unknown"
    delta = current - prior
    if delta >= 2:
        return delta, "worsening"
    if delta <= -2:
        return delta, "improving"
    return delta, "stable"


def _build_rationale(total: int, contribs: list[DeteriorationFactor],
                      severity: str, response: str, trend_delta: int,
                      trend_direction: str) -> str:
    top = sorted(contribs, key=lambda c: c.points, reverse=True)[:3]
    parts = [f"NEWS2 total = {total} ({severity} tier -> {response})."]
    if top and top[0].points > 0:
        items = ", ".join(f"{c.parameter}={c.value} (+{c.points})"
                           for c in top if c.points > 0)
        parts.append(f"Highest-contribution parameters: {items}.")
    if trend_direction != "unknown":
        if trend_direction == "worsening":
            parts.append(
                f"Trend: WORSENING (Δ +{trend_delta} vs prior); patient "
                f"is deteriorating despite current management."
            )
        elif trend_direction == "improving":
            parts.append(
                f"Trend: improving (Δ {trend_delta} vs prior)."
            )
        else:
            parts.append(f"Trend: stable (Δ {trend_delta:+d}).")
    return " ".join(parts)


def _rr_rationale(rr: float, pts: int) -> str:
    if pts == 0:
        return f"RR={rr}/min within normal (12-20)"
    if rr <= 8:
        return f"RR={rr}/min ≤ 8 (severe bradypnea, +3)"
    if rr <= 11:
        return f"RR={rr}/min 9-11 (mild bradypnea, +1)"
    if rr <= 24:
        return f"RR={rr}/min 21-24 (tachypnea, +2)"
    return f"RR={rr}/min ≥ 25 (severe tachypnea, +3)"


def _spo2_rationale(spo2: float, pts: int) -> str:
    if pts == 0:
        return f"SpO₂={spo2}% ≥ 96 (normal)"
    return f"SpO₂={spo2}% (+{pts}, NEWS2 Scale 1 -- non-COPD baseline)"


def _sbp_rationale(sbp: float, pts: int) -> str:
    if pts == 0:
        return f"SBP={sbp} mmHg within normal (111-219)"
    if sbp <= 90:
        return f"SBP={sbp} mmHg ≤ 90 (hypotension, +3)"
    if sbp <= 100:
        return f"SBP={sbp} mmHg 91-100 (mild hypotension, +2)"
    if sbp <= 110:
        return f"SBP={sbp} mmHg 101-110 (borderline, +1)"
    return f"SBP={sbp} mmHg ≥ 220 (severe hypertension, +3)"


def _hr_rationale(hr: float, pts: int) -> str:
    if pts == 0:
        return f"HR={hr}/min within normal (51-90)"
    if hr <= 40:
        return f"HR={hr}/min ≤ 40 (severe bradycardia, +3)"
    if hr <= 50:
        return f"HR={hr}/min 41-50 (mild bradycardia, +1)"
    if hr <= 110:
        return f"HR={hr}/min 91-110 (mild tachycardia, +1)"
    if hr <= 130:
        return f"HR={hr}/min 111-130 (moderate tachycardia, +2)"
    return f"HR={hr}/min ≥ 131 (severe tachycardia, +3)"


def _temp_rationale(t: float, pts: int) -> str:
    if pts == 0:
        return f"T={t}°C within normal (36.1-38.0)"
    if t <= 35.0:
        return f"T={t}°C ≤ 35.0 (hypothermia, +3)"
    if t <= 36.0:
        return f"T={t}°C 35.1-36.0 (mild hypothermia, +1)"
    if t <= 39.0:
        return f"T={t}°C 38.1-39.0 (mild fever, +1)"
    return f"T={t}°C ≥ 39.1 (high fever, +2)"


def _is_truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) > 0.0
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "1", "on", "supplemental")
    return False


def _coerce_vitals(items: Iterable[Any]) -> list[VitalSign]:
    out: list[VitalSign] = []
    for it in items:
        if isinstance(it, VitalSign):
            out.append(it)
            continue
        if not isinstance(it, dict):
            continue
        try:
            data = dict(it)
            if "observed_at" not in data:
                data["observed_at"] = datetime.now(timezone.utc).isoformat()
            out.append(VitalSign.model_validate(data))
        except Exception:
            continue
    return out


# Map LOINC + display-text patterns -> NEWS2 vital types
_VITAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"respiratory[\s_]?rate|breaths.*minute", re.I), "respiratory_rate"),
    (re.compile(r"\bspo2\b|oxygen[\s_]saturation|pulse\s*ox", re.I), "spo2"),
    (re.compile(r"systolic.*pressure|systolic\s*bp", re.I), "systolic_bp"),
    (re.compile(r"\bheart[\s_]?rate\b|pulse\s*rate", re.I), "heart_rate"),
    (re.compile(r"\btemperature\b|body[\s_]temp", re.I), "temperature"),
    (re.compile(r"avpu|consciousness|alertness|gcs", re.I), "consciousness"),
    (re.compile(r"oxygen.*therapy|supplemental.*oxygen|on[\s_]?O2", re.I),
     "supplemental_oxygen"),
]


def _extract_vitals_from_bundle(bundle: dict[str, Any]) -> list[VitalSign]:
    out: list[VitalSign] = []
    for entry in bundle.get("entry") or []:
        r = (entry or {}).get("resource") or {}
        if r.get("resourceType") != "Observation":
            continue
        code = r.get("code") or {}
        text = str(code.get("text") or "")
        kind: str | None = None
        for pat, k in _VITAL_PATTERNS:
            if pat.search(text):
                kind = k
                break
        if kind is None:
            continue
        vq = r.get("valueQuantity") or {}
        value = vq.get("value")
        if value is None:
            continue
        unit = vq.get("unit")
        observed = r.get("effectiveDateTime") or r.get("issued")
        try:
            ts = (datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
                  if observed else datetime.now(timezone.utc))
        except (TypeError, ValueError):
            ts = datetime.now(timezone.utc)
        out.append(VitalSign(
            type=kind,  # type: ignore[arg-type]
            value=float(value),
            unit=unit,
            observed_at=ts,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_clinical_deterioration_score)
