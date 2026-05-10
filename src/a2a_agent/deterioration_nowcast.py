"""TIME-3 -- Real-time deterioration nowcasting.

Sliding-window analysis over recent vital-sign Observations to detect
trends that exceed a subgroup-specific baseline. Triggers an `alert` tier
when ≥2 vitals trend abnormal in the same window.

Window default: 6 hours. Vitals tracked:
  - heart_rate           (trend > +20 bpm or > 120 absolute -> alert)
  - respiratory_rate     (trend > +5 /min or > 24 absolute -> alert)
  - systolic_bp          (trend < −20 mmHg or < 90 absolute -> alert)
  - oxygen_saturation    (trend < −3% or < 92% absolute -> alert)
  - temperature_c        (trend > +1.0°C or > 38.5 absolute -> alert)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from shared.schemas import DeteriorationNowcast


# ─────────────────────── Vital-sign canonicalization ───────────────────────

_VITAL_LOINC: dict[str, str] = {
    "8867-4": "heart_rate",
    "9279-1": "respiratory_rate",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "8310-5": "temperature_c",
    "59408-5": "oxygen_saturation",
}


_VITAL_NAME_KEYWORDS: dict[str, list[str]] = {
    "heart_rate": ["heart rate", "pulse"],
    "respiratory_rate": ["respiratory rate", "resp rate"],
    "systolic_bp": ["systolic", "sbp"],
    "diastolic_bp": ["diastolic", "dbp"],
    "temperature_c": ["temperature", "temp"],
    "oxygen_saturation": ["oxygen saturation", "spo2", "sao2"],
}


def _canonical_vital(obs: dict[str, Any]) -> str | None:
    code = obs.get("code") or {}
    for coding in code.get("coding", []) or []:
        c = (coding.get("code") or "").strip()
        if c in _VITAL_LOINC:
            return _VITAL_LOINC[c]
    text = (code.get("text") or "").lower()
    for canonical, keywords in _VITAL_NAME_KEYWORDS.items():
        if any(k in text for k in keywords):
            return canonical
    return None


def _value(obs: dict[str, Any]) -> float | None:
    vq = obs.get("valueQuantity") or {}
    v = vq.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _timestamp(obs: dict[str, Any]) -> datetime | None:
    for key in ("effectiveDateTime", "issued"):
        v = obs.get(key)
        if isinstance(v, str):
            try:
                s = v.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s[:25] if len(s) > 25 else s)
                return (dt if dt.tzinfo else
                            dt.replace(tzinfo=timezone.utc))
            except Exception:
                continue
    return None


# ─────────────────────── Trend computation ───────────────────────

def _slope_per_hour(points: list[tuple[datetime, float]]) -> float:
    """Simple least-squares slope, units of value/hour."""
    if len(points) < 2:
        return 0.0
    base_t = points[0][0]
    xs = [(p[0] - base_t).total_seconds() / 3600.0 for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


# ─────────────────────── Threshold matrix ───────────────────────
#
# (vital, slope_threshold_per_hour, abs_threshold, comparison_direction).
# 'positive' = slope > threshold or abs > threshold triggers
# 'negative' = slope < threshold or abs < threshold triggers

_THRESHOLDS: list[tuple[str, float, float, str]] = [
    ("heart_rate",         3.0,  120.0, "positive"),
    ("respiratory_rate",   1.0,   24.0, "positive"),
    ("systolic_bp",       -3.0,   90.0, "negative"),
    ("oxygen_saturation", -0.5,   92.0, "negative"),
    ("temperature_c",      0.15,  38.5, "positive"),
]


def nowcast_deterioration(
    observations: list[dict[str, Any]],
    *,
    window_hours: int = 6,
    now: datetime | None = None,
) -> DeteriorationNowcast:
    """Compute a sliding-window deterioration nowcast over vital signs.

    Args:
        observations: list of FHIR Observation dicts.
        window_hours: window size in hours (default 6).
        now: fixed reference timestamp for testing; defaults to UTC now.

    Returns:
        DeteriorationNowcast with per-vital slope (value/hour) +
        triggered_signals + overall alert_tier.
    """
    if not isinstance(observations, list):
        raise ValueError("observations must be a list of dicts.")
    if window_hours < 1 or window_hours > 48:
        raise ValueError("window_hours must be in [1, 48].")

    ref = now or datetime.now(timezone.utc)
    window_start = ref - timedelta(hours=window_hours)

    # Group observations by canonical vital, retain only those in-window
    by_vital: dict[str, list[tuple[datetime, float]]] = {}
    for o in observations:
        if not isinstance(o, dict):
            continue
        vital = _canonical_vital(o)
        if vital is None:
            continue
        v = _value(o)
        t = _timestamp(o)
        if v is None or t is None:
            continue
        if t < window_start or t > ref:
            continue
        by_vital.setdefault(vital, []).append((t, v))

    # Sort each series ascending in time
    for v in by_vital.values():
        v.sort(key=lambda x: x[0])

    trends: dict[str, float] = {}
    signals: list[str] = []
    for vital, slope_thr, abs_thr, direction in _THRESHOLDS:
        series = by_vital.get(vital, [])
        if not series:
            continue
        slope = _slope_per_hour(series)
        trends[vital] = round(slope, 4)
        latest_value = series[-1][1]
        if direction == "positive":
            if slope > slope_thr:
                signals.append(f"{vital}_trend_up({slope:+.2f}/h)")
            if latest_value > abs_thr:
                signals.append(f"{vital}_abs_high({latest_value:.1f})")
        else:
            if slope < slope_thr:
                signals.append(f"{vital}_trend_down({slope:+.2f}/h)")
            if latest_value < abs_thr:
                signals.append(f"{vital}_abs_low({latest_value:.1f})")

    if len(signals) >= 2:
        tier = "alert"
    elif len(signals) == 1:
        tier = "watch"
    else:
        tier = "ok"

    n_in_window = sum(len(v) for v in by_vital.values())
    rationale = (
        f"Window {window_hours}h ending {ref.isoformat()}: "
        f"{n_in_window} vital observations across "
        f"{len(by_vital)} parameter(s). "
        f"{len(signals)} threshold signal(s) -> tier={tier}."
    )

    return DeteriorationNowcast(
        window_start_iso=window_start.isoformat(),
        window_end_iso=ref.isoformat(),
        n_observations=n_in_window,
        trends=trends,
        alert_tier=tier,                      # type: ignore[arg-type]
        triggered_signals=signals,
        rationale=rationale,
    )
