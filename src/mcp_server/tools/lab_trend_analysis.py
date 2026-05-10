"""healthcare.compute_lab_trend_analysis -- quantitative trend over recent labs.

Quantifies whether key labs are *improving*, *declining*, or *flat* over time
by fitting a linear regression on (timestamp, value) pairs per lab and
classifying the slope against a per-lab clinical-direction map.

Why this matters for grounding:

  ground_claim's "vitals stable" sub-claim used to be a keyword match on the
  most recent observation's text. With this tool, the validation layer can
  ask the quantitative question -- "is the BNP actually trending down,
  consistent with the discharge plan?" -- and get a defensible answer with a
  rate-per-day slope and an in-range / out-of-range flag.

Per-lab clinical direction:

  - "lower_is_better"   -> creatinine, BUN, BNP, troponin, lactate
  - "higher_is_better"  -> hemoglobin, oxygen saturation, eGFR
  - "in_range_is_best"  -> sodium, potassium, glucose, INR, calcium

For "in_range_is_best" labs, an *improving* trend means moving toward the
midpoint of the reference range from outside it. A flat-in-range value is
still classified as "flat".

Sources for the curated table: AHRQ MATCH med-recon toolkit (lab->drug class
links) + standard clinical reference ranges (CDC adult-cohort norms). Edit
`_LAB_DIRECTORY` below to extend coverage.
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Iterable

from shared.schemas import LabTrend, LabTrendReport

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


# ─────────────────────── Curated lab directory ───────────────────────

# Each entry: name -> {clinical_direction, reference_range, name_aliases (regex)}
_LAB_DIRECTORY: dict[str, dict[str, Any]] = {
    "hemoglobin": {
        "clinical_direction": "higher_is_better",
        "reference_range": (12.0, 17.0),  # g/dL -- adult average band
        "loinc": "718-7",
        "match": re.compile(r"\bhemoglobin\b|\bhgb\b|\bhb\b", re.IGNORECASE),
    },
    "creatinine": {
        "clinical_direction": "lower_is_better",
        "reference_range": (0.7, 1.3),  # mg/dL -- adult
        "loinc": "2160-0",
        "match": re.compile(r"\bcreatinine\b", re.IGNORECASE),
    },
    "egfr": {
        "clinical_direction": "higher_is_better",
        "reference_range": (60.0, 120.0),  # mL/min/1.73m² -- non-CKD threshold
        "loinc": "33914-3",
        "match": re.compile(r"egfr|estimated\s+glomerular", re.IGNORECASE),
    },
    "bun": {
        "clinical_direction": "lower_is_better",
        "reference_range": (7.0, 20.0),  # mg/dL
        "loinc": "3094-0",
        "match": re.compile(r"\bbun\b|blood\s+urea\s+nitrogen|urea\s+nitrogen", re.IGNORECASE),
    },
    "bnp": {
        "clinical_direction": "lower_is_better",
        "reference_range": (0.0, 100.0),  # pg/mL -- non-HF cutoff
        "loinc": "30934-4",
        "match": re.compile(r"\bbnp\b|nt[\s-]?probnp|natriuretic\s+peptide", re.IGNORECASE),
    },
    "troponin": {
        "clinical_direction": "lower_is_better",
        "reference_range": (0.0, 0.04),  # ng/mL -- high-sensitivity 99th percentile (varies by assay)
        "loinc": "10839-9",
        "match": re.compile(r"troponin", re.IGNORECASE),
    },
    "lactate": {
        "clinical_direction": "lower_is_better",
        "reference_range": (0.5, 2.0),  # mmol/L
        "loinc": "2524-7",
        "match": re.compile(r"\blactate\b|lactic\s+acid", re.IGNORECASE),
    },
    "sodium": {
        "clinical_direction": "in_range_is_best",
        "reference_range": (135.0, 145.0),  # mEq/L
        "loinc": "2951-2",
        "match": re.compile(r"\bsodium\b|\bna\+?\b", re.IGNORECASE),
    },
    "potassium": {
        "clinical_direction": "in_range_is_best",
        "reference_range": (3.5, 5.0),  # mEq/L
        "loinc": "2823-3",
        "match": re.compile(r"\bpotassium\b|\bk\+?\b", re.IGNORECASE),
    },
    "glucose": {
        "clinical_direction": "in_range_is_best",
        "reference_range": (70.0, 180.0),  # mg/dL -- accept post-prandial up to 180
        "loinc": "2345-7",
        "match": re.compile(r"\bglucose\b|\bblood\s+sugar\b|fingerstick", re.IGNORECASE),
    },
    "inr": {
        "clinical_direction": "in_range_is_best",
        "reference_range": (2.0, 3.0),  # standard warfarin target range
        "loinc": "6301-6",
        "match": re.compile(r"\binr\b|prothrombin\s+time\s+ratio", re.IGNORECASE),
    },
    "oxygen_saturation": {
        "clinical_direction": "higher_is_better",
        "reference_range": (94.0, 100.0),  # SpO2 % -- adult resting
        "loinc": "2708-6",
        "match": re.compile(r"\bspo2\b|oxygen\s+saturation|o2\s+sat", re.IGNORECASE),
    },
}


# Slope thresholds (relative to the absolute mean value of the series) below
# which a trend is considered "flat". Higher noise -> larger threshold needed.
_FLAT_RELATIVE_SLOPE_THRESHOLD = 0.02   # 2 % per day in either direction
_FLAT_ABSOLUTE_SLOPE_THRESHOLD = 1e-6   # for series with mean ≈ 0


# ─────────────────────── Tool implementation ───────────────────────

async def compute_lab_trend_analysis(
    observations: list[dict[str, Any]] | None = None,
    patient_id: str | None = None,
    min_observations_per_lab: int = 2,
) -> LabTrendReport:
    """Per-lab trend analysis over a list of recent Observation records.

    Args:
        observations: optional explicit list (FHIR Observation-shaped dicts
            or simpler {name, value, unit, observed_at} dicts). When omitted,
            the tool fetches the patient bundle from FHIR.
        patient_id: FHIR Patient ID. Used only when `observations` is None.
        min_observations_per_lab: a lab needs ≥ this many points to fit a
            slope; below the threshold the trend is `insufficient_data`.

    Returns:
        LabTrendReport with one LabTrend per identified lab.
    """
    pid: str | None = None
    if observations is None:
        pid = await resolve_patient_id(patient_id)
        bundle = await fetch_patient_bundle(pid)
        observations = _extract_observations_from_bundle(bundle)

    grouped = _group_by_lab(observations)
    trends: list[LabTrend] = []
    n_total = sum(len(v) for v in grouped.values())

    for lab_name, points in grouped.items():
        trends.append(_analyze_lab_series(lab_name, points,
                                            min_observations=min_observations_per_lab))

    flagged = sum(1 for t in trends if t.trend_clinical in ("declining", "borderline"))
    summary = _build_summary(trends, flagged)

    return LabTrendReport(
        patient_id=pid or patient_id,
        n_labs_analyzed=len(trends),
        n_observations_total=n_total,
        trends=trends,
        flagged_labs_count=flagged,
        summary=summary,
    )


# ─────────────────────── Bundle -> observation list ───────────────────────

def _extract_observations_from_bundle(bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(bundle, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in bundle.get("entry") or []:
        res = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(res, dict) or res.get("resourceType") != "Observation":
            continue
        # Pull primary numeric value if present
        vq = res.get("valueQuantity") or {}
        value = vq.get("value") if isinstance(vq, dict) else None
        if value is None:
            continue
        unit = vq.get("unit") if isinstance(vq, dict) else None

        # Lab name from code.text or first coding.display
        code = res.get("code") or {}
        name = ""
        loinc = None
        if isinstance(code, dict):
            name = str(code.get("text") or "").strip()
            for c in code.get("coding") or []:
                if isinstance(c, dict):
                    if not name and c.get("display"):
                        name = str(c["display"]).strip()
                    sys = (c.get("system") or "").lower()
                    if "loinc" in sys and c.get("code"):
                        loinc = str(c["code"])
        observed_at = res.get("effectiveDateTime") or res.get("issued")

        out.append({
            "name": name,
            "value": float(value),
            "unit": unit,
            "observed_at": observed_at,
            "loinc": loinc,
        })
    return out


# ─────────────────────── Grouping ───────────────────────

def _group_by_lab(
    observations: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Match each observation to a known lab via name regex or LOINC. Skip
    observations that don't map to any known lab."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        name = str(obs.get("name", "")).strip()
        loinc = obs.get("loinc")
        matched = _match_lab(name, loinc)
        if matched is None:
            continue
        try:
            value = float(obs.get("value"))
        except (TypeError, ValueError):
            continue
        ts = _parse_observed_at(obs.get("observed_at"))
        if ts is None:
            continue
        grouped.setdefault(matched, []).append({
            "value": value, "observed_at": ts, "unit": obs.get("unit"),
        })
    # Sort each group by timestamp
    for k in grouped:
        grouped[k].sort(key=lambda p: p["observed_at"])
    return grouped


def _match_lab(name: str, loinc: str | None) -> str | None:
    """Return the canonical lab key for an observation."""
    if loinc:
        for key, meta in _LAB_DIRECTORY.items():
            if meta.get("loinc") == loinc:
                return key
    if not name:
        return None
    for key, meta in _LAB_DIRECTORY.items():
        if meta["match"].search(name):
            return key
    return None


def _parse_observed_at(s: Any) -> datetime | None:
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ─────────────────────── Trend analysis ───────────────────────

def _analyze_lab_series(
    lab_name: str,
    points: list[dict[str, Any]],
    min_observations: int,
) -> LabTrend:
    """Fit slope, classify direction + clinical trend for one lab."""
    meta = _LAB_DIRECTORY[lab_name]
    n = len(points)
    if n < min_observations:
        return LabTrend(
            lab_name=lab_name, loinc_code=meta.get("loinc"),
            n_observations=n,
            direction="insufficient_data",
            trend_clinical="insufficient_data",
            rate_per_day=None,
            latest_value=(points[-1]["value"] if points else None),
            reference_range=meta.get("reference_range"),
            in_normal_range=_in_range(points[-1]["value"], meta["reference_range"]) if points else None,
            rationale=f"{n} observation(s); need ≥ {min_observations} to fit a trend.",
        )

    # Linear regression slope (value vs day-offset from earliest)
    earliest = points[0]["observed_at"]
    xs = [(p["observed_at"] - earliest).total_seconds() / 86400.0 for p in points]
    ys = [p["value"] for p in points]
    slope = _linear_slope(xs, ys)
    mean_y = statistics.fmean(ys)
    latest = ys[-1]

    # Direction (up / down / flat) based on slope vs noise threshold
    direction = _classify_direction(slope, mean_y)

    # Clinical interpretation per lab
    cl_dir = meta["clinical_direction"]
    rng = meta["reference_range"]
    in_range = _in_range(latest, rng)
    trend_clinical = _classify_clinical(direction, latest, slope, cl_dir, rng)

    rationale = _build_rationale(lab_name, n, slope, latest, direction, trend_clinical, in_range, rng)

    return LabTrend(
        lab_name=lab_name, loinc_code=meta.get("loinc"),
        n_observations=n,
        direction=direction,  # type: ignore[arg-type]
        trend_clinical=trend_clinical,  # type: ignore[arg-type]
        rate_per_day=round(slope, 4),
        latest_value=round(latest, 4),
        reference_range=rng,
        in_normal_range=in_range,
        rationale=rationale,
    )


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary-least-squares slope of ys ~ xs. Returns 0 when xs is constant."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


def _classify_direction(slope: float, mean_value: float) -> str:
    """Map slope sign + magnitude to a discrete up/down/flat label.

    Threshold is relative to mean (default 2 %/day). For series with mean
    very close to 0, falls back to an absolute threshold so we don't
    classify pure noise as a trend.
    """
    if abs(mean_value) > 1e-9:
        rel = abs(slope) / abs(mean_value)
        if rel < _FLAT_RELATIVE_SLOPE_THRESHOLD:
            return "flat"
    elif abs(slope) < _FLAT_ABSOLUTE_SLOPE_THRESHOLD:
        return "flat"
    return "up" if slope > 0 else "down"


def _classify_clinical(
    direction: str,
    latest: float,
    slope: float,
    clinical_direction: str,
    reference_range: tuple[float, float] | None,
) -> str:
    """Translate (direction, latest, lab semantics) -> clinical trend label."""
    if direction == "flat":
        if reference_range and not _in_range(latest, reference_range):
            return "borderline"  # flat OUTSIDE the range = sustained problem
        return "flat"

    if clinical_direction == "lower_is_better":
        return "improving" if direction == "down" else "declining"
    if clinical_direction == "higher_is_better":
        return "improving" if direction == "up" else "declining"
    if clinical_direction == "in_range_is_best":
        if reference_range is None:
            return "flat"
        lo, hi = reference_range
        midpoint = (lo + hi) / 2.0
        # Improving = moving toward midpoint
        moving_up = direction == "up"
        if latest < lo:
            return "improving" if moving_up else "declining"
        if latest > hi:
            return "improving" if not moving_up else "declining"
        # Latest is in range
        in_target = _in_range(latest, reference_range)
        return "flat" if in_target else "borderline"

    return "flat"


def _in_range(value: float | None, rng: tuple[float, float] | None) -> bool | None:
    if value is None or rng is None:
        return None
    lo, hi = rng
    return lo <= value <= hi


def _build_rationale(
    lab_name: str, n: int, slope: float, latest: float,
    direction: str, trend_clinical: str,
    in_range: bool | None, rng: tuple[float, float] | None,
) -> str:
    parts = [
        f"{n} observations; latest value {latest:.3f}",
    ]
    if rng is not None:
        parts[-1] += f" (ref {rng[0]:.2f}-{rng[1]:.2f})"
    parts.append(f"slope {slope:+.4f}/day")
    parts.append(f"direction={direction}, clinical_trend={trend_clinical}")
    if in_range is False:
        parts.append("⚠ value outside reference range")
    return "; ".join(parts) + "."


def _build_summary(trends: list[LabTrend], flagged: int) -> str:
    if not trends:
        return "No observations matched the curated lab directory; analysis empty."
    by_clinical = {"improving": 0, "declining": 0, "flat": 0,
                    "borderline": 0, "insufficient_data": 0}
    for t in trends:
        by_clinical[t.trend_clinical] = by_clinical.get(t.trend_clinical, 0) + 1
    parts = [f"{len(trends)} lab(s) analyzed:"]
    parts.append(", ".join(f"{n} {k}" for k, n in by_clinical.items() if n > 0))
    if flagged:
        parts.append(f"{flagged} flagged for clinical review (declining or borderline).")
    return " ".join(parts)


# ─────────────────────── MCP registration ───────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_lab_trend_analysis)
