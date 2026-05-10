"""SCALE-1 -- Evaluate the calibrated readmission model against a Synthea cohort.

Synthea (Walonoski JAMIA 2018) generates synthetic FHIR R4 bundles whose
encounter histories include realistic admission patterns. We:

  1. Walk each Patient bundle, extract LACE features via the same helpers
     `compute_readmission_risk` uses at runtime.
  2. Look up the calibrated posterior in `data/coefficients.json`.
  3. Derive a 30-day readmission label from the encounter sequence: was
     there an inpatient/emergency Encounter within 30 days after the index
     inpatient discharge?
  4. Aggregate predictions + outcomes -> ECE / AUROC / Brier + per-bucket
     calibration table.

The runner script (`scripts/synthea_cohort_runner.py`) handles download +
generation; this module is the evaluation core, kept dependency-light so
the test suite can exercise it without invoking Java / Synthea.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ─────────────────────── Encounter parsing ───────────────────────

_INPATIENT_TYPES = {"IMP", "EMER", "ACUTE", "inpatient", "emergency"}


def _is_inpatient(enc: dict[str, Any]) -> bool:
    """Heuristic -- Synthea uses Encounter.class.code 'IMP'/'EMER'."""
    cls = enc.get("class") or {}
    if isinstance(cls, dict):
        code = (cls.get("code") or "").upper()
        if code in _INPATIENT_TYPES:
            return True
    for t in enc.get("type", []) or []:
        if isinstance(t, dict):
            for c in t.get("coding") or []:
                code = (c.get("code") or "").lower()
                if code in {"183452005", "32485007"}:  # SNOMED inpatient codes
                    return True
                if code in _INPATIENT_TYPES:
                    return True
    return False


def _enc_period_dates(enc: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    period = enc.get("period") or {}
    start = period.get("start")
    end = period.get("end")
    s = _parse_iso(start) if start else None
    e = _parse_iso(end) if end else None
    return s, e


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        # Trailing Z handling
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _extract_encounters(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in bundle.get("entry", []) or []:
        res = entry.get("resource") or {}
        if (res.get("resourceType") or "").lower() == "encounter":
            out.append(res)
    return out


# ─────────────────────── Outcome label derivation ───────────────────────

def derive_readmission_label(
    bundle: dict[str, Any], horizon_days: int = 30,
) -> int | None:
    """CMS-style 30-day readmission label.

    Returns 1 if any inpatient/emergency encounter starts within
    `horizon_days` of an EARLIER inpatient encounter's discharge.
    Returns 0 if at least one inpatient encounter exists with no such
    follow-on. Returns None if no inpatient encounter is found (no
    label possible).
    """
    encs = _extract_encounters(bundle)
    inp_encs: list[tuple[datetime, datetime]] = []
    for e in encs:
        if not _is_inpatient(e):
            continue
        s, end = _enc_period_dates(e)
        if s is None or end is None:
            continue
        inp_encs.append((s, end))
    if not inp_encs:
        return None
    inp_encs.sort(key=lambda t: t[0])

    horizon = _delta_days(horizon_days)
    for i, (s_i, e_i) in enumerate(inp_encs):
        for s_j, _ in inp_encs[i + 1:]:
            if s_j > e_i and (s_j - e_i) <= horizon:
                return 1
    return 0


def _delta_days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


# ─────────────────────── Calibration metrics ───────────────────────

def _ece(predictions: list[float], outcomes: list[int],
          n_bins: int = 10) -> float:
    if not predictions:
        return 0.0
    pred = np.asarray(predictions, dtype=float)
    obs = np.asarray(outcomes, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        gap = abs(pred[mask].mean() - obs[mask].mean())
        ece += (n_in / len(pred)) * gap
    return float(ece)


def _auroc(predictions: list[float], outcomes: list[int]) -> float | None:
    if not predictions:
        return None
    obs = np.asarray(outcomes, dtype=int)
    if obs.sum() == 0 or obs.sum() == len(obs):
        return None
    pred = np.asarray(predictions, dtype=float)
    order = np.argsort(-pred)
    o = obs[order]
    n_pos = int(o.sum()); n_neg = len(o) - n_pos
    tps = np.cumsum(o); fps = np.cumsum(1 - o)
    tpr = tps / n_pos; fpr = fps / n_neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapezoid(tpr, fpr))


def _brier(predictions: list[float], outcomes: list[int]) -> float:
    if not predictions:
        return 0.0
    pred = np.asarray(predictions, dtype=float)
    obs = np.asarray(outcomes, dtype=int)
    return float(np.mean((pred - obs) ** 2))


# ─────────────────────── Cohort evaluation ───────────────────────

def _load_coefficients(path: Path | None = None) -> dict[str, Any]:
    coef_path = path or Path("data/coefficients.json")
    if not coef_path.exists():
        raise FileNotFoundError(f"coefficients.json not found at {coef_path}")
    return json.loads(coef_path.read_text(encoding="utf-8"))


def _predict_from_lace(coef: dict[str, Any], lace: int) -> float | None:
    lookup = (coef.get("runtime_coefficients") or {}).get("lookup_table", {})
    entry = lookup.get(str(lace))
    if entry is None:
        return None
    return float(entry["prob_mean"])


def evaluate_synthea_cohort(
    bundles: Iterable[dict[str, Any]],
    *,
    cohort_label: str = "synthea_unknown_n",
    coefficients_path: Path | None = None,
    horizon_days: int = 30,
):
    """Evaluate a cohort of Synthea FHIR bundles and return a SyntheaCohortReport."""
    from mcp_server.tools.readmission_risk import _compute_lace_components
    from shared.schemas import SyntheaCohortReport

    coef = _load_coefficients(coefficients_path)
    coef_version = str(coef.get("model_version", "unknown"))

    predictions: list[float] = []
    outcomes: list[int] = []
    n_patients = 0
    n_with_label = 0

    for bundle in bundles:
        n_patients += 1
        label = derive_readmission_label(bundle, horizon_days=horizon_days)
        if label is None:
            continue
        try:
            comps = _compute_lace_components(bundle)
        except Exception:
            continue
        lace = max(0, min(19, sum(comps[k] for k in ("l", "a", "c", "e"))))
        prob = _predict_from_lace(coef, lace)
        if prob is None:
            continue
        n_with_label += 1
        predictions.append(prob)
        outcomes.append(int(label))

    n_readmitted = int(sum(outcomes))
    base_rate = (n_readmitted / n_with_label) if n_with_label > 0 else 0.0

    ece_v = _ece(predictions, outcomes) if predictions else None
    auroc_v = _auroc(predictions, outcomes) if predictions else None
    brier_v = _brier(predictions, outcomes) if predictions else None

    # Bucketed calibration table
    bucket_table: dict[str, dict[str, float]] = {}
    if predictions:
        n_bins = 5
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        pred_arr = np.asarray(predictions); obs_arr = np.asarray(outcomes)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i == n_bins - 1:
                mask = (pred_arr >= lo) & (pred_arr <= hi)
            else:
                mask = (pred_arr >= lo) & (pred_arr < hi)
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            bucket_table[f"{lo:.2f}-{hi:.2f}"] = {
                "n": float(n_in),
                "predicted_mean": float(pred_arr[mask].mean()),
                "observed_mean": float(obs_arr[mask].mean()),
            }

    rationale = (
        f"Evaluated {n_patients} Synthea patients ({n_with_label} with "
        f"derivable {horizon_days}-day readmission label, "
        f"base rate {base_rate:.3f}). ECE = "
        f"{ece_v:.4f}" if ece_v is not None else "n/a"
    ) + (
        f", AUROC = {auroc_v:.4f}" if auroc_v is not None else ", AUROC = n/a"
    ) + (
        f", Brier = {brier_v:.4f}" if brier_v is not None else ""
    )

    return SyntheaCohortReport(
        cohort_label=cohort_label,
        n_patients=n_patients,
        n_with_readmission_label=n_with_label,
        n_readmitted=n_readmitted,
        base_rate_readmission=base_rate,
        ece=ece_v,
        auroc=auroc_v,
        brier_score=brier_v,
        calibration_buckets=bucket_table,
        coefficients_version=coef_version,
        rationale=rationale,
    )
