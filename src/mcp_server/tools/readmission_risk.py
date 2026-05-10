"""healthcare.compute_readmission_risk -- MCP tool.

Algorithm (per design doc §3.3.1):
  1. Resolve patient_id from explicit arg or X-Patient-ID header.
  2. Fetch Patient + Encounter + Condition + MedicationRequest + Observation
     from the FHIR server via the SHARP context.
  3. Compute LACE features: L (length of stay), A (acuity), C (Charlson),
     E (ED visits 6mo).
  4. Lookup posterior Beta(α_post, β_post) in coefficients.json by lace_total.
  5. Return RiskEstimate with CI + contributing_factors + fhir_observations_used.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.schemas import (
    SUPPORTED_MODEL_NAMES,
    Factor,
    RiskEstimate,
)
from shared.validity import compute_valid_until, compute_validity_window_minutes

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


# ─────────────────────────────────────────────────────────────────────
# Load coefficients.json at module import (startup)
# ─────────────────────────────────────────────────────────────────────

_COEFFICIENTS: dict[str, Any] | None = None
_CONFORMAL: dict[str, Any] | None = None
_CONFORMAL_LOAD_ATTEMPTED: bool = False


def _load_coefficients() -> dict[str, Any]:
    global _COEFFICIENTS
    if _COEFFICIENTS is not None:
        return _COEFFICIENTS
    path = os.environ.get(
        "TRUSTEDRISK_COEFFICIENTS_PATH", "data/coefficients.json"
    )
    fp = Path(path)
    if not fp.exists():
        raise RuntimeError(
            f"coefficients.json not found at {path}. "
            f"Run `make promote-artifacts` to stage the W1 output."
        )
    with fp.open(encoding="utf-8") as f:
        data = json.load(f)

    # Version validation
    model_name = data.get("model_name")
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise RuntimeError(
            f"coefficients.json model_name={model_name!r} not in "
            f"SUPPORTED_MODEL_NAMES={SUPPORTED_MODEL_NAMES}"
        )
    _COEFFICIENTS = data
    return data


def _load_conformal() -> dict[str, Any] | None:
    """Lazy-load `data/conformal_readmission.json` if present. Returns
    None when the artefact is missing -- the runtime gracefully falls
    back to CI95-only output. Cached after the first attempt."""
    global _CONFORMAL, _CONFORMAL_LOAD_ATTEMPTED
    if _CONFORMAL_LOAD_ATTEMPTED:
        return _CONFORMAL
    _CONFORMAL_LOAD_ATTEMPTED = True
    path = os.environ.get(
        "TRUSTEDRISK_CONFORMAL_PATH", "data/conformal_readmission.json"
    )
    fp = Path(path)
    if not fp.exists():
        _CONFORMAL = None
        return None
    try:
        with fp.open(encoding="utf-8") as f:
            _CONFORMAL = json.load(f)
    except (json.JSONDecodeError, OSError):
        _CONFORMAL = None
    return _CONFORMAL


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_readmission_risk(
    horizon_days: int = 30,
    patient_id: str | None = None,
    outcome: str = "readmission_30d",
) -> RiskEstimate:
    """Calibrated probability of a clinical outcome within horizon_days.

    Originally introduced for 30-day readmission. The `outcome` parameter
    (default `readmission_30d` for backward compat) lets the same calibration
    backbone (LACE features + Beta-Binomial posterior) drive other outcomes
    that share the LACE feature space (e.g. mortality_30d). Outcomes whose
    feature space differs (sepsis_24h, deterioration_24h) have dedicated
    tools -- see compute_clinical_deterioration_score.

    Args:
        horizon_days: prediction window in days (default 30).
        patient_id: FHIR Patient resource ID. If None, uses X-Patient-ID header.
        outcome: which calibrated outcome to compute. Currently the runtime
            ships only `readmission_30d` coefficients; the field is recorded
            on the RiskEstimate so downstream consumers know which calibration
            applied. Future extensions (mortality_30d, etc.) require a new
            coefficient bundle.

    Returns:
        RiskEstimate with probability_mean + CI + contributing factors,
        tagged with `outcome_id`.

    Raises:
        ValueError: if patient_id not resolvable.
        RuntimeError: if coefficients.json missing or version-incompatible.
    """
    pid = await resolve_patient_id(patient_id)
    coef = _load_coefficients()

    bundle = await fetch_patient_bundle(pid)

    # Extract LACE features + observation IDs used
    lace_components = _compute_lace_components(bundle)
    lace_total = sum(lace_components[k] for k in ("l", "a", "c", "e"))
    lace_total = max(0, min(19, lace_total))

    obs_ids_used = _extract_observation_ids(bundle)

    # Lookup posterior from coefficients.json
    lookup = (coef.get("runtime_coefficients") or {}).get("lookup_table", {})
    entry = lookup.get(str(lace_total))
    if entry is None:
        raise RuntimeError(
            f"coefficients.json lookup_table missing entry for LACE={lace_total}"
        )

    prob_mean = float(entry["prob_mean"])
    ci_low, ci_high = (float(v) for v in entry["prob_ci95"])
    ci_width = float(entry.get("ci_width", ci_high - ci_low))

    # OOD / calibration plateau guard. The shipped coefficients.json was
    # trained on a population whose LACE distribution thinned out above
    # the moderate-risk band; LACE bins 12 through 19 collapse to the
    # same prob_mean (~0.282) instead of monotonically rising. Returning
    # that constant for a high-risk patient understates the true risk
    # and would let a downstream caller present an under-estimated
    # readmission probability as if it were calibrated.
    #
    # When LACE >= LACE_PLATEAU_FLOOR we mark the estimate as
    # abstain_recommended=True with confidence='degraded', keep the
    # numeric fields populated for transparency (the caller can inspect
    # what the underpowered bucket would have said), and require the
    # caller to either re-evaluate manually or seek a model trained on
    # a wider LACE distribution. The threshold is read from the
    # coefficients artefact when present, defaulting to 12.
    LACE_PLATEAU_FLOOR = int(
        coef.get("calibration_metadata", {}).get(
            "lace_plateau_floor", 12,
        )
    )
    abstain_recommended = lace_total >= LACE_PLATEAU_FLOOR
    abstain_reason: str | None = None
    confidence_value = str(coef.get("confidence", "preferred"))
    if abstain_recommended:
        abstain_reason = (
            f"lace_calibration_plateau_OOD: this patient's LACE "
            f"score ({lace_total}) falls in the high-risk band "
            f"({LACE_PLATEAU_FLOOR}-19) where the coefficients.json "
            "lookup table degenerates to a constant "
            f"prob_mean={prob_mean:.4f}, because the calibration "
            "training set was underpowered above the moderate-risk "
            "tier. The probability shown is the bucket mean of an "
            "underpowered population, NOT a calibrated estimate for "
            "this patient. The true 30-day readmission risk is most "
            "likely higher (van Walraven 2010 Table 3 reports >40% "
            "for LACE 16+). Do NOT use this number for clinical "
            "decision-making. Re-evaluate manually with a clinician, "
            "or wait for a re-calibrated coefficients release."
        )
        confidence_value = "degraded"

    contributing = [
        Factor(
            name="LACE_length_of_stay",
            raw_value=float(lace_components["l_raw"]),
            lace_points=int(lace_components["l"]),
            weight=_weight_for_component(lace_components["l"], lace_total),
        ),
        Factor(
            name="LACE_acuity",
            raw_value=float(lace_components["a_raw"]),
            lace_points=int(lace_components["a"]),
            weight=_weight_for_component(lace_components["a"], lace_total),
        ),
        Factor(
            name="LACE_comorbidity",
            raw_value=float(lace_components["c_raw"]),
            lace_points=int(lace_components["c"]),
            weight=_weight_for_component(lace_components["c"], lace_total),
        ),
        Factor(
            name="LACE_ed_visits_6mo",
            raw_value=float(lace_components["e_raw"]),
            lace_points=int(lace_components["e"]),
            weight=_weight_for_component(lace_components["e"], lace_total),
        ),
    ]

    computed_at = datetime.now(timezone.utc)
    valid_for_minutes = compute_validity_window_minutes(lace_total)
    valid_until = compute_valid_until(computed_at, lace_total)

    # Validate outcome against the schema's enum; fall back to default if
    # an unknown string is passed (rather than crashing -- clinicians might
    # type a free-form outcome name).
    from shared.schemas import OutcomeId  # type: ignore[import-untyped]
    valid_outcomes = set(OutcomeId.__args__)  # type: ignore[attr-defined]
    if outcome not in valid_outcomes:
        outcome = "readmission_30d"

    # Phase 11.1 -- attach conformal interval + prediction set when the
    # calibration artefact is available. Pure additive: when missing
    # the four fields stay None and downstream consumers use CI95.
    conf_lower: float | None = None
    conf_upper: float | None = None
    conf_set: list[int] | None = None
    conf_target: float | None = None
    conf = _load_conformal()
    if conf is not None:
        conf_target = float(conf.get("target_coverage", 0.90))
        q_abs = float(
            (conf.get("abs_residual") or {}).get("quantile_threshold", 0.0)
        )
        if q_abs > 0:
            conf_lower = max(0.0, prob_mean - q_abs)
            conf_upper = min(1.0, prob_mean + q_abs)
        q_bin = float(
            (conf.get("binary_one_minus_p") or {}).get("quantile_threshold",
                                                          0.0)
        )
        if q_bin > 0:
            pred_set: list[int] = []
            if (1.0 - prob_mean) <= q_bin:
                pred_set.append(1)
            if prob_mean <= q_bin:
                pred_set.append(0)
            conf_set = pred_set

    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version=str(coef.get("model_version", "unknown")),
        outcome_id=outcome,  # type: ignore[arg-type]
        horizon_days=horizon_days,
        lace_raw_score=lace_total,
        probability_mean=prob_mean,
        probability_ci95=(ci_low, ci_high),
        probability_ci_width=ci_width,
        contributing_factors=contributing,
        fhir_observations_used=obs_ids_used[:20],  # cap for token efficiency
        computed_at=computed_at,
        confidence=confidence_value,  # type: ignore[arg-type]
        valid_for_minutes=valid_for_minutes,
        valid_until=valid_until,
        conformal_interval_lower=(round(conf_lower, 4)
                                          if conf_lower is not None else None),
        conformal_interval_upper=(round(conf_upper, 4)
                                          if conf_upper is not None else None),
        conformal_prediction_set=conf_set,                   # type: ignore[arg-type]
        conformal_target_coverage=conf_target,
        abstain_recommended=abstain_recommended,
        abstain_reason=abstain_reason,
    )


def _compute_lace_components(bundle: dict[str, Any]) -> dict[str, int]:
    """Extract L/A/C/E LACE components from a FHIR Bundle.

    Returns a dict with:
      l, a, c, e: int points per van Walraven 2010 Table 3.
      l_raw, a_raw, c_raw, e_raw: raw feature values for the Factor.raw_value field.
    """
    entries = bundle.get("entry") or []
    encounters = [e.get("resource", {}) for e in entries if _rtype(e) == "Encounter"]
    conditions = [e.get("resource", {}) for e in entries if _rtype(e) == "Condition"]

    # Length of stay from most recent hospitalization (class=IMP)
    los_days = 0.0
    for enc in encounters:
        cls = enc.get("class") or {}
        if isinstance(cls, dict) and cls.get("code") == "IMP":
            period = enc.get("period") or {}
            start = period.get("start", "")
            end = period.get("end", "")
            if start and end:
                try:
                    ds = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    de = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    delta = (de - ds).total_seconds() / 86400.0
                    if delta > los_days:
                        los_days = delta
                except (ValueError, TypeError):
                    continue

    l_points = _score_los(los_days)

    # Acuity: 3 points if any hospitalization is marked acute (class=IMP = acute inpatient)
    a_raw = int(any((e.get("class") or {}).get("code") == "IMP" for e in encounters))
    a_points = 3 if a_raw else 0

    # Charlson comorbidity index: simplified as min(len(conditions), 6)
    c_raw = len(conditions)
    c_points = _score_charlson(c_raw)

    # ED visits in last 6 months: count non-IMP encounters as proxy
    e_raw = sum(
        1 for e in encounters
        if (e.get("class") or {}).get("code") != "IMP"
    )
    e_points = min(4, e_raw)

    return {
        "l": l_points, "l_raw": int(los_days),
        "a": a_points, "a_raw": a_raw,
        "c": c_points, "c_raw": c_raw,
        "e": e_points, "e_raw": e_raw,
    }


def _score_los(days: float) -> int:
    """van Walraven 2010 LACE-L: <1->0, 1->1, 2->2, 3->3, 4-6->4, 7-13->5, ≥14->7."""
    if days < 1:
        return 0
    if days < 2:
        return 1
    if days < 3:
        return 2
    if days < 4:
        return 3
    if days < 7:
        return 4
    if days < 14:
        return 5
    return 7


def _score_charlson(n: int) -> int:
    """LACE-C: 0->0, 1->1, 2->2, 3->3, ≥4->5."""
    if n <= 0:
        return 0
    if n >= 4:
        return 5
    return int(n)


def _weight_for_component(points: int, total: int) -> float:
    """Contribution weight of a LACE component to the total score."""
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, points / total))


def _rtype(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    res = entry.get("resource") if isinstance(entry, dict) else None
    if not isinstance(res, dict):
        return ""
    return str(res.get("resourceType", ""))


def _extract_observation_ids(bundle: dict[str, Any]) -> list[str]:
    entries = bundle.get("entry") or []
    return [
        str((e.get("resource") or {}).get("id", ""))
        for e in entries
        if _rtype(e) == "Observation"
    ]


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    """Register this tool on the shared FastMCP instance."""
    mcp.tool()(compute_readmission_risk)
