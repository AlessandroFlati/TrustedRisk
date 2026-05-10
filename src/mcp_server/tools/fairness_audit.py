"""healthcare.compute_fairness_audit -- surface residual subgroup bias.

The CMS HRRP risk-adjustment formula explicitly excludes socioeconomic status,
race, and insurance type -- a methodological choice that has been criticized
(JAMA 2018, Wadhera et al.) for disadvantaging safety-net hospitals serving
complex populations. Our calibrated `coefficients.json` inherits the same
omission: it's calibrated on a single Synthea-dominant cohort with no
subgroup stratification.

This tool is the runtime's response: rather than pretend the model is
fair, surface the **expected residual drift** by subgroup against published
baselines, and recommend a confidence action ranging from "no_action" to
"abstain_recommended". The DecisionCard renders a Fairness Watch block
that the clinician sees alongside the risk estimate.

Baselines are conservative literature-derived multipliers:
  - AHRQ HCUP Statistical Brief #278 (2022) -- readmission by payer / race
  - JAMA 2018 (Wadhera) -- HRRP impact by hospital safety-net status
  - JGIM 2014 (Joynt & Jha) -- sociodemographic readmission disparities

The point is not to claim the multipliers are correct in any one institution.
It's to make the bias **declared** rather than hidden.
"""

from __future__ import annotations

import json
import os
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

from shared.schemas import (
    FairnessReport,
    RiskEstimate,
    SubgroupCalibration,
)


# ─────────────────────────────────────────────────────────────────────
# Optional calibrated baseline (W5 output) overrides literature defaults
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_calibrated_baseline() -> dict[str, Any] | None:
    """Read fairness_baseline.json if staged. None means runtime falls back to
    literature defaults defined below."""
    path = os.environ.get("TRUSTEDRISK_FAIRNESS_BASELINE_PATH",
                          "data/fairness_baseline.json")
    fp = Path(path)
    if not fp.exists():
        return None
    try:
        with fp.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Baseline subgroup multipliers (literature-derived, conservative)
# ─────────────────────────────────────────────────────────────────────
#
# Each entry: subgroup -> multiplier-vs-cohort-baseline + citation.
# Multipliers are unitless ratios applied to the predicted rate. A multiplier
# of 1.18 means "this subgroup historically readmits at 1.18× the cohort
# baseline at the same risk score". Drift fires when our prediction is more
# than 20% off the baseline-adjusted expectation.

_RACE_BASELINES: dict[str, dict[str, Any]] = {
    "black": {
        "multiplier": 1.18,
        "citation": "AHRQ HCUP Statistical Brief #278 (2022) -- Black/AA patients "
                    "show 1.15-1.22× readmission rate vs White at matched LACE.",
    },
    "black_african_american": {
        "multiplier": 1.18,
        "citation": "AHRQ HCUP Statistical Brief #278 (2022).",
    },
    "hispanic": {
        "multiplier": 1.10,
        "citation": "Joynt & Jha 2014 (JGIM): Hispanic patients ~1.08-1.12× "
                    "readmission rate at matched LACE in Medicare data.",
    },
    "asian": {
        "multiplier": 0.95,
        "citation": "AHRQ HCUP 2022: Asian patients show modestly LOWER "
                    "readmission rates (~0.90-0.96×) on Medicare cohorts.",
    },
    "white": {
        "multiplier": 1.00,
        "citation": "Reference category in HRRP risk-adjustment.",
    },
    "unknown": {
        "multiplier": 1.10,
        "citation": "AHRQ HCUP 2022: race-unknown patients trend slightly "
                    "higher (~1.05-1.15×) due to documentation gaps correlating "
                    "with care-discontinuity.",
    },
}

# Age-based multipliers (relative to 18-64 baseline)
_AGE_BAND_BASELINES: list[tuple[int, int, float, str]] = [
    (18, 64, 1.00, "Reference adult cohort"),
    (65, 74, 1.20, "AHRQ HCUP 2022: 65-74 age band ~1.18-1.22× baseline"),
    (75, 84, 1.35, "AHRQ HCUP 2022: 75-84 age band ~1.30-1.40× baseline"),
    (85, 120, 1.50, "AHRQ HCUP 2022: 85+ age band ~1.45-1.55× baseline; frailty effect"),
]

# Insurance multipliers (relative to private/commercial)
_INSURANCE_BASELINES: dict[str, dict[str, Any]] = {
    "medicare": {
        "multiplier": 1.10,
        "citation": "HRRP Medicare cohort baseline (used in CMS adjustment).",
    },
    "medicaid": {
        "multiplier": 1.18,
        "citation": "Wadhera 2018 (JAMA): Medicaid beneficiaries ~1.15-1.22× "
                    "readmission rate at matched diagnosis. Safety-net effect.",
    },
    "uninsured": {
        "multiplier": 1.25,
        "citation": "AHRQ Statistical Brief 2022: uninsured patients trend "
                    "~1.20-1.30× due to outpatient access barriers.",
    },
    "private": {
        "multiplier": 1.00,
        "citation": "Reference category.",
    },
    "commercial": {
        "multiplier": 1.00,
        "citation": "Reference category.",
    },
}

# Drift thresholds (absolute relative drift to severity)
_DRIFT_TO_SEVERITY: list[tuple[float, str]] = [
    (0.30, "high"),     # > 30% drift
    (0.20, "medium"),   # 20-30%
    (0.10, "low"),      # 10-20%
    # < 10% -> "none"
]


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

_HL7_UNKNOWN_SENTINELS: frozenset[str] = frozenset({"unknown", "u", "unk", ""})


def _is_hl7_unknown(value: Any) -> bool:
    """Return True when value carries the HL7 'unknown' sentinel or is absent.

    Recognises the HL7 V3 null-flavor 'UNK', the two-letter code 'U', and the
    plain string "unknown". Empty string is also treated as unknown because it
    indicates a field that was present in the source message but carried no
    discriminating information.
    """
    if value is None:
        return False  # absent field -- caller decides whether to treat as unknown
    return str(value).strip().lower() in _HL7_UNKNOWN_SENTINELS


async def compute_fairness_audit(
    risk: dict[str, Any] | RiskEstimate,
    patient_demographics: dict[str, Any],
) -> FairnessReport:
    """Audit a RiskEstimate against literature subgroup baselines.

    Args:
        risk: a RiskEstimate object (or its dict form) -- typically the output of
            compute_readmission_risk for the same patient.
        patient_demographics: dict with keys among {age, sex, race, ethnicity,
            insurance_type}. Missing keys reduce the audit's coverage. When a
            key is present but carries the HL7 'unknown' sentinel ('U',
            'unknown', 'unk', or empty string), the tool returns
            abstain_recommended=True rather than substituting a population mean,
            because the sentinel cannot be assumed equivalent to any tracked
            subgroup.

    Returns:
        FairnessReport with per-subgroup calibration drift + confidence action.
        When any critical demographic field carries an HL7 'unknown' sentinel,
        abstain_recommended is True and no subgroup drift is computed for that
        field.
    """
    if isinstance(risk, dict):
        predicted_rate = float(risk.get("probability_mean", 0.0))
    else:
        predicted_rate = float(risk.probability_mean)

    if not (0.0 <= predicted_rate <= 1.0):
        raise ValueError(f"predicted_rate must be in [0,1], got {predicted_rate}")

    # Detect demographics shape. Three cases matter for the abstain
    # contract:
    #
    #  1. Every axis (age, race, insurance) is either absent or an HL7
    #     unknown sentinel. The audit cannot run at all -- abstain.
    #
    #  2. At least one axis carries an explicit unknown sentinel
    #     (HL7 'U' / 'unknown' / 'unk' / ''). The caller deliberately
    #     marked the field as "data absent" -- substituting a baseline
    #     would misreport the coverage. Abstain.
    #
    #  3. At least one axis is populated and the rest are simply missing
    #     keys (None / not in dict). The audit runs on the populated
    #     subset; the rationale enumerates which subgroups were in scope.
    #     This is the common "partial demographic" path.
    def _has_explicit_unknown(raw: Any) -> bool:
        return raw is not None and _is_hl7_unknown(raw)

    def _is_present(raw: Any) -> bool:
        return raw is not None and not _is_hl7_unknown(raw)

    age_raw = patient_demographics.get("age")
    race_raw = patient_demographics.get("race")
    _ins_type_raw = patient_demographics.get("insurance_type")
    _ins_raw = patient_demographics.get("insurance")
    insurance_raw = _ins_type_raw if _ins_type_raw is not None else _ins_raw

    explicit_unknowns: list[str] = []
    if _has_explicit_unknown(age_raw):
        explicit_unknowns.append("age")
    if _has_explicit_unknown(race_raw):
        explicit_unknowns.append("race")
    if _has_explicit_unknown(insurance_raw):
        explicit_unknowns.append("insurance_type")

    any_present = (
        _is_present(age_raw) or _is_present(race_raw)
        or _is_present(insurance_raw)
    )

    # Case 1+2 collapse: abstain when no axis is usable, or when any
    # axis carries an explicit unknown sentinel.
    if not any_present or explicit_unknowns:
        if not any_present:
            reason = (
                "missing_critical_demographic: no demographic axis "
                "(age, race, insurance_type) was supplied. Fairness "
                "audit refuses to substitute population means for the "
                "absent fields."
            )
        else:
            reason = (
                "missing_critical_demographic: "
                + ", ".join(explicit_unknowns)
                + " carry an HL7 'unknown' sentinel. The sentinel "
                "cannot be substituted with a population mean -- the "
                "caller deliberately marked these fields as data-absent."
            )
        return FairnessReport(
            n_subgroups_assessed=0,
            subgroup_drifts=[],
            max_relative_drift=0.0,
            confidence_action="no_action",
            rationale=(
                f"Fairness audit aborted: {reason} "
                "Abstain is the safe disposition."
            ),
            abstain_recommended=True,
            abstain_reason=reason,
        )

    drifts: list[SubgroupCalibration] = []
    calibrated = _load_calibrated_baseline()

    age = patient_demographics.get("age")
    if age is not None:
        try:
            age_int = int(age)
            entry = _age_to_baseline_with_calibration(age_int, calibrated)
            if entry is not None:
                lo, hi, mult, citation = entry
                expected = min(1.0, predicted_rate * mult)
                rel_drift = (predicted_rate - expected) / expected if expected > 0 else 0.0
                drifts.append(SubgroupCalibration(
                    subgroup_name="age_band",
                    subgroup_value=f"{lo}-{hi}",
                    expected_rate_baseline=expected,
                    predicted_rate_for_patient=predicted_rate,
                    relative_drift=rel_drift,
                    severity=_drift_severity(rel_drift),  # type: ignore[arg-type]
                    citation=citation,
                ))
        except (TypeError, ValueError):
            pass

    race = patient_demographics.get("race")
    if race:
        race_norm = str(race).strip().lower().replace(" ", "_").replace("/", "_")
        race_norm = race_norm.replace("__", "_")
        baseline = _race_baseline_with_calibration(race_norm, calibrated)
        if baseline is not None:
            mult = baseline["multiplier"]
            expected = min(1.0, predicted_rate * mult)
            rel_drift = (predicted_rate - expected) / expected if expected > 0 else 0.0
            drifts.append(SubgroupCalibration(
                subgroup_name="race",
                subgroup_value=str(race),
                expected_rate_baseline=expected,
                predicted_rate_for_patient=predicted_rate,
                relative_drift=rel_drift,
                severity=_drift_severity(rel_drift),  # type: ignore[arg-type]
                citation=baseline["citation"],
            ))

    insurance = patient_demographics.get("insurance_type") or patient_demographics.get("insurance")
    if insurance:
        ins_norm = str(insurance).strip().lower()
        baseline = _INSURANCE_BASELINES.get(ins_norm)
        if baseline is not None:
            mult = baseline["multiplier"]
            expected = min(1.0, predicted_rate * mult)
            rel_drift = (predicted_rate - expected) / expected if expected > 0 else 0.0
            drifts.append(SubgroupCalibration(
                subgroup_name="insurance_type",
                subgroup_value=str(insurance),
                expected_rate_baseline=expected,
                predicted_rate_for_patient=predicted_rate,
                relative_drift=rel_drift,
                severity=_drift_severity(rel_drift),  # type: ignore[arg-type]
                citation=baseline["citation"],
            ))

    max_abs_drift = max((abs(d.relative_drift) for d in drifts), default=0.0)
    confidence_action = _grade_confidence_action(drifts)
    rationale = _build_fairness_rationale(drifts, confidence_action)

    return FairnessReport(
        n_subgroups_assessed=len(drifts),
        subgroup_drifts=drifts,
        max_relative_drift=max_abs_drift,
        confidence_action=confidence_action,  # type: ignore[arg-type]
        rationale=rationale,
    )


def _age_to_baseline(age: int) -> tuple[int, int, float, str] | None:
    for lo, hi, mult, citation in _AGE_BAND_BASELINES:
        if lo <= age <= hi:
            return lo, hi, mult, citation
    return None


def _age_to_baseline_with_calibration(
    age: int, calibrated: dict[str, Any] | None,
) -> tuple[int, int, float, str] | None:
    """Prefer the calibrated multiplier when present; fall back to literature."""
    lit = _age_to_baseline(age)
    if calibrated is None or lit is None:
        return lit
    lo, hi, lit_mult, lit_citation = lit
    label = f"{lo}-{hi}" if hi < 120 else "85+"
    by_age = calibrated.get("by_age_band", {}) or {}
    cal_entry = by_age.get(label)
    if not isinstance(cal_entry, dict) or "multiplier" not in cal_entry:
        return lit
    return (lo, hi, float(cal_entry["multiplier"]), str(cal_entry.get("citation", lit_citation)))


def _race_baseline_with_calibration(
    race_norm: str, calibrated: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer calibrated race multiplier; fall back to literature."""
    by_race = (calibrated or {}).get("by_race", {}) or {}
    # Calibrated keys are lowercase exact strings ("white", "black or african american", etc.)
    lit = _RACE_BASELINES.get(race_norm) or _RACE_BASELINES.get(race_norm.split("_")[0])
    # Try exact, then a fuzzy match (replace _ with spaces)
    cal_entry = by_race.get(race_norm) or by_race.get(race_norm.replace("_", " "))
    if isinstance(cal_entry, dict) and "multiplier" in cal_entry:
        return {
            "multiplier": float(cal_entry["multiplier"]),
            "citation": str(cal_entry.get("citation", "Calibrated on receiving cohort.")),
        }
    return lit


def _drift_severity(rel_drift: float) -> str:
    abs_drift = abs(rel_drift)
    for threshold, sev in _DRIFT_TO_SEVERITY:
        if abs_drift >= threshold:
            return sev
    return "none"


def _grade_confidence_action(drifts: list[SubgroupCalibration]) -> str:
    if not drifts:
        return "no_action"
    n_high = sum(1 for d in drifts if d.severity == "high")
    n_medium = sum(1 for d in drifts if d.severity == "medium")
    if n_high >= 2:
        return "abstain_recommended"
    if n_high >= 1:
        return "downgrade_confidence"
    if n_medium >= 2:
        return "downgrade_confidence"
    if n_medium >= 1:
        return "flag_for_review"
    return "no_action"


def _build_fairness_rationale(
    drifts: list[SubgroupCalibration],
    confidence_action: str,
) -> str:
    if not drifts:
        return (
            "No subgroup-relevant demographics provided; fairness audit produced "
            "no findings. The underlying calibration cohort is Synthea-dominant -- "
            "absence of audit is not absence of bias."
        )

    high_drifts = [d for d in drifts if d.severity == "high"]
    medium_drifts = [d for d in drifts if d.severity == "medium"]
    parts = []

    if high_drifts:
        items = ", ".join(f"{d.subgroup_name}={d.subgroup_value} ({d.relative_drift:+.0%})"
                          for d in high_drifts)
        parts.append(f"HIGH-severity drift on: {items}.")
    if medium_drifts:
        items = ", ".join(f"{d.subgroup_name}={d.subgroup_value} ({d.relative_drift:+.0%})"
                          for d in medium_drifts)
        parts.append(f"Medium-severity drift on: {items}.")

    parts.append(f"Confidence action: {confidence_action}.")

    if confidence_action == "abstain_recommended":
        parts.append(
            "Recommend ABSTAIN: multiple high-severity drifts indicate the "
            "patient lies in a subgroup intersection where the calibration "
            "baseline disagrees substantially with literature. A clinician "
            "should make the recommendation, not the model."
        )
    elif confidence_action == "downgrade_confidence":
        parts.append(
            "Recommend downgrading the DecisionCard's confidence flag from "
            "preferred to degraded -- the patient's subgroup baseline differs "
            "materially from the predicted rate."
        )
    elif confidence_action == "flag_for_review":
        parts.append("Flag the case for clinician review; do not auto-act on the recommendation.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_fairness_audit)
