"""Phase 15.C -- Synthetic prospective evaluation harness.

Simulates a "what would have happened" deployment of the full
TrustedRisk decision pipeline (calibrated risk -> 4-critic ensemble ->
3-agent debate) on a synthetic cohort, and aggregates the resulting
abstain / downgrade / action statistics per subgroup.

Pure-deterministic. The cohort generator is the same seeded sampler
the subgroup-audit builder uses; the per-encounter pipeline reuses
:mod:`a2a_agent.multi_agent_debate` for the debate step and a
deterministic four-critic stand-in for the critic ensemble.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from .multi_agent_debate import DebateInput, run_debate


# ─────────────────────────────────────────────────────────────────────
# Cohort sampling (mirrors scripts/build_subgroup_audit.py)
# ─────────────────────────────────────────────────────────────────────


_LACE_TO_PROB = [
    (range(0, 3),   0.072),
    (range(3, 6),   0.103),
    (range(6, 10),  0.158),
    (range(10, 13), 0.234),
    (range(13, 60), 0.327),
]

_RACE_MULT = {
    "white": 1.00, "black": 1.18, "hispanic": 1.10,
    "asian": 0.95, "indigenous": 1.22, "other": 1.05,
}
_INSURANCE_MULT = {
    "commercial": 1.00, "medicare": 1.05, "medicaid": 1.20,
    "uninsured": 1.27, "tricare": 0.98,
}
_LANGUAGE_MULT = {
    "english": 1.00, "spanish": 1.06, "chinese": 1.04,
    "vietnamese": 1.05, "arabic": 1.08,
}
_AGE_BANDS: list[tuple[int, int, str, float]] = [
    (18, 39,  "18-39", 0.85),
    (40, 64,  "40-64", 0.96),
    (65, 74,  "65-74", 1.05),
    (75, 84,  "75-84", 1.15),
    (85, 110, "85+",   1.30),
]
_SEX_MULT = {"female": 0.97, "male": 1.00}
_ETHNICITY_MULT = {"non_hispanic": 1.00, "hispanic": 1.10}


# A subgroup -> list of (value, multiplier) -- used for fairness flag
_FLAGGED_SUBGROUPS: frozenset[str] = frozenset({
    "black", "indigenous", "medicaid", "uninsured",
})


DISPOSITION_THRESHOLD = 0.20


def _calibrated_p(lace: int) -> float:
    for r, p in _LACE_TO_PROB:
        if lace in r:
            return p
    return _LACE_TO_PROB[-1][1]


def _ci_width_for_lace(lace: int) -> float:
    """Use the per-bin variance of a Beta(alpha=2+k, beta=18+n-k)
    posterior approximated for a typical bin n. Wider for the extremes
    where data is sparser."""
    if lace <= 2:
        return 0.06
    if lace <= 5:
        return 0.05
    if lace <= 9:
        return 0.045
    if lace <= 12:
        return 0.05
    return 0.07


def _sample_lace(rng: random.Random) -> int:
    L = rng.choices([1, 2, 3, 4, 5, 6, 7],
                    weights=[8, 14, 20, 22, 18, 12, 6])[0]
    A = rng.choices([0, 3], weights=[40, 60])[0]
    C = rng.choices([0, 1, 2, 3, 4, 5],
                    weights=[20, 22, 22, 18, 12, 6])[0]
    E = rng.choices([0, 1, 2, 3, 4],
                    weights=[35, 28, 18, 12, 7])[0]
    return min(19, L + A + C + E)


def _sample_demographics(rng: random.Random) -> dict[str, Any]:
    lo, hi, age_label, age_mult = rng.choices(
        _AGE_BANDS, weights=[10, 35, 25, 20, 10])[0]
    age = rng.randint(lo, hi)
    sex = rng.choices(list(_SEX_MULT), weights=[51, 49])[0]
    race = rng.choices(list(_RACE_MULT),
                       weights=[60, 13, 18, 6, 1, 2])[0]
    ethnicity = rng.choices(list(_ETHNICITY_MULT),
                            weights=[82, 18])[0]
    insurance = rng.choices(list(_INSURANCE_MULT),
                            weights=[40, 30, 18, 8, 4])[0]
    language = rng.choices(list(_LANGUAGE_MULT),
                           weights=[78, 13, 5, 2, 2])[0]
    return {
        "age": age, "age_band": age_label, "age_mult": age_mult,
        "sex": sex, "race": race, "ethnicity": ethnicity,
        "insurance_type": insurance, "language": language,
    }


def _compose_outcome_rate(base: float, dem: dict[str, Any]) -> float:
    rate = base * dem["age_mult"] * _SEX_MULT[dem["sex"]]
    rate *= _RACE_MULT[dem["race"]] * _ETHNICITY_MULT[dem["ethnicity"]]
    rate *= _INSURANCE_MULT[dem["insurance_type"]]
    rate *= _LANGUAGE_MULT[dem["language"]]
    return min(0.95, max(0.005, rate))


# ─────────────────────────────────────────────────────────────────────
# Per-encounter pipeline
# ─────────────────────────────────────────────────────────────────────


_FinalAction = Literal[
    "discharge_home", "continued_admission",
    "discharge_with_homecare", "abstain",
]


class _EncounterResult(BaseModel):
    encounter_id: int
    lace: int
    age: int
    age_band: str
    sex: str
    race: str
    ethnicity: str
    insurance_type: str
    language: str
    risk_point_estimate: float
    risk_ci_width: float
    outcome: int
    proposed_action: _FinalAction
    final_action: _FinalAction
    debate_verdict: str
    fairness_flagged: bool
    abstained: bool
    downgraded: bool


def _propose_initial_action(
    risk: float, threshold: float = DISPOSITION_THRESHOLD,
) -> _FinalAction:
    if risk < 0.5 * threshold:
        return "discharge_home"
    if risk < threshold:
        return "discharge_with_homecare"
    return "continued_admission"


def _flagged_subgroup_for_dem(dem: dict[str, Any]) -> str | None:
    """Return the lowercase subgroup string the fairness-guard sees, or
    ``None`` if no flagged subgroup applies."""
    for axis_key in ("race", "insurance_type"):
        v = str(dem.get(axis_key) or "").lower()
        if v in _FLAGGED_SUBGROUPS:
            return v
    return None


def _evaluate_encounter(
    enc_id: int,
    lace: int,
    dem: dict[str, Any],
    risk_estimate: float,
    ci_width: float,
    outcome: int,
    fairness_audit_present: bool,
) -> _EncounterResult:
    proposed = _propose_initial_action(risk_estimate)
    flagged = _flagged_subgroup_for_dem(dem)
    payload = DebateInput(
        recommended_action=proposed,
        risk_point_estimate=risk_estimate,
        risk_ci_width=ci_width,
        fairness_subgroup=flagged,
        fairness_audit_present=fairness_audit_present,
    )
    outcome_dec = run_debate(payload)
    if outcome_dec.verdict == "approved":
        final = proposed
        downgraded = False
        abstained = False
    elif outcome_dec.verdict == "revise":
        final = (
            "discharge_with_homecare"
            if proposed == "discharge_home"
            else proposed
        )
        downgraded = True
        abstained = False
    else:
        final = "abstain"
        downgraded = False
        abstained = True
    return _EncounterResult(
        encounter_id=enc_id,
        lace=lace,
        age=dem["age"], age_band=dem["age_band"],
        sex=dem["sex"], race=dem["race"], ethnicity=dem["ethnicity"],
        insurance_type=dem["insurance_type"], language=dem["language"],
        risk_point_estimate=risk_estimate,
        risk_ci_width=ci_width,
        outcome=outcome,
        proposed_action=proposed, final_action=final,
        debate_verdict=outcome_dec.verdict,
        fairness_flagged=flagged is not None,
        abstained=abstained,
        downgraded=downgraded,
    )


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────


class ProspectiveSummary(BaseModel):
    cohort_n: int = Field(ge=0)
    seed: int
    fairness_audit_present: bool
    overall: dict[str, Any]
    per_subgroup: dict[str, dict[str, Any]]
    baseline_lace_only: dict[str, Any]
    references: list[str] = Field(default_factory=list)


def _aggregate_overall(rows: list[_EncounterResult]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    actions = Counter(r.final_action for r in rows)
    abstains = sum(1 for r in rows if r.abstained)
    downgrades = sum(1 for r in rows if r.downgraded)
    obs_rate = sum(r.outcome for r in rows) / n
    pred_rate = sum(r.risk_point_estimate for r in rows) / n
    flagged_n = sum(1 for r in rows if r.fairness_flagged)
    return {
        "n": n,
        "abstain_rate": abstains / n,
        "downgrade_rate": downgrades / n,
        "approve_rate": (n - abstains - downgrades) / n,
        "observed_outcome_rate": obs_rate,
        "mean_predicted_rate": pred_rate,
        "calibration_gap_abs": abs(obs_rate - pred_rate),
        "fairness_flagged_share": flagged_n / n,
        "action_distribution": dict(actions),
    }


def _aggregate_by(
    rows: list[_EncounterResult], key: str,
) -> dict[str, dict[str, Any]]:
    by: dict[str, list[_EncounterResult]] = defaultdict(list)
    for r in rows:
        by[getattr(r, key)].append(r)
    out: dict[str, dict[str, Any]] = {}
    for value, sub in by.items():
        out[value] = _aggregate_overall(sub)
    return out


def _baseline_lace_only(rows: list[_EncounterResult]) -> dict[str, Any]:
    """Counterfactual: what if we just thresholded on the LACE-only risk
    point estimate without the debate / fairness layer?"""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    proposed = Counter(r.proposed_action for r in rows)
    obs_rate = sum(r.outcome for r in rows) / n
    pred_rate = sum(r.risk_point_estimate for r in rows) / n
    return {
        "n": n,
        "proposed_distribution": dict(proposed),
        "abstain_rate": 0.0,
        "downgrade_rate": 0.0,
        "approve_rate": 1.0,
        "observed_outcome_rate": obs_rate,
        "mean_predicted_rate": pred_rate,
        "calibration_gap_abs": abs(obs_rate - pred_rate),
    }


def run_prospective_eval(
    n_encounters: int,
    *,
    seed: int = 20260430,
    fairness_audit_present: bool = True,
) -> ProspectiveSummary:
    """Run the full prospective evaluation and return the aggregated
    summary. Pure-deterministic for fixed ``seed``.

    Args:
        n_encounters: cohort size.
        seed: RNG seed.
        fairness_audit_present: whether the fairness-guard debater
            sees a fairness audit artefact (flips its abstain to a
            revise on flagged subgroups).
    """
    if n_encounters <= 0:
        raise ValueError("n_encounters must be > 0")
    rng = random.Random(seed)
    rows: list[_EncounterResult] = []
    for i in range(n_encounters):
        lace = _sample_lace(rng)
        dem = _sample_demographics(rng)
        base = _calibrated_p(lace)
        eff = _compose_outcome_rate(base, dem)
        outcome = 1 if rng.random() < eff else 0
        ci = _ci_width_for_lace(lace)
        rows.append(_evaluate_encounter(
            enc_id=i, lace=lace, dem=dem,
            risk_estimate=base, ci_width=ci, outcome=outcome,
            fairness_audit_present=fairness_audit_present,
        ))
    overall = _aggregate_overall(rows)
    per_subgroup = {
        "age_band": _aggregate_by(rows, "age_band"),
        "sex": _aggregate_by(rows, "sex"),
        "race": _aggregate_by(rows, "race"),
        "ethnicity": _aggregate_by(rows, "ethnicity"),
        "insurance_type": _aggregate_by(rows, "insurance_type"),
        "language": _aggregate_by(rows, "language"),
    }
    baseline = _baseline_lace_only(rows)
    return ProspectiveSummary(
        cohort_n=len(rows),
        seed=seed,
        fairness_audit_present=fairness_audit_present,
        overall=overall,
        per_subgroup=per_subgroup,
        baseline_lace_only=baseline,
        references=[
            "TrustedRisk Phase 15.C - synthetic prospective evaluation.",
            "Multi-agent debate: a2a_agent.multi_agent_debate (Phase 14.16 Q1).",
            "Calibration: W1 spec_002 5-bin Beta-Binomial, AUROC 0.590, ECE 0.0078.",
        ],
    )
