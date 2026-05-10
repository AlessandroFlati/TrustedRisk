"""healthcare.compute_counterfactual_explanation -- what-if analysis on a RiskEstimate.

The tool answers: *what would the predicted readmission probability be if a
given LACE component were different?* For each of the four LACE factors
(L, A, C, E) it computes:

  - the probability if the factor's points were zero (counterfactual "no risk
    contribution from this axis"),
  - the probability if the factor's points were at the maximum allowed value,
  - the *delta* vs the current prediction.

It also identifies the single most-influential factor and describes a minimum
modification path to flip the recommendation toward a safer or riskier bin.

Implementation note: because the W1 coefficients are a lookup table indexed
by `lace_total`, we don't need gradient surgery -- we can simply re-index the
table at counterfactual LACE values. This makes the tool deterministic and
free of model surrogates.

The tool is **explanatory**, not **actionable**: most LACE factors are fixed
at discharge time (acuity already happened, prior ED visits already counted).
The counterfactual is for rendering "why this number?" not "how to change it".
"""

from __future__ import annotations

from typing import Any

from shared.schemas import (
    CounterfactualPath,
    CounterfactualReport,
    FactorCounterfactual,
    RiskEstimate,
)

from .readmission_risk import _load_coefficients


# ─────────────────────────────────────────────────────────────────────
# Per-factor metadata: max points + modifiability tag + plain rationale
# ─────────────────────────────────────────────────────────────────────

_FACTOR_META: dict[str, dict[str, Any]] = {
    "LACE_length_of_stay": {
        "max_points": 7,
        "modifiability": "partial",
        "rationale": (
            "Length of stay is partly modifiable at discharge time -- earlier "
            "discharge would lower this component, but only insofar as the "
            "clinical state actually permits it. Most often fixed by the time "
            "the decision is made."
        ),
    },
    "LACE_acuity": {
        "max_points": 3,
        "modifiability": "fixed",
        "rationale": (
            "Acuity-of-admission (via ED yes/no) is a property of the index "
            "admission itself. It cannot be changed at discharge time."
        ),
    },
    "LACE_comorbidity": {
        "max_points": 5,
        "modifiability": "fixed",
        "rationale": (
            "Charlson comorbidity index reflects pre-existing chronic conditions. "
            "Not modifiable at the discharge decision point; it is what it is."
        ),
    },
    "LACE_ed_visits_6mo": {
        "max_points": 4,
        "modifiability": "fixed",
        "rationale": (
            "Prior emergency-department visits in the last 6 months. Counted "
            "from the patient's history; cannot be modified retrospectively."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_counterfactual_explanation(
    risk: dict[str, Any] | RiskEstimate,
) -> CounterfactualReport:
    """Per-factor what-if sweep over a RiskEstimate.

    Args:
        risk: a `RiskEstimate` (or its dict form). Must have a non-empty
            `contributing_factors` list whose names match the LACE component
            convention (`LACE_length_of_stay`, `LACE_acuity`, `LACE_comorbidity`,
            `LACE_ed_visits_6mo`).

    Returns:
        CounterfactualReport: per-factor sweep + safer/riskier flip-paths.
    """
    risk_dict = _coerce_to_dict(risk)
    factors = risk_dict.get("contributing_factors") or []
    if not factors:
        raise ValueError("risk.contributing_factors is empty; cannot run counterfactual sweep")

    coef = _load_coefficients()
    lookup = (coef.get("runtime_coefficients") or {}).get("lookup_table", {})
    if not lookup:
        raise RuntimeError("coefficients.json runtime_coefficients.lookup_table is missing")

    current_lace = int(risk_dict.get("lace_raw_score", 0))
    current_prob = float(risk_dict.get("probability_mean", 0.0))

    # Build a name->points map from the current factors
    pts_by_name: dict[str, int] = {}
    raws_by_name: dict[str, Any] = {}
    for f in factors:
        n = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
        p = f.get("lace_points") if isinstance(f, dict) else getattr(f, "lace_points", 0)
        r = f.get("raw_value") if isinstance(f, dict) else getattr(f, "raw_value", 0.0)
        if n:
            pts_by_name[n] = int(p)
            raws_by_name[n] = r

    # Sum should match current_lace within ±1 (rounding/saturation)
    sum_check = sum(pts_by_name.values())
    sum_check = min(19, max(0, sum_check))

    factor_reports: list[FactorCounterfactual] = []
    most_influential_name = ""
    most_influential_abs = -1.0

    for name, current_pts in pts_by_name.items():
        meta = _FACTOR_META.get(name) or {
            "max_points": 7, "modifiability": "fixed",
            "rationale": "(no metadata for this factor)",
        }
        max_pts = int(meta["max_points"])

        # If this factor were ZERO: subtract current_pts from total
        lace_zero = max(0, min(19, current_lace - current_pts))
        prob_zero = _lookup_prob(lookup, lace_zero, current_prob)

        # If this factor were MAX: add (max_pts - current_pts) to total
        lace_max = max(0, min(19, current_lace + (max_pts - current_pts)))
        prob_max = _lookup_prob(lookup, lace_max, current_prob)

        d_zero = prob_zero - current_prob
        d_max = prob_max - current_prob

        factor_reports.append(FactorCounterfactual(
            factor_name=name,
            current_points=current_pts,
            current_raw=raws_by_name.get(name, 0.0),
            if_zero_lace_total=lace_zero,
            if_zero_prob_mean=prob_zero,
            if_max_lace_total=lace_max,
            if_max_prob_mean=prob_max,
            delta_prob_if_zero=d_zero,
            delta_prob_if_max=d_max,
            modifiability=meta["modifiability"],
            rationale=meta["rationale"],
        ))

        max_swing = max(abs(d_zero), abs(d_max))
        if max_swing > most_influential_abs:
            most_influential_abs = max_swing
            most_influential_name = name

    # Flip paths
    flip_safer = _compute_flip_path(
        lookup, current_lace, current_prob, pts_by_name, direction="safer",
    )
    flip_riskier = _compute_flip_path(
        lookup, current_lace, current_prob, pts_by_name, direction="riskier",
    )

    rationale = _build_rationale(
        current_lace, current_prob, factor_reports, most_influential_name,
        flip_safer, flip_riskier,
    )

    return CounterfactualReport(
        current_lace_total=current_lace,
        current_prob_mean=current_prob,
        factors=factor_reports,
        most_influential_factor=most_influential_name,
        flip_path_safer=flip_safer,
        flip_path_riskier=flip_riskier,
        rationale=rationale,
    )


def _coerce_to_dict(risk: Any) -> dict[str, Any]:
    if isinstance(risk, dict):
        return risk
    if hasattr(risk, "model_dump"):
        return risk.model_dump()
    raise TypeError(f"risk must be RiskEstimate or dict, got {type(risk).__name__}")


def _lookup_prob(lookup: dict[str, Any], lace: int, fallback: float) -> float:
    """Read prob_mean from the W1 coefficient lookup_table at LACE total."""
    entry = lookup.get(str(lace))
    if not isinstance(entry, dict):
        return fallback
    p = entry.get("prob_mean")
    return float(p) if p is not None else fallback


# ─────────────────────────────────────────────────────────────────────
# Flip-path search
# ─────────────────────────────────────────────────────────────────────

def _compute_flip_path(
    lookup: dict[str, Any],
    current_lace: int,
    current_prob: float,
    pts_by_name: dict[str, int],
    *,
    direction: str,  # "safer" | "riskier"
) -> CounterfactualPath | None:
    """Find the minimum-points modification that crosses a recommendation tier.

    Tier boundaries (heuristic, mirrors the v0.3 design):
      - low risk:   prob <= 0.10
      - moderate:   0.10 < prob <= 0.20
      - high:       prob > 0.20

    The function searches greedily over modifiable factors (in modifiability
    order) and returns the first path that crosses a tier boundary in the
    chosen direction. If no path exists in [0, 19] LACE space, returns None.
    """
    current_tier = _tier(current_prob)

    if direction == "safer":
        # Decrement points starting from most-modifiable factors
        modifiable_order = sorted(
            pts_by_name.keys(),
            key=lambda n: _modifiability_rank(n),
        )
        cumulative_decrement = 0
        chosen: list[str] = []
        for name in modifiable_order:
            cur_pts = pts_by_name[name]
            for delta in range(1, cur_pts + 1):
                trial_lace = max(0, current_lace - cumulative_decrement - delta)
                trial_prob = _lookup_prob(lookup, trial_lace, current_prob)
                if _tier(trial_prob) != current_tier and trial_prob < current_prob:
                    chosen.append(name)
                    return CounterfactualPath(
                        target_action="lower_risk_tier",
                        target_lace_max=trial_lace,
                        factors_to_change=chosen,
                        cumulative_delta_points=cumulative_decrement + delta,
                        achievable=any(
                            _FACTOR_META.get(n, {}).get("modifiability") in ("modifiable", "partial")
                            for n in chosen
                        ),
                        explanation=(
                            f"Reducing {chosen[-1]} by {delta} point(s) "
                            f"(plus prior reductions of {cumulative_decrement} pts) "
                            f"would land LACE at {trial_lace} (prob {trial_prob:.3f}), "
                            f"crossing into the {_tier_label(_tier(trial_prob))} risk tier."
                        ),
                    )
            cumulative_decrement += cur_pts
            chosen.append(name)
        return None

    # direction == "riskier"
    chosen = []
    cumulative_increment = 0
    # Pretend each factor could go up to its max
    for name in sorted(pts_by_name.keys()):
        cur_pts = pts_by_name[name]
        max_pts = _FACTOR_META.get(name, {}).get("max_points", 7)
        room = max(0, max_pts - cur_pts)
        for delta in range(1, room + 1):
            trial_lace = min(19, current_lace + cumulative_increment + delta)
            trial_prob = _lookup_prob(lookup, trial_lace, current_prob)
            if _tier(trial_prob) != current_tier and trial_prob > current_prob:
                chosen.append(name)
                return CounterfactualPath(
                    target_action="higher_risk_tier",
                    target_lace_max=trial_lace,
                    factors_to_change=chosen,
                    cumulative_delta_points=cumulative_increment + delta,
                    achievable=False,  # riskier paths are not actionable
                    explanation=(
                        f"Increasing {chosen[-1]} by {delta} point(s) "
                        f"(plus prior increases of {cumulative_increment} pts) "
                        f"would land LACE at {trial_lace} (prob {trial_prob:.3f}), "
                        f"crossing into the {_tier_label(_tier(trial_prob))} risk tier."
                    ),
                )
        cumulative_increment += room
        chosen.append(name)
    return None


def _modifiability_rank(name: str) -> int:
    rank_map = {"modifiable": 0, "partial": 1, "fixed": 2}
    mod = _FACTOR_META.get(name, {}).get("modifiability", "fixed")
    return rank_map.get(mod, 2)


def _tier(prob: float) -> str:
    if prob <= 0.10:
        return "low"
    if prob <= 0.20:
        return "moderate"
    return "high"


def _tier_label(t: str) -> str:
    return {"low": "low-risk", "moderate": "moderate-risk", "high": "high-risk"}.get(t, t)


def _build_rationale(
    current_lace: int, current_prob: float,
    factors: list[FactorCounterfactual], most_influential: str,
    flip_safer: CounterfactualPath | None,
    flip_riskier: CounterfactualPath | None,
) -> str:
    parts = [
        f"Current LACE = {current_lace} -> prob {current_prob:.3f}.",
        f"Most influential single factor: {most_influential}.",
    ]
    if flip_safer is not None:
        parts.append(
            f"Safer flip path: {flip_safer.cumulative_delta_points} pts reduction "
            f"reaches LACE {flip_safer.target_lace_max} "
            f"({'modifiable' if flip_safer.achievable else 'not actionable'})."
        )
    else:
        parts.append("No safer flip path within the LACE 0-19 space.")
    if flip_riskier is not None:
        parts.append(
            f"Riskier flip path: {flip_riskier.cumulative_delta_points} pts increase "
            f"reaches LACE {flip_riskier.target_lace_max} (illustrative only)."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_counterfactual_explanation)
