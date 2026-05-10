"""healthcare.compute_decision_utility -- expected utility over discharge actions.

Per design doc §3.3.2:
  E[U(a)] = sum_outcome U(outcome) * P(outcome | action) - cost(action)

With CI intervals on P(outcome), we compute stochastic dominance confidence
(% of MC samples where action A dominates all others in expected utility).

Coefficients (qaly_weights + cost_usd_per_day) come from coefficients.json
utility_fn block, produced by W1 critic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from shared.schemas import (
    Action,
    ProbInterval,
    UtilityAnalysis,
)

from ..fhir.client import resolve_patient_id


# ─────────────────────────────────────────────────────────────────────
# Load utility_fn from coefficients.json at module import
# ─────────────────────────────────────────────────────────────────────

_UTILITY_FN: dict[str, Any] | None = None


def _load_utility_fn() -> dict[str, Any]:
    global _UTILITY_FN
    if _UTILITY_FN is not None:
        return _UTILITY_FN
    path = os.environ.get(
        "TRUSTEDRISK_COEFFICIENTS_PATH", "data/coefficients.json"
    )
    fp = Path(path)
    if not fp.exists():
        raise RuntimeError(f"coefficients.json not found at {path}")
    with fp.open(encoding="utf-8") as f:
        data = json.load(f)
    utility_fn = data.get("utility_fn")
    if not utility_fn:
        raise RuntimeError("coefficients.json missing 'utility_fn' block")
    _UTILITY_FN = utility_fn
    return utility_fn


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_decision_utility(
    outcome_probs: dict[str, dict[str, dict[str, Any]]],
    patient_id: str | None = None,
    n_monte_carlo: int = 1000,
) -> UtilityAnalysis:
    """Expected QALY-weighted utility analysis across discharge actions.

    Args:
        outcome_probs: dict mapping action_name -> outcome_name -> {mean, ci95}.
            Typically wraps output of compute_readmission_risk per action.
        patient_id: for audit trail; not used in calculation.
        n_monte_carlo: bootstrap samples for dominance confidence (default 1000).

    Returns:
        UtilityAnalysis with scores, costs, dominant_action, dominance_confidence.
    """
    if patient_id is not None:
        await resolve_patient_id(patient_id)  # Validates but unused in calc

    utility_fn = _load_utility_fn()
    qaly_weights: dict[str, float] = utility_fn["qaly_weights"]
    cost_per_day: dict[str, float] = utility_fn["cost_usd_per_day"]
    horizon_days: int = int(utility_fn.get("time_horizon_days", 90))

    action_scores_mean: dict[Action, float] = {}
    action_scores_ci: dict[Action, tuple[float, float]] = {}
    action_costs: dict[Action, float] = {}
    mc_samples: dict[Action, np.ndarray] = {}

    rng = np.random.default_rng(seed=42)  # Reproducible audit

    # For each candidate action, compute expected QALY + MC samples for dominance
    for action_str, outcome_map in outcome_probs.items():
        try:
            action = Action(action_str)
        except ValueError:
            continue  # Skip unknown actions

        # Cost: fixed per-day * horizon_days
        cost = float(cost_per_day.get(action_str, 0.0)) * horizon_days
        action_costs[action] = cost

        # Compute MC samples of expected utility
        samples = np.zeros(n_monte_carlo, dtype=np.float64)
        for outcome_name, prob_block in outcome_map.items():
            prob_mean = float(prob_block.get("mean", 0.0))
            ci95 = prob_block.get("ci95", [prob_mean, prob_mean])
            ci_lo, ci_hi = float(ci95[0]), float(ci95[1])

            # Beta-approximation: sample from uniform over CI as a simple proxy
            prob_samples = rng.uniform(ci_lo, ci_hi, size=n_monte_carlo)

            outcome_weight = float(
                qaly_weights.get(f"{action_str}_{outcome_name}")
                or qaly_weights.get(outcome_name)
                or 0.0
            )
            samples += prob_samples * outcome_weight

        # Convert QALY-years to QALY-weeks (for display) -- weeks = years * 52
        # Subtract cost penalty normalized to a small share of QALY (placeholder $ to QALY)
        samples_weeks = samples * 52.0 - (cost / 10000.0)  # $10k ≈ 1 QALY-week penalty
        mc_samples[action] = samples_weeks
        action_scores_mean[action] = float(samples_weeks.mean())
        action_scores_ci[action] = (
            float(np.quantile(samples_weeks, 0.025)),
            float(np.quantile(samples_weeks, 0.975)),
        )

    # Dominance analysis
    dominant_action, dominance_confidence = _stochastic_dominance(mc_samples)

    # Human-readable trace
    reasoning_trace = _build_reasoning_trace(
        action_scores_mean, action_costs, dominant_action, dominance_confidence
    )

    return UtilityAnalysis(
        action_scores_qaly_weeks=action_scores_mean,
        action_scores_ci95=action_scores_ci,
        action_costs_usd=action_costs,
        dominant_action=dominant_action,
        dominance_confidence=dominance_confidence,
        reasoning_trace=reasoning_trace,
    )


def _stochastic_dominance(
    mc_samples: dict[Action, np.ndarray],
) -> tuple[Action | str, float]:
    """Compute stochastic dominance: % of MC iterations where one action wins."""
    if not mc_samples:
        return "INCONCLUSIVE", 0.0

    actions = list(mc_samples.keys())
    if len(actions) == 1:
        return actions[0], 1.0

    # For each action, fraction of samples where it strictly dominates all others
    n_samples = next(iter(mc_samples.values())).shape[0]
    wins = {a: 0 for a in actions}
    for i in range(n_samples):
        values = [(mc_samples[a][i], a) for a in actions]
        best_val, best_action = max(values)
        # Check strictness: no other action within 1% of best
        others_close = sum(1 for v, a in values if a != best_action and v >= best_val * 0.99)
        if others_close == 0:
            wins[best_action] += 1

    best_action, best_wins = max(wins.items(), key=lambda kv: kv[1])
    dominance_confidence = best_wins / n_samples

    # Declare INCONCLUSIVE if dominance is weak
    if dominance_confidence < 0.50:
        return "INCONCLUSIVE", dominance_confidence
    return best_action, dominance_confidence


def _build_reasoning_trace(
    scores: dict[Action, float],
    costs: dict[Action, float],
    dominant: Action | str,
    confidence: float,
) -> str:
    parts = []
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    score_summary = ", ".join(
        f"{a.value}={v:.2f} QALY-wk (${costs.get(a, 0):.0f})"
        for a, v in sorted_scores
    )
    parts.append(f"Action scores (sorted desc): {score_summary}.")
    if dominant == "INCONCLUSIVE":
        parts.append(
            f"No stochastic dominance (best candidate wins only {confidence * 100:.0f}% of "
            f"bootstrap iterations). Recommend clinician review."
        )
    else:
        action_name = dominant.value if hasattr(dominant, "value") else str(dominant)
        parts.append(
            f"{action_name} stochastically dominates alternatives in "
            f"{confidence * 100:.0f}% of bootstrap iterations."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_decision_utility)
