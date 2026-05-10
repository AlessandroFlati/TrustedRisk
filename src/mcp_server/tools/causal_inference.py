"""healthcare.compute_causal_treatment_effect -- Phase 12.6 B1.

MCP-surface wrapper around `a2a_agent.causal_inference.compute_average_treatment_effect`.

Estimates the average treatment effect (ATE) of an intervention on a
binary outcome from observational cohort data, controlling for the
caller-supplied confounders. Goes beyond the literature-RRR shortcut
in `compute_expected_value_of_intervention` by fitting the effect on
the caller's own cohort and running DoWhy's standard refutations:

  - random_common_cause: add a noise common cause; ATE should be
    unchanged under refutation.
  - placebo_treatment: replace treatment with random; ATE should
    collapse to ≈ 0 under refutation.

Either refutation failing surfaces in `assumptions_warnings`.

Pure-deterministic in the sense that the computation is reproducible
given the same input cohort + method + RNG seed. The DoWhy backbone
is the only dependency.

References:
- Pearl J. *Causality: Models, Reasoning and Inference*. 2nd ed.
  Cambridge University Press (2009).
- Sharma A, Kiciman E. DoWhy: An End-to-End Library for Causal
  Inference. arXiv:2011.04216 (2020).
"""

from __future__ import annotations

from typing import Any

from shared.schemas import CausalATEReport


async def compute_causal_treatment_effect(
    cohort: list[dict[str, Any]],
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
    method: str = "linear_regression",
    run_refutations: bool = True,
) -> CausalATEReport:
    """Estimate the ATE of `treatment` on `outcome` from cohort data.

    Args:
        cohort: list of dicts; each represents one observation.
        treatment: column name (binary 0/1 column expected).
        outcome: column name (binary or continuous).
        confounders: list of columns to adjust for. Pass `[]` to
            estimate the naive difference (no adjustment) -- equivalent
            to a t-test.
        method: DoWhy backdoor estimator. Default "linear_regression".
            Other supported: "propensity_score_matching",
            "propensity_score_weighting".
        run_refutations: when True, run random_common_cause + placebo
            refutations and surface failures in `assumptions_warnings`.

    Returns:
        CausalATEReport with point estimate, CI, refutations, and any
        assumptions_warnings (e.g. unmeasured-confounder sensitivity).
    """
    # Lazy import -- avoids dragging DoWhy / pandas into the registry
    # boot path when this tool is not invoked.
    from a2a_agent.causal_inference import (
        compute_average_treatment_effect,
    )
    return compute_average_treatment_effect(
        cohort=cohort,
        treatment=treatment,
        outcome=outcome,
        confounders=confounders,
        method=method,
        run_refutations=run_refutations,
    )


def register(mcp) -> None:
    mcp.tool()(compute_causal_treatment_effect)
