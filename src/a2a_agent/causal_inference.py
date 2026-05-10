"""SCALE-2 -- Causal-inference Average Treatment Effect via DoWhy.

Estimates the ATE of an intervention (e.g. pharmacist-led counseling) on a
binary outcome (e.g. 30-day readmission) from observational cohort data,
controlling for the supplied confounders. Goes beyond the literature-RRR
shortcut used in `compute_expected_value_of_intervention`: instead of
trusting a published RRR point estimate, this layer fits the effect on the
caller's own cohort and runs DoWhy's standard refutations:

  - random_common_cause: add a noise common cause; ATE should be unchanged
  - placebo_treatment: replace treatment with random; ATE should ≈ 0

Either refutation failing surfaces in `assumptions_warnings`.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import numpy as np

from shared.schemas import CausalATEReport, CausalRefutation


# Suppress DoWhy's chatty warnings during fit
warnings.filterwarnings("ignore", category=FutureWarning, module="dowhy")
warnings.filterwarnings("ignore", category=UserWarning, module="dowhy")


# ─────────────────────── Cohort coercion ───────────────────────

def _to_dataframe(rows: Iterable[dict[str, Any]]):
    """Build a pandas DataFrame from a list of cohort dicts."""
    import pandas as pd
    rows_list = list(rows)
    if not rows_list:
        return pd.DataFrame()
    return pd.DataFrame(rows_list)


def _validate_columns(df, treatment: str, outcome: str,
                        confounders: list[str]) -> None:
    for col in [treatment, outcome] + confounders:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in cohort.")


def _binary_or_numeric(series) -> bool:
    """Best-effort check that a column is numeric / 0-1 binary."""
    import pandas as pd
    if pd.api.types.is_numeric_dtype(series):
        return True
    return False


# ─────────────────────── Naive baseline ───────────────────────

def _naive_diff_in_means(df, treatment: str, outcome: str) -> float:
    treated = df[df[treatment] == 1][outcome]
    control = df[df[treatment] == 0][outcome]
    if len(treated) == 0 or len(control) == 0:
        return 0.0
    return float(treated.mean() - control.mean())


# ─────────────────────── ATE estimation via DoWhy ───────────────────────

def _estimate_with_dowhy(df, treatment: str, outcome: str,
                            confounders: list[str],
                            method: str = "linear_regression",
                            run_refutations: bool = True,
                            ) -> dict[str, Any]:
    """Run the DoWhy 4-step pipeline. Returns a dict -- no schema coupling here."""
    from dowhy import CausalModel
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimand_text = str(identified)

    if method == "linear_regression":
        method_name = "backdoor.linear_regression"
    elif method == "propensity_score_matching":
        method_name = "backdoor.propensity_score_matching"
    elif method == "propensity_score_weighting":
        method_name = "backdoor.propensity_score_weighting"
    else:
        method_name = "backdoor." + method

    estimate = model.estimate_effect(
        identified, method_name=method_name,
        target_units="ate",
    )

    refutations: list[dict[str, Any]] = []
    warnings_list: list[str] = []
    if run_refutations:
        try:
            r1 = model.refute_estimate(
                identified, estimate, method_name="random_common_cause",
            )
            new_effect = float(r1.new_effect)
            refutations.append({
                "name": "random_common_cause",
                "new_effect": new_effect,
                "p_value": getattr(r1, "p_value", None),
                "detail": "ATE should be unchanged after adding a random common cause.",
            })
            if abs(new_effect - float(estimate.value)) > \
                    0.5 * abs(float(estimate.value)) + 1e-3:
                warnings_list.append(
                    "random_common_cause refutation diverges > 50% from "
                    "the original ATE -- the estimate may be sensitive to "
                    "unmeasured confounders.")
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(
                f"random_common_cause refutation failed: {type(exc).__name__}")

        try:
            r2 = model.refute_estimate(
                identified, estimate, method_name="placebo_treatment_refuter",
                placebo_type="permute",
            )
            new_effect = float(r2.new_effect)
            refutations.append({
                "name": "placebo_treatment",
                "new_effect": new_effect,
                "p_value": getattr(r2, "p_value", None),
                "detail": "ATE should ≈ 0 when treatment is permuted.",
            })
            if abs(new_effect) > 0.5 * abs(float(estimate.value)) + 1e-3:
                warnings_list.append(
                    "placebo_treatment refutation produced a non-zero ATE -- "
                    "the original effect may be confounded.")
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(
                f"placebo_treatment refutation failed: {type(exc).__name__}")

    # Try to extract CI + std error from the underlying estimator. DoWhy's
    # generic `estimate.value` has both a point + (sometimes) interval.
    ci_low = getattr(estimate, "effect_interval_low", None) or \
        getattr(estimate, "lower_bound", None)
    ci_high = getattr(estimate, "effect_interval_high", None) or \
        getattr(estimate, "upper_bound", None)
    se = getattr(estimate, "std_err", None) or \
        getattr(estimate, "stderr", None)
    if hasattr(estimate, "estimator") and \
            hasattr(estimate.estimator, "model"):
        try:
            sm_results = estimate.estimator.model
            if hasattr(sm_results, "bse"):
                # statsmodels OLS -- pull SE for the treatment coefficient
                se = float(sm_results.bse[treatment])
                ci = sm_results.conf_int().loc[treatment]
                ci_low = float(ci[0])
                ci_high = float(ci[1])
        except Exception:
            pass

    return {
        "ate_point": float(estimate.value),
        "ate_ci95_low": (float(ci_low) if ci_low is not None else None),
        "ate_ci95_high": (float(ci_high) if ci_high is not None else None),
        "ate_std_error": (float(se) if se is not None else None),
        "estimand_text": estimand_text,
        "refutations": refutations,
        "warnings": warnings_list,
    }


# ─────────────────────── Public API ───────────────────────

def compute_average_treatment_effect(
    cohort: list[dict[str, Any]],
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
    method: str = "linear_regression",
    run_refutations: bool = True,
) -> CausalATEReport:
    """Estimate the ATE of `treatment` on `outcome` using DoWhy.

    Args:
        cohort: list of dicts; each represents one observation.
        treatment: column name (binary 0/1 column expected).
        outcome: column name (binary or continuous).
        confounders: list of columns to adjust for. Pass [] to estimate the
            naive difference (no adjustment) -- equivalent to a t-test.
        method: DoWhy backdoor estimator. Default "linear_regression".
            Other supported: "propensity_score_matching",
            "propensity_score_weighting".
        run_refutations: when True, run random_common_cause + placebo
            refutations.

    Returns:
        CausalATEReport with point estimate, CI, refutations, and any
        assumptions_warnings (e.g. unmeasured-confounder sensitivity).
    """
    confounders = list(confounders or [])

    df = _to_dataframe(cohort)
    if df.empty:
        raise ValueError("cohort is empty.")

    _validate_columns(df, treatment, outcome, confounders)

    if not _binary_or_numeric(df[treatment]):
        raise ValueError(
            f"treatment column {treatment!r} must be numeric (0/1 binary).")
    if not _binary_or_numeric(df[outcome]):
        raise ValueError(
            f"outcome column {outcome!r} must be numeric.")

    if df[treatment].nunique() < 2:
        raise ValueError(
            f"treatment column {treatment!r} has only one unique value -- "
            "cannot estimate ATE.")

    n = len(df)
    n_treated = int((df[treatment] == 1).sum())
    n_untreated = int((df[treatment] == 0).sum())
    naive = _naive_diff_in_means(df, treatment, outcome)

    if not confounders:
        # No adjustment -- return the naive estimator with explicit warning
        return CausalATEReport(
            treatment_name=treatment,
            outcome_name=outcome,
            n_observations=n,
            n_treated=n_treated,
            n_untreated=n_untreated,
            estimation_method="naive_difference_in_means",
            ate_point=naive,
            naive_difference_in_means=naive,
            assumptions_warnings=[
                "No confounders specified -- naive difference-in-means is "
                "biased when treatment assignment is not randomized."
            ],
            rationale=(
                f"Naive difference-in-means: {naive:.4f}. "
                "No backdoor adjustment performed."),
        )

    result = _estimate_with_dowhy(
        df, treatment, outcome, confounders,
        method=method, run_refutations=run_refutations,
    )

    refutations = [
        CausalRefutation(
            name=r["name"],                 # type: ignore[arg-type]
            new_effect=r["new_effect"],
            p_value=r.get("p_value"),
            detail=r.get("detail", ""),
        )
        for r in result["refutations"]
    ]

    rationale = (
        f"DoWhy backdoor.{method} ATE = {result['ate_point']:.4f} "
        f"(naive diff-in-means = {naive:.4f}, n_treated={n_treated}, "
        f"n_untreated={n_untreated}). "
        f"Confounders adjusted: {', '.join(confounders) if confounders else '(none)'}. "
        f"{len(refutations)} refutation(s) run; "
        f"{len(result['warnings'])} assumption warning(s)."
    )

    return CausalATEReport(
        treatment_name=treatment,
        outcome_name=outcome,
        n_observations=n,
        n_treated=n_treated,
        n_untreated=n_untreated,
        estimation_method=f"dowhy.backdoor.{method}",
        ate_point=result["ate_point"],
        ate_ci95_low=result["ate_ci95_low"],
        ate_ci95_high=result["ate_ci95_high"],
        ate_std_error=result["ate_std_error"],
        naive_difference_in_means=naive,
        identified_estimand_text=result["estimand_text"][:1500],
        refutations=refutations,
        assumptions_warnings=result["warnings"],
        rationale=rationale,
    )
