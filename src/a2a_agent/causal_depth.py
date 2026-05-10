"""Phase 17.M - Causal + counterfactual depth.

Three pure-Python tools that close the causal-inference gap:

  1. **Wachter 2018 min-distance counterfactual** -
     ``find_min_distance_counterfactual`` solves the L1-minimisation
     :math:`\\min_{x'} ||x' - x||_1` s.t. :math:`f(x') >= flip`
     via projected coordinate descent on a sigmoid model. No
     scipy/cvxpy dependency.

  2. **Rosenbaum sensitivity bounds** -
     ``rosenbaum_gamma_bound`` returns the upper-bound on the
     unmeasured-confounder strength :math:`Gamma` that would explain
     away an observed treatment effect (matched-pairs design).

  3. **Instrumental variables + front-door criterion** -
     ``wald_iv_estimate`` (Wald estimator: cov(Y, Z) / cov(D, Z)) and
     ``front_door_adjustment`` per Pearl 2009 §3.3.

Pure-Python, deterministic, stdlib-only.
"""

from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Wachter 2018 min-distance counterfactual
# ─────────────────────────────────────────────────────────────────────


class WachterCounterfactualReport(BaseModel):
    original_features: dict[str, float]
    counterfactual_features: dict[str, float]
    original_score: float
    counterfactual_score: float
    target_score_at_least: float
    delta_l1: float = Field(ge=0.0)
    delta_per_feature: dict[str, float]
    n_iterations: int
    converged: bool
    rationale: str


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _logistic_score(
    features: dict[str, float],
    coefficients: dict[str, float],
    intercept: float,
) -> float:
    z = intercept + sum(
        coefficients.get(k, 0.0) * features.get(k, 0.0)
        for k in coefficients
    )
    return _sigmoid(z)


def find_min_distance_counterfactual(
    *,
    features: dict[str, float],
    coefficients: dict[str, float],
    intercept: float,
    target_score_at_least: float = 0.5,
    feature_min: dict[str, float] | None = None,
    feature_max: dict[str, float] | None = None,
    max_iterations: int = 500,
    learning_rate: float = 2.0,
    lambda_l1: float = 0.0005,
) -> WachterCounterfactualReport:
    """Return the minimum-L1-distance counterfactual feature vector
    that pushes the logistic score above ``target_score_at_least``.

    Wachter et al. 2018 "Counterfactual Explanations Without Opening
    the Black Box" - minimisation
    :math:`L(x') = (f(x') - target)^2 + lambda * ||x' - x||_1`.
    Solved by projected gradient descent.

    All-stdlib. ``feature_min`` / ``feature_max`` clip per-feature so
    the counterfactual stays in a plausible domain (e.g. age >= 0,
    LACE component in [0, 5]).
    """
    if not coefficients:
        raise ValueError("coefficients cannot be empty")
    if not 0.0 < target_score_at_least < 1.0:
        raise ValueError("target_score_at_least must be in (0,1)")
    feature_min = feature_min or {}
    feature_max = feature_max or {}

    cf = dict(features)
    original_score = _logistic_score(features, coefficients, intercept)

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iterations + 1):
        score = _logistic_score(cf, coefficients, intercept)
        if score >= target_score_at_least:
            converged = True
            break
        # Gradient of (score - target)^2 + lambda * |x' - x|_1
        # d/dx_k = 2 * (score - target) * score * (1 - score) * w_k
        #        + lambda * sign(x'_k - x_k)
        diff = score - target_score_at_least
        sigma = score * (1.0 - score)
        for k, w in coefficients.items():
            sign = 0.0
            cf_k = cf.get(k, 0.0)
            x_k = features.get(k, 0.0)
            if cf_k > x_k:
                sign = 1.0
            elif cf_k < x_k:
                sign = -1.0
            grad = 2.0 * diff * sigma * w + lambda_l1 * sign
            cf[k] = cf_k - learning_rate * grad
            if k in feature_min:
                cf[k] = max(cf[k], feature_min[k])
            if k in feature_max:
                cf[k] = min(cf[k], feature_max[k])

    counterfactual_score = _logistic_score(
        cf, coefficients, intercept)
    delta_l1 = sum(
        abs(cf.get(k, 0.0) - features.get(k, 0.0))
        for k in coefficients
    )
    delta_per_feature = {
        k: round(cf.get(k, 0.0) - features.get(k, 0.0), 6)
        for k in coefficients
    }
    return WachterCounterfactualReport(
        original_features={k: round(v, 6)
                           for k, v in features.items()},
        counterfactual_features={k: round(v, 6) for k, v in cf.items()},
        original_score=round(original_score, 6),
        counterfactual_score=round(counterfactual_score, 6),
        target_score_at_least=target_score_at_least,
        delta_l1=round(delta_l1, 6),
        delta_per_feature=delta_per_feature,
        n_iterations=n_iter,
        converged=converged,
        rationale=(
            f"Wachter 2018 min-distance counterfactual via projected "
            f"gradient descent ({n_iter} iterations, "
            f"{'converged' if converged else 'budget exhausted'}). "
            f"Score moved {original_score:.3f} -> "
            f"{counterfactual_score:.3f} at L1 distance "
            f"{delta_l1:.3f}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Rosenbaum sensitivity bounds
# ─────────────────────────────────────────────────────────────────────


class RosenbaumSensitivityReport(BaseModel):
    n_pairs_total: int
    n_pairs_discordant: int
    s_pairs_treatment_better: int
    observed_p_value: float
    gamma_upper_bound: float
    gamma_critical_for_p_05: float
    rationale: str


def _binomial_p_one_sided(s: int, n: int, p: float) -> float:
    """One-sided binomial: P(X >= s) when X ~ Bin(n, p).
    Pure-stdlib: log-space cumulative."""
    if n == 0 or s > n:
        return 0.0 if s > 0 else 1.0
    if s <= 0:
        return 1.0
    log_p = math.log(p) if p > 0 else float("-inf")
    log_q = math.log(1.0 - p) if p < 1 else float("-inf")
    log_total: float = float("-inf")
    log_choose = 0.0
    # log_choose = log(n choose k) - we step k from 0 to s-1 then
    # accumulate from k=s onward, but easier to just sum from s..n.
    # Compute log(n choose k) iteratively.
    # Init for k=0: log_choose=0
    # Step to k via multiplying by (n-k+1)/k
    for k in range(0, n + 1):
        if k > 0:
            log_choose += math.log(n - k + 1) - math.log(k)
        if k >= s:
            log_term = log_choose + k * log_p + (n - k) * log_q
            if log_total == float("-inf"):
                log_total = log_term
            else:
                m = max(log_total, log_term)
                log_total = m + math.log(
                    math.exp(log_total - m) + math.exp(log_term - m)
                )
    return math.exp(log_total) if log_total != float("-inf") else 0.0


def rosenbaum_gamma_bound(
    *,
    n_discordant_pairs: int,
    s_treatment_better: int,
    target_p_value: float = 0.05,
) -> RosenbaumSensitivityReport:
    """Compute the Rosenbaum upper-bound :math:`\\Gamma` - the
    strength of an unmeasured confounder that would push the matched-
    pairs sign test's p-value to ``target_p_value``.

    Under the null + Gamma=1, sign-test uses p=0.5; under unmeasured
    confounding bounded by :math:`\\Gamma`, the upper bound on the
    treatment-better probability is :math:`p^+ = Gamma / (1 + Gamma)`.
    Solve for the smallest :math:`\\Gamma` whose
    :math:`P(X >= s | n, p^+) <= target_p_value`.
    """
    if n_discordant_pairs <= 0:
        raise ValueError("n_discordant_pairs must be > 0")
    if not 0 <= s_treatment_better <= n_discordant_pairs:
        raise ValueError(
            "s_treatment_better must be in [0, n_discordant_pairs]"
        )
    if not 0.0 < target_p_value < 1.0:
        raise ValueError("target_p_value must be in (0,1)")

    observed_p = _binomial_p_one_sided(
        s_treatment_better, n_discordant_pairs, 0.5)

    gamma = 1.0
    step = 0.05
    critical_gamma = 1.0
    found = False
    while gamma <= 50.0:
        p_plus = gamma / (1.0 + gamma)
        p_value_at_gamma = _binomial_p_one_sided(
            s_treatment_better, n_discordant_pairs, p_plus)
        if p_value_at_gamma > target_p_value and not found:
            critical_gamma = round(gamma, 4)
            found = True
            break
        gamma += step
    if not found:
        critical_gamma = 50.0

    return RosenbaumSensitivityReport(
        n_pairs_total=n_discordant_pairs,
        n_pairs_discordant=n_discordant_pairs,
        s_pairs_treatment_better=s_treatment_better,
        observed_p_value=round(observed_p, 6),
        gamma_upper_bound=critical_gamma,
        gamma_critical_for_p_05=critical_gamma,
        rationale=(
            f"Rosenbaum sensitivity: observed sign-test p="
            f"{observed_p:.4f}; the smallest unmeasured-confounder "
            f"strength Gamma at which p exceeds {target_p_value} is "
            f"{critical_gamma:.3f}. Larger Gamma -> result is more "
            f"robust to hidden bias."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Instrumental variables (Wald estimator)
# ─────────────────────────────────────────────────────────────────────


class WaldIVReport(BaseModel):
    n: int
    cov_y_z: float
    cov_d_z: float
    iv_ate_estimate: float
    rationale: str


def _cov(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    return sum(
        (x - mx) * (y - my) for x, y in zip(xs, ys)
    ) / (n - 1)


def wald_iv_estimate(
    *,
    outcome: list[float],
    treatment: list[float],
    instrument: list[float],
) -> WaldIVReport:
    """Wald estimator for IV ATE:

        ATE = cov(Y, Z) / cov(D, Z)

    Assumes the binary IV (Z) satisfies relevance + exclusion +
    monotonicity. Under those, this is the LATE estimate."""
    if not (len(outcome) == len(treatment) == len(instrument)):
        raise ValueError(
            "outcome / treatment / instrument lists must align")
    if len(outcome) < 2:
        raise ValueError("need >=2 observations")
    cov_yz = _cov(outcome, instrument)
    cov_dz = _cov(treatment, instrument)
    if abs(cov_dz) < 1e-12:
        raise ValueError(
            "instrument has zero covariance with treatment - "
            "instrument not relevant"
        )
    ate = cov_yz / cov_dz
    return WaldIVReport(
        n=len(outcome),
        cov_y_z=round(cov_yz, 6),
        cov_d_z=round(cov_dz, 6),
        iv_ate_estimate=round(ate, 6),
        rationale=(
            f"Wald IV ATE = cov(Y, Z) / cov(D, Z) = "
            f"{cov_yz:.4f} / {cov_dz:.4f} = {ate:.4f}. Assumes "
            f"relevance + exclusion + monotonicity (LATE)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Front-door adjustment (Pearl 2009)
# ─────────────────────────────────────────────────────────────────────


class FrontDoorReport(BaseModel):
    p_y_do_x: dict[str, float]
    rationale: str


def front_door_adjustment(
    *,
    p_m_given_x: dict[str, dict[str, float]],
    p_y_given_xm: dict[str, dict[str, dict[str, float]]],
    p_x: dict[str, float],
    treatment_values: list[str],
    outcome_value: str,
) -> FrontDoorReport:
    """Compute :math:`P(Y=y | do(X=x))` via the Pearl 2009 front-door
    adjustment formula:

        P(Y=y | do(X=x)) = sum_m P(M=m | X=x) *
            sum_x' P(X=x') * P(Y=y | X=x', M=m)

    Args:
        p_m_given_x: P(M=m | X=x) - p_m_given_x[x][m].
        p_y_given_xm: P(Y=y | X=x, M=m) - p_y_given_xm[x][m][y].
        p_x: marginal P(X=x).
        treatment_values: the X values to compute do(X=x) for.
        outcome_value: the y value to compute P(Y=y | do(X=x)) for.
    """
    if not treatment_values:
        raise ValueError("treatment_values cannot be empty")
    out: dict[str, float] = {}
    for x_target in treatment_values:
        if x_target not in p_m_given_x:
            raise ValueError(
                f"missing P(M | X={x_target}) entry")
        total = 0.0
        for m, p_m in p_m_given_x[x_target].items():
            inner = 0.0
            for x_prime, p_xp in p_x.items():
                cond = (
                    p_y_given_xm.get(x_prime, {})
                    .get(m, {}).get(outcome_value, 0.0)
                )
                inner += p_xp * cond
            total += p_m * inner
        out[x_target] = round(total, 6)
    return FrontDoorReport(
        p_y_do_x=out,
        rationale=(
            f"Front-door adjustment for outcome y={outcome_value} "
            f"across treatment values {treatment_values}. "
            f"Formula: P(Y=y|do(X=x)) = sum_m P(M=m|X=x) * "
            f"sum_x' P(X=x') P(Y=y|X=x',M=m). Pearl 2009 §3.3."
        ),
    )
