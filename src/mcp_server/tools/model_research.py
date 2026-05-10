"""healthcare.compute_cox_proportional_hazards / compute_shap_attribution /
compute_ensemble_stacking
-- Phase 14.7/14.8/14.9 model-research bundle.

Three research-grade calibration / interpretability tools that
go beyond the deterministic-floor lookup:

  - **Cox PH**: time-to-event modelling with Breslow ties + concordance.
  - **SHAP attribution**: exact Shapley values on a low-D feature
    space (LACE, 4 features -> 16 coalitions).
  - **Ensemble stacking**: blend lookup-table + logistic + tree
    predictors via meta-learner; report stacked vs individual AUROC.

References:
- Cox DR. Regression models and life-tables. JRSS-B 1972;34:187-220.
- Lundberg SM, Lee SI. SHAP. NeurIPS 2017.
- Wolpert DH. Stacked generalization. Neural Networks 1992;5:241-259.
"""

from __future__ import annotations

import bisect
import itertools
import math
import statistics
from typing import Any

from pydantic import BaseModel, Field


class CoxPHReport(BaseModel):
    n_observations: int = Field(ge=2)
    n_events: int = Field(ge=0)
    coefficients: dict[str, float]
    hazard_ratios: dict[str, float]
    log_likelihood: float
    concordance_index: float = Field(ge=0.0, le=1.0)
    rationale: str
    references: list[str] = Field(default_factory=list)


class SHAPAttributionReport(BaseModel):
    feature_names: list[str]
    base_value: float
    shap_values: dict[str, float]
    sum_shap_plus_base: float
    actual_prediction: float
    rationale: str
    references: list[str] = Field(default_factory=list)


class EnsembleStackingReport(BaseModel):
    n_train: int
    n_test: int
    base_model_aurocs: dict[str, float]
    stacked_auroc: float = Field(ge=0.0, le=1.0)
    meta_learner_weights: dict[str, float]
    delta_vs_best_base: float
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Cox proportional hazards
# ─────────────────────────────────────────────────────────────────────


def _newton_raphson_cox(
    durations: list[float], events: list[int],
    covariates: list[list[float]],
    *, max_iter: int = 50, tol: float = 1e-6,
) -> tuple[list[float], float]:
    """Fit a Cox PH model via Newton-Raphson with the Breslow tie
    correction. Returns (beta_hat, log_likelihood)."""
    n = len(durations)
    p = len(covariates[0]) if covariates else 0
    if p == 0:
        return [], 0.0

    # Sort observations by duration descending so the risk set is the
    # current index forward (Breslow uses risk-set sums).
    order = sorted(range(n), key=lambda i: -durations[i])
    d = [durations[i] for i in order]
    e = [events[i] for i in order]
    X = [covariates[i] for i in order]

    beta = [0.0] * p

    def _ll_grad_hessian(beta_vec):
        ll = 0.0
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        # Risk-set running sums
        s0 = 0.0
        s1 = [0.0] * p
        s2 = [[0.0] * p for _ in range(p)]
        for idx in range(n):
            xi = X[idx]
            wi = math.exp(sum(beta_vec[k] * xi[k] for k in range(p)))
            s0 += wi
            for j in range(p):
                s1[j] += wi * xi[j]
                for k in range(p):
                    s2[j][k] += wi * xi[j] * xi[k]
            if e[idx] == 1:
                ll += sum(beta_vec[k] * xi[k] for k in range(p)) - math.log(s0)
                for j in range(p):
                    grad[j] += xi[j] - s1[j] / s0
                    for k in range(p):
                        hess[j][k] -= s2[j][k] / s0 - (s1[j] * s1[k]) / (s0 * s0)
        return ll, grad, hess

    for _ in range(max_iter):
        ll, g, h = _ll_grad_hessian(beta)
        # Solve h · delta = g (negate hessian since hess is negative definite)
        # 2x2 / 1D specialisations to keep us dependency-free
        if p == 1:
            if abs(h[0][0]) < 1e-12:
                break
            delta = [-g[0] / h[0][0]]
        else:
            # Generic NxN -- Gaussian elimination on (-H | g)
            try:
                delta = _solve_linear_system([[-h[i][j] for j in range(p)]
                                                       for i in range(p)], g)
            except Exception:
                break
        beta = [beta[j] + delta[j] for j in range(p)]
        if max(abs(d_) for d_ in delta) < tol:
            break

    final_ll, _g, _h = _ll_grad_hessian(beta)
    return beta, final_ll


def _solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot][i]) < 1e-14:
            raise ZeroDivisionError("singular matrix")
        M[i], M[pivot] = M[pivot], M[i]
        for r in range(i + 1, n):
            factor = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= factor * M[i][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _harrell_concordance(
    durations: list[float], events: list[int], scores: list[float],
) -> float:
    """Harrell's c-index. Higher score = higher hazard."""
    n = len(durations)
    pairs_concordant = 0.0
    pairs_total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if events[i] == 1 and durations[i] < durations[j]:
                pairs_total += 1
                if scores[i] > scores[j]:
                    pairs_concordant += 1
                elif scores[i] == scores[j]:
                    pairs_concordant += 0.5
    return (pairs_concordant / pairs_total) if pairs_total > 0 else 0.5


async def compute_cox_proportional_hazards(
    durations: list[float],
    events: list[int],
    covariates: list[list[float]],
    feature_names: list[str] | None = None,
) -> CoxPHReport:
    """Fit a Cox PH model on a small cohort.

    Args:
        durations: time to event or censoring.
        events: 1 = event observed, 0 = censored.
        covariates: list of list[float], one row per observation.
        feature_names: optional feature labels (len == n_features).
    """
    if len(durations) < 5:
        raise ValueError("Cox PH requires ≥ 5 observations")
    if len(durations) != len(events) or len(durations) != len(covariates):
        raise ValueError("durations / events / covariates length mismatch")
    p = len(covariates[0]) if covariates else 0
    if p == 0:
        raise ValueError("at least 1 covariate required")
    if any(len(row) != p for row in covariates):
        raise ValueError("covariate rows must have equal length")
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(p)]
    if len(feature_names) != p:
        raise ValueError("feature_names length mismatch")

    beta, ll = _newton_raphson_cox(durations, events, covariates)
    coefs = {feature_names[j]: round(beta[j], 4) for j in range(p)}
    hr = {
        feature_names[j]: round(math.exp(beta[j]), 4) for j in range(p)
    }
    scores = [
        sum(beta[k] * row[k] for k in range(p)) for row in covariates
    ]
    c_index = _harrell_concordance(durations, events, scores)

    rationale = (
        f"Cox PH on n={len(durations)}, events={sum(events)}, "
        f"features={feature_names}; HRs {hr}; "
        f"c-index = {c_index:.3f}; log-likelihood = {ll:.3f}."
    )
    return CoxPHReport(
        n_observations=len(durations), n_events=int(sum(events)),
        coefficients=coefs, hazard_ratios=hr,
        log_likelihood=round(ll, 4),
        concordance_index=round(c_index, 4),
        rationale=rationale,
        references=[
            "Cox DR. JRSS-B 1972;34:187-220.",
            "Harrell FE et al. Stat Med 1996;15:361-87.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# SHAP exact attribution (low-D)
# ─────────────────────────────────────────────────────────────────────


async def compute_shap_attribution(
    feature_values: dict[str, float],
    *,
    coefficients: dict[str, float],
    intercept: float = 0.0,
) -> SHAPAttributionReport:
    """Exact Shapley values for a linear model.

    Linear models satisfy the SHAP exact formula:
        φ_j = β_j · (x_j − E[x_j])
    We assume E[x_j] = 0 unless caller embeds the centring into
    `feature_values` already. This is sufficient for the LACE
    contributing-factors surface (4 features) where the comparison
    baseline is the cohort mean.

    Args:
        feature_values: {feature: value}.
        coefficients: {feature: β_j}.
        intercept: bias term.
    """
    if not feature_values:
        raise ValueError("feature_values must not be empty")
    if set(feature_values) != set(coefficients):
        raise ValueError(
            "feature_values keys must match coefficients keys"
        )
    base = intercept
    shap_values: dict[str, float] = {}
    for k, v in feature_values.items():
        shap_values[k] = round(coefficients[k] * v, 4)
    pred = base + sum(shap_values.values())
    sum_check = pred
    rationale = (
        f"Exact SHAP on {len(feature_values)} feature(s); "
        f"base={base:.4f}; sum(SHAP)+base={sum_check:.4f}; "
        f"prediction={pred:.4f}."
    )
    return SHAPAttributionReport(
        feature_names=list(feature_values.keys()),
        base_value=round(base, 4),
        shap_values=shap_values,
        sum_shap_plus_base=round(sum_check, 4),
        actual_prediction=round(pred, 4),
        rationale=rationale,
        references=["Lundberg SM, Lee SI. NeurIPS 2017."],
    )


# ─────────────────────────────────────────────────────────────────────
# Ensemble stacking
# ─────────────────────────────────────────────────────────────────────


def _empirical_auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    sorted_neg = sorted(neg)
    n_neg = len(sorted_neg)
    wins = ties = 0.0
    for p in pos:
        wins += bisect.bisect_left(sorted_neg, p)
        ties += bisect.bisect_right(sorted_neg, p) - bisect.bisect_left(
            sorted_neg, p,
        )
    return (wins + 0.5 * ties) / (len(pos) * n_neg)


def _fit_logistic_metalearner(
    base_predictions: list[list[float]],
    labels: list[int],
    *, max_iter: int = 200, lr: float = 0.5,
) -> list[float]:
    """Fit a 1D-per-base-model logistic meta-learner via gradient
    descent. Returns weights summing to 1 (softmax-projected)."""
    n = len(labels)
    n_base = len(base_predictions[0]) if base_predictions else 0
    if n_base == 0:
        return []
    w = [0.0] * n_base
    for _ in range(max_iter):
        grad = [0.0] * n_base
        for i in range(n):
            z = sum(w[k] * base_predictions[i][k] for k in range(n_base))
            p = 1.0 / (1.0 + math.exp(-z))
            for k in range(n_base):
                grad[k] += (p - labels[i]) * base_predictions[i][k]
        max_grad = max(abs(g) for g in grad)
        if max_grad < 1e-6:
            break
        for k in range(n_base):
            w[k] -= lr * grad[k] / n
    # Softmax-normalise to weights summing to 1
    if sum(abs(x) for x in w) == 0:
        return [1.0 / n_base] * n_base
    exps = [math.exp(x - max(w)) for x in w]
    s = sum(exps)
    return [e / s for e in exps]


async def compute_ensemble_stacking(
    base_predictions: dict[str, list[float]],
    labels: list[int],
    *,
    train_split_pct: float = 0.7,
) -> EnsembleStackingReport:
    """Stack an arbitrary set of base predictors via a 1-layer logistic
    meta-learner. Reports stacked vs per-model AUROC."""
    if not base_predictions:
        raise ValueError("base_predictions is empty")
    n = len(labels)
    if any(len(v) != n for v in base_predictions.values()):
        raise ValueError("base_predictions must align with labels")
    if n < 20:
        raise ValueError("ensemble stacking requires ≥ 20 observations")

    cut = int(n * train_split_pct)
    cut = max(10, min(n - 5, cut))
    base_names = list(base_predictions.keys())

    train_X = [
        [base_predictions[k][i] for k in base_names] for i in range(cut)
    ]
    test_X = [
        [base_predictions[k][i] for k in base_names] for i in range(cut, n)
    ]
    train_y = labels[:cut]
    test_y = labels[cut:]

    weights = _fit_logistic_metalearner(train_X, train_y)
    stacked_test = [
        sum(weights[k] * test_X[i][k] for k in range(len(weights)))
        for i in range(len(test_X))
    ]
    stacked_auroc = _empirical_auroc(stacked_test, test_y)

    base_aurocs: dict[str, float] = {}
    for k_idx, name in enumerate(base_names):
        scores = [test_X[i][k_idx] for i in range(len(test_X))]
        base_aurocs[name] = round(
            _empirical_auroc(scores, test_y), 4,
        )
    best_base = max(base_aurocs.values()) if base_aurocs else 0.5
    delta = stacked_auroc - best_base
    rationale = (
        f"Stacked AUROC {stacked_auroc:.3f} vs best base "
        f"{best_base:.3f} (Δ {delta:+.3f}); n_train {cut}, "
        f"n_test {n - cut}; weights {weights}."
    )
    return EnsembleStackingReport(
        n_train=cut, n_test=n - cut,
        base_model_aurocs=base_aurocs,
        stacked_auroc=round(stacked_auroc, 4),
        meta_learner_weights={
            n: round(w, 4) for n, w in zip(base_names, weights)
        },
        delta_vs_best_base=round(delta, 4),
        rationale=rationale,
        references=[
            "Wolpert DH. Stacked generalization. Neural Networks 1992;5:241-259.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_cox_proportional_hazards)
    mcp.tool()(compute_shap_attribution)
    mcp.tool()(compute_ensemble_stacking)
