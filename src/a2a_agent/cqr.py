"""Phase 17.AY - Conformalised quantile regression (Romano 2019).

Romano, Patterson, Candès 2019 "Conformalized Quantile Regression"
(NeurIPS). Combines a quantile-regression model + a conformal
calibration step to produce prediction intervals with marginal
coverage guarantee:

    P( y_test in [q_lo(x), q_hi(x)] + adjustment ) >= 1 - alpha

The pipeline:
  1. Fit a *lower* quantile regression at level alpha/2 + an *upper*
     quantile regression at 1 - alpha/2 on the training half.
  2. On a held-out calibration set compute the conformity score
        E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
     This is positive when y is *outside* the predicted interval.
  3. Take the (1 - alpha) empirical quantile Q of {E_i}.
  4. The CQR prediction interval is
        [q_lo(x_test) - Q, q_hi(x_test) + Q]

This module fits the quantile regressors itself via *pinball-loss
gradient descent* on a linear surrogate (no NumPy / no autograd) -
sufficient for the small LACE-style feature spaces TrustedRisk uses.

Pure-Python deterministic.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Pinball-loss linear quantile regression
# ─────────────────────────────────────────────────────────────────────


class LinearQuantileModel(BaseModel):
    quantile: float = Field(gt=0.0, lt=1.0)
    feature_names: list[str]
    coefficients: list[float]
    intercept: float


def _pinball_subgradient(
    pred: float, y: float, quantile: float,
) -> float:
    """Subgradient of the pinball loss w.r.t. ``pred``."""
    if y >= pred:
        return -quantile
    return 1.0 - quantile


def fit_linear_quantile(
    *,
    X: list[list[float]],
    y: list[float],
    quantile: float,
    feature_names: list[str] | None = None,
    learning_rate: float = 0.01,
    n_iterations: int = 500,
    seed: int = 17,
) -> LinearQuantileModel:
    """SGD on the pinball loss for a linear quantile regressor."""
    if not X:
        raise ValueError("X cannot be empty")
    if len(X) != len(y):
        raise ValueError("X and y must align")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    p = len(X[0])
    feature_names = (
        feature_names or [f"x{i}" for i in range(p)]
    )
    if len(feature_names) != p:
        raise ValueError(
            "feature_names must align with X")
    rng = random.Random(seed)
    beta = [0.0] * p
    intercept = 0.0
    n = len(X)
    for it in range(n_iterations):
        i = rng.randrange(n)
        xi = X[i]
        pred = intercept + sum(b * v for b, v in zip(beta, xi))
        sg = _pinball_subgradient(pred, y[i], quantile)
        for k in range(p):
            beta[k] -= learning_rate * sg * xi[k]
        intercept -= learning_rate * sg
    return LinearQuantileModel(
        quantile=quantile,
        feature_names=list(feature_names),
        coefficients=[round(b, 6) for b in beta],
        intercept=round(intercept, 6),
    )


def predict_quantile(
    model: LinearQuantileModel, x: list[float],
) -> float:
    return model.intercept + sum(
        b * v for b, v in zip(model.coefficients, x)
    )


# ─────────────────────────────────────────────────────────────────────
# Conformalisation
# ─────────────────────────────────────────────────────────────────────


def _quantile_of(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_v[lo]
    frac = idx - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


class CQRReport(BaseModel):
    target_coverage: float = Field(ge=0.0, le=1.0)
    n_train: int = Field(ge=0)
    n_calibration: int = Field(ge=0)
    n_test: int = Field(ge=0)
    quantile_threshold: float
    empirical_coverage_test: float = Field(ge=0.0, le=1.0)
    average_interval_width: float = Field(ge=0.0)
    intervals: list[tuple[float, float]]
    rationale: str


def fit_cqr(
    *,
    X_train: list[list[float]], y_train: list[float],
    X_calibration: list[list[float]], y_calibration: list[float],
    X_test: list[list[float]], y_test: list[float] | None = None,
    target_coverage: float = 0.90,
    feature_names: list[str] | None = None,
    learning_rate: float = 0.01,
    n_iterations: int = 500,
    seed: int = 17,
) -> CQRReport:
    """Train + conformalise + emit prediction intervals for X_test.

    When y_test is supplied we also compute the empirical coverage."""
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be in (0, 1)")
    if not X_train or not X_calibration:
        raise ValueError("training + calibration sets cannot be empty")
    alpha = 1.0 - target_coverage
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    lo_model = fit_linear_quantile(
        X=X_train, y=y_train, quantile=q_lo,
        feature_names=feature_names,
        learning_rate=learning_rate, n_iterations=n_iterations,
        seed=seed,
    )
    hi_model = fit_linear_quantile(
        X=X_train, y=y_train, quantile=q_hi,
        feature_names=feature_names,
        learning_rate=learning_rate, n_iterations=n_iterations,
        seed=seed + 1,
    )
    # Conformity scores on calibration set
    scores: list[float] = []
    for x, y in zip(X_calibration, y_calibration):
        lo = predict_quantile(lo_model, x)
        hi = predict_quantile(hi_model, x)
        scores.append(max(lo - y, y - hi))
    n_cal = len(scores)
    # CQR uses (1 - alpha)(1 + 1/n_cal) quantile (Romano 2019 §3)
    q_for_quantile = min(
        1.0,
        (1.0 - alpha) * (1.0 + 1.0 / max(1, n_cal)),
    )
    threshold = _quantile_of(scores, q_for_quantile)
    intervals: list[tuple[float, float]] = []
    n_in = 0
    for i, x in enumerate(X_test):
        lo = predict_quantile(lo_model, x) - threshold
        hi = predict_quantile(hi_model, x) + threshold
        intervals.append((round(lo, 6), round(hi, 6)))
        if y_test is not None and lo <= y_test[i] <= hi:
            n_in += 1
    coverage = (
        n_in / len(X_test) if y_test is not None else 0.0
    )
    avg_width = (
        sum(h - l for l, h in intervals) / len(intervals)
        if intervals else 0.0
    )
    return CQRReport(
        target_coverage=target_coverage,
        n_train=len(X_train),
        n_calibration=n_cal,
        n_test=len(X_test),
        quantile_threshold=round(threshold, 6),
        empirical_coverage_test=round(coverage, 6),
        average_interval_width=round(avg_width, 6),
        intervals=intervals,
        rationale=(
            f"CQR target coverage {target_coverage*100:.0f}%; "
            f"conformity-score quantile threshold "
            f"{threshold:.4f} on {n_cal} calibration samples; "
            f"empirical coverage {coverage*100:.2f}% on test."
        ),
    )
