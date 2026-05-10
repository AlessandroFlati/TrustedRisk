"""Phase 17.AG - Anchors + LIME local explanations.

Two pure-Python local explainers for any opaque-but-callable
classifier ``f(features: dict[str, float]) -> float``:

  - **LIME** (Ribeiro 2016) - fit a sparse linear surrogate around
    the instance via local perturbations + locality-weighted least
    squares. Returns per-feature contributions.

  - **Anchors** (Ribeiro 2018) - greedy beam search for the
    minimum-cardinality "anchor" set of features whose presence
    keeps the prediction stable above a confidence threshold.

Pure-Python deterministic. Suited to small feature spaces (LACE,
HEART, etc.). No NumPy/SciPy.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from pydantic import BaseModel, Field


_Predictor = Callable[[dict[str, float]], float]


# ─────────────────────────────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────────────────────────────


class LIMEAttribution(BaseModel):
    feature: str
    weight: float
    direction: str   # "increases" / "decreases" / "neutral"


class LIMEReport(BaseModel):
    n_perturbations: int
    instance: dict[str, float]
    instance_score: float
    attributions: list[LIMEAttribution]
    intercept: float
    surrogate_r2: float
    rationale: str


def _kernel_weight(distance: float, kernel_width: float) -> float:
    return math.exp(-(distance ** 2) / max(1e-9, kernel_width ** 2))


def _solve_normal_equations(
    X: list[list[float]], y: list[float], w: list[float],
) -> tuple[list[float], float]:
    """Weighted least squares with manual Gaussian elimination.
    Returns (coefficients, intercept)."""
    n = len(y)
    if n == 0:
        return [], 0.0
    # Augment X with a 1 column (intercept)
    p = len(X[0]) + 1
    XT_W_X = [[0.0] * p for _ in range(p)]
    XT_W_y = [0.0] * p
    for i in range(n):
        row = X[i] + [1.0]
        wi = w[i]
        for a in range(p):
            for b in range(p):
                XT_W_X[a][b] += wi * row[a] * row[b]
            XT_W_y[a] += wi * row[a] * y[i]
    # Solve via Gaussian elimination with partial pivoting
    for col in range(p):
        pivot = col
        for r in range(col + 1, p):
            if abs(XT_W_X[r][col]) > abs(XT_W_X[pivot][col]):
                pivot = r
        if abs(XT_W_X[pivot][col]) < 1e-12:
            continue
        XT_W_X[col], XT_W_X[pivot] = XT_W_X[pivot], XT_W_X[col]
        XT_W_y[col], XT_W_y[pivot] = XT_W_y[pivot], XT_W_y[col]
        pv = XT_W_X[col][col]
        for j in range(col, p):
            XT_W_X[col][j] /= pv
        XT_W_y[col] /= pv
        for r in range(p):
            if r == col:
                continue
            factor = XT_W_X[r][col]
            for j in range(col, p):
                XT_W_X[r][j] -= factor * XT_W_X[col][j]
            XT_W_y[r] -= factor * XT_W_y[col]
    coeffs = XT_W_y[:-1]
    intercept = XT_W_y[-1]
    return coeffs, intercept


def lime_explain(
    *,
    instance: dict[str, float],
    predictor: _Predictor,
    feature_perturbation_scale: dict[str, float] | None = None,
    n_perturbations: int = 200,
    kernel_width: float = 0.75,
    seed: int = 17,
    top_k: int | None = None,
) -> LIMEReport:
    """Local linear surrogate around ``instance``."""
    if not instance:
        raise ValueError("instance cannot be empty")
    rng = random.Random(seed)
    feature_names = list(instance.keys())
    perturb = (
        feature_perturbation_scale
        if feature_perturbation_scale is not None
        else {f: 1.0 for f in feature_names}
    )
    instance_score = float(predictor(instance))
    X: list[list[float]] = []
    y: list[float] = []
    distances: list[float] = []
    for _ in range(n_perturbations):
        candidate: dict[str, float] = {}
        d = 0.0
        for f in feature_names:
            scale = max(1e-6, perturb.get(f, 1.0))
            delta = rng.gauss(0, scale)
            candidate[f] = float(instance[f]) + delta
            d += (delta / scale) ** 2
        X.append([candidate[f] - instance[f] for f in feature_names])
        y.append(float(predictor(candidate)) - instance_score)
        distances.append(math.sqrt(d))
    weights = [
        _kernel_weight(d, kernel_width) for d in distances
    ]
    coeffs, intercept = _solve_normal_equations(X, y, weights)
    # Surrogate R^2 on the perturbations (weighted)
    y_mean_w = (
        sum(w * yi for w, yi in zip(weights, y))
        / max(1e-12, sum(weights))
    )
    ss_tot = sum(
        w * (yi - y_mean_w) ** 2 for w, yi in zip(weights, y)
    )
    ss_res = 0.0
    for i in range(len(y)):
        pred = intercept + sum(
            coeffs[j] * X[i][j] for j in range(len(coeffs))
        )
        ss_res += weights[i] * (y[i] - pred) ** 2
    r2 = 1.0 - (ss_res / max(1e-12, ss_tot))
    attributions: list[LIMEAttribution] = []
    for f, w in zip(feature_names, coeffs):
        if w > 1e-9:
            direction = "increases"
        elif w < -1e-9:
            direction = "decreases"
        else:
            direction = "neutral"
        attributions.append(LIMEAttribution(
            feature=f, weight=round(w, 6), direction=direction,
        ))
    attributions.sort(key=lambda a: -abs(a.weight))
    if top_k is not None:
        attributions = attributions[:top_k]
    return LIMEReport(
        n_perturbations=n_perturbations,
        instance={k: round(v, 6) for k, v in instance.items()},
        instance_score=round(instance_score, 6),
        attributions=attributions,
        intercept=round(intercept, 6),
        surrogate_r2=round(r2, 6),
        rationale=(
            f"LIME local linear surrogate over {n_perturbations} "
            f"weighted perturbations; surrogate R^2={r2:.3f}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Anchors (Ribeiro 2018)
# ─────────────────────────────────────────────────────────────────────


class AnchorReport(BaseModel):
    instance: dict[str, float]
    instance_prediction: int
    anchor_features: list[str]
    precision: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    n_samples_evaluated: int
    rationale: str


def anchors_explain(
    *,
    instance: dict[str, float],
    predictor: _Predictor,
    feature_perturbation_scale: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
    target_precision: float = 0.95,
    max_anchor_size: int = 4,
    n_samples: int = 500,
    seed: int = 21,
) -> AnchorReport:
    """Greedy minimum-anchor search.

    For a binary classifier wrapped as ``predictor(features) -> p``
    + a threshold ``decision_threshold``, find the smallest set of
    features ``A subset features`` whose presence (i.e. fixing them
    at their instance values while perturbing the rest) keeps the
    prediction's class equal to the instance's class with
    ``precision >= target_precision``.
    """
    if not instance:
        raise ValueError("instance cannot be empty")
    rng = random.Random(seed)
    feature_names = list(instance.keys())
    perturb = (
        feature_perturbation_scale
        if feature_perturbation_scale is not None
        else {f: 1.0 for f in feature_names}
    )
    instance_p = float(predictor(instance))
    instance_class = int(instance_p >= decision_threshold)

    def _precision(anchor: list[str]) -> tuple[float, int]:
        n_match = 0
        for _ in range(n_samples):
            cand = {}
            for f in feature_names:
                if f in anchor:
                    cand[f] = float(instance[f])
                else:
                    scale = max(1e-6, perturb.get(f, 1.0))
                    cand[f] = float(instance[f]) + rng.gauss(0, scale)
            p = float(predictor(cand))
            if int(p >= decision_threshold) == instance_class:
                n_match += 1
        return n_match / n_samples, n_samples

    # Greedy search: start from empty + expand by best-precision feature
    anchor: list[str] = []
    current_prec = 0.0
    n_eval = 0
    while len(anchor) < max_anchor_size:
        best_f: str | None = None
        best_prec = current_prec
        for f in feature_names:
            if f in anchor:
                continue
            cand = anchor + [f]
            prec, n = _precision(cand)
            n_eval += n
            if prec > best_prec:
                best_prec = prec
                best_f = f
        if best_f is None:
            break
        anchor.append(best_f)
        current_prec = best_prec
        if current_prec >= target_precision:
            break
    coverage = (
        1.0 / (2 ** len(anchor)) if anchor else 1.0
    )
    return AnchorReport(
        instance={k: round(v, 6) for k, v in instance.items()},
        instance_prediction=instance_class,
        anchor_features=anchor,
        precision=round(current_prec, 6),
        coverage=round(coverage, 6),
        n_samples_evaluated=n_eval,
        rationale=(
            f"Greedy anchor of size {len(anchor)} reached precision "
            f"{current_prec:.3f} on {n_eval} perturbation samples."
        ),
    )
