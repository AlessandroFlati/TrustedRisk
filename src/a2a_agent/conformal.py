"""Phase 10.9 -- Split conformal prediction with marginal coverage.

Implements the Vovk/Shafer split conformal procedure: fit the model
on a training fold, compute non-conformity scores on a held-out
calibration fold, and use the empirical (n+1)(1-α)/n quantile as the
threshold for downstream prediction sets / intervals.

For TrustedRisk's calibrated readmission risk:

  - **regression style** (abs_residual): score = |y - p|; the
    conformal interval is `[p - q, p + q]` clipped to [0, 1].
  - **binary classification** (binary_one_minus_p): score = 1 - p_y
    where p_y is the predicted probability of the observed class. The
    conformal prediction set is `{c : 1 - p_c ≤ q}`. An empty set
    indicates abstention.

Marginal coverage:
        Pr[ y ∈ C(x) ] ≥ 1 - α
where the probability is over a fresh exchangeable (x, y) draw from
the same population as the calibration fold.

References:
    - Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a
      Random World. 2nd ed. Springer (2022).
    - Angelopoulos AN, Bates S. A Gentle Introduction to Conformal
      Prediction and Distribution-Free Uncertainty Quantification.
      arXiv:2107.07511 (2022).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from shared.schemas import (
    ConformalCalibrationReport,
    ConformalPredictionInterval,
    ConformalPredictionSet,
)


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────


def _empirical_quantile(scores: Sequence[float], q: float) -> float:
    """The (1-α) finite-sample-corrected quantile used in split
    conformal: take the ceil((n+1)(1-α))-th order statistic."""
    if not scores:
        raise ValueError("scores must be non-empty")
    n = len(scores)
    rank = math.ceil((n + 1) * q)
    rank = max(1, min(n, rank))
    sorted_scores = sorted(scores)
    return float(sorted_scores[rank - 1])


def _abs_residual_score(p: float, y: float) -> float:
    return abs(y - p)


def _binary_one_minus_p_score(p_y: float) -> float:
    """For a class-1 outcome y=1 we use 1 - p; for y=0 we use p."""
    return 1.0 - p_y


def fit_split_conformal(
    calibration_predictions: Sequence[float],
    calibration_outcomes: Sequence[float],
    *,
    target_coverage: float = 0.90,
    score_function_id: str = "abs_residual",
) -> ConformalCalibrationReport:
    """Compute the conformal threshold on a held-out calibration fold.

    Args:
        calibration_predictions: model predictions on the calibration
            fold (probabilities in [0, 1]).
        calibration_outcomes: ground-truth outcomes on the same fold.
            For regression: any float; for binary: 0 or 1.
        target_coverage: 1 - α ∈ (0, 1).
        score_function_id: 'abs_residual' or 'binary_one_minus_p'.

    Returns:
        ConformalCalibrationReport.
    """
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be in (0, 1)")
    if len(calibration_predictions) != len(calibration_outcomes):
        raise ValueError(
            "calibration_predictions and calibration_outcomes must "
            "have equal length"
        )
    if not calibration_predictions:
        raise ValueError("calibration set must be non-empty")
    if score_function_id not in ("abs_residual", "binary_one_minus_p"):
        raise ValueError(
            "score_function_id must be 'abs_residual' or "
            "'binary_one_minus_p'"
        )

    if score_function_id == "abs_residual":
        scores = [
            _abs_residual_score(p, y)
            for p, y in zip(calibration_predictions, calibration_outcomes)
        ]
    else:
        scores = []
        for p, y in zip(calibration_predictions, calibration_outcomes):
            p_y = p if int(y) == 1 else (1.0 - p)
            scores.append(_binary_one_minus_p_score(p_y))

    q = _empirical_quantile(scores, target_coverage)

    # Empirical coverage on the calibration fold (sanity check)
    empirical = (
        sum(1 for s in scores if s <= q) / len(scores)
    )

    rationale = (
        f"Split conformal calibration on n={len(scores)} held-out "
        f"point(s); target coverage {target_coverage:.3f}; "
        f"empirical coverage {empirical:.3f}; "
        f"quantile threshold q = {q:.4f} "
        f"(score function = {score_function_id})."
    )

    return ConformalCalibrationReport(
        n_calibration=len(scores),
        target_coverage=target_coverage,
        empirical_coverage=round(empirical, 4),
        quantile_threshold=round(q, 4),
        score_function_id=score_function_id,            # type: ignore[arg-type]
        rationale=rationale,
        references=[
            "Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a "
            "Random World. 2nd ed. Springer (2022).",
            "Angelopoulos AN, Bates S. arXiv:2107.07511 (2022).",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────


def predict_interval(
    point_estimate: float,
    calibration: ConformalCalibrationReport,
) -> ConformalPredictionInterval:
    """Build a regression-style conformal interval `[p-q, p+q]`,
    clipped to [0, 1]."""
    if calibration.score_function_id != "abs_residual":
        raise ValueError(
            "predict_interval requires a calibration fitted with "
            "score_function_id='abs_residual'"
        )
    q = calibration.quantile_threshold
    lower = max(0.0, point_estimate - q)
    upper = min(1.0, point_estimate + q)
    return ConformalPredictionInterval(
        point_estimate=round(point_estimate, 4),
        lower_bound=round(lower, 4),
        upper_bound=round(upper, 4),
        target_coverage=calibration.target_coverage,
        quantile_threshold_used=q,
    )


def predict_set(
    point_estimate_probability: float,
    calibration: ConformalCalibrationReport,
) -> ConformalPredictionSet:
    """Build a binary conformal prediction set under the
    `binary_one_minus_p` score.

    A class c is included in the set when `1 - p_c <= q`. For binary
    {0, 1} this means:
        include 1 when (1 - p) <= q   i.e. p >= 1 - q
        include 0 when p     <= q     i.e. p <= q
    """
    if calibration.score_function_id != "binary_one_minus_p":
        raise ValueError(
            "predict_set requires a calibration fitted with "
            "score_function_id='binary_one_minus_p'"
        )
    if not 0.0 <= point_estimate_probability <= 1.0:
        raise ValueError(
            "point_estimate_probability must be in [0, 1]"
        )
    q = calibration.quantile_threshold
    p = float(point_estimate_probability)
    pred_set: list[int] = []
    if (1.0 - p) <= q:
        pred_set.append(1)
    if p <= q:
        pred_set.append(0)
    return ConformalPredictionSet(
        point_estimate_probability=round(p, 4),
        prediction_set=pred_set,                          # type: ignore[arg-type]
        target_coverage=calibration.target_coverage,
        quantile_threshold_used=q,
    )
