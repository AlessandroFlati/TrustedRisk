"""TIME-4 -- Bayesian state-space trajectory predictor.

Predicts the calibrated readmission probability over a 1-72h horizon.
Pure-numpy implementation: a discrete-time linear Gaussian state-space
model (predict step only -- no observation update inside this function;
the observation update is handled by `incremental_risk` for streaming
events). For each predicted hour we emit a (mean, CI95) tuple.

Model:
  state_t   = A * state_{t-1} + w_t,    w ~ N(0, Q)
  prob_t    = sigmoid(state_t)

State is a single scalar logit corresponding to the calibrated readmission
log-odds. Q is set to a fixed per-hour drift (configurable via
`drift_per_hour`). The prediction widens uncertainty linearly in
sqrt(horizon) -- a standard property of Brownian motion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from shared.schemas import (
    TrajectoryPoint,
    TrajectoryPrediction,
)


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def predict_trajectory(
    current_probability: float,
    horizon_hours: int = 72,
    drift_per_hour: float = 0.005,
    direction: str = "increasing",
) -> TrajectoryPrediction:
    """Predict the readmission-risk trajectory over a horizon.

    Args:
        current_probability: latest calibrated risk (0-1).
        horizon_hours: prediction window (1-720 hours).
        drift_per_hour: per-hour drift in logit units. Default 0.005 ≈
            slow upward drift consistent with average post-discharge
            decay of clinical stability.
        direction: "increasing" / "decreasing" / "neutral" -- flips the
            sign of the drift term.

    Returns:
        TrajectoryPrediction with per-hour points + expected peak.
    """
    if not (0.0 <= current_probability <= 1.0):
        raise ValueError(
            "current_probability must be in [0, 1].")
    if horizon_hours < 1 or horizon_hours > 720:
        raise ValueError("horizon_hours must be in [1, 720].")
    if drift_per_hour < 0:
        raise ValueError("drift_per_hour must be ≥ 0.")
    valid_dirs = {"increasing", "decreasing", "neutral"}
    if direction not in valid_dirs:
        raise ValueError(f"direction must be one of {sorted(valid_dirs)}.")

    sign = (1.0 if direction == "increasing"
            else (-1.0 if direction == "decreasing" else 0.0))
    base_logit = _logit(current_probability)

    # Per-hour variance grows linearly; std grows as sqrt(t)
    sigma_per_hour_sq = 0.04   # logit-space variance per hour
    points: list[TrajectoryPoint] = []
    peak_prob = current_probability
    peak_hour = 0
    for h in range(horizon_hours + 1):
        mean_logit = base_logit + sign * drift_per_hour * h
        std_logit = math.sqrt(sigma_per_hour_sq * max(1, h))
        # 95% CI in logit space
        lo_logit = mean_logit - 1.96 * std_logit
        hi_logit = mean_logit + 1.96 * std_logit
        mean_p = _sigmoid(mean_logit)
        lo_p = _sigmoid(lo_logit)
        hi_p = _sigmoid(hi_logit)
        points.append(TrajectoryPoint(
            hour_offset=h,
            probability_mean=round(mean_p, 4),
            probability_ci95_low=round(lo_p, 4),
            probability_ci95_high=round(hi_p, 4),
        ))
        if mean_p > peak_prob:
            peak_prob = mean_p
            peak_hour = h

    rationale = (
        f"Bayesian state-space prediction over {horizon_hours}h "
        f"(drift={drift_per_hour}/h, direction={direction}). "
        f"Peak {peak_prob:.3f} at +{peak_hour}h. "
        f"Initial probability {current_probability:.3f}."
    )

    return TrajectoryPrediction(
        horizon_hours=horizon_hours,
        points=points,
        expected_peak_probability=round(peak_prob, 4),
        expected_peak_hour=peak_hour,
        method="bayesian_state_space",
        rationale=rationale,
    )
