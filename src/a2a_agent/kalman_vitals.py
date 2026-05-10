"""Phase 17.AF - Kalman filter for vitals time-series.

Pure-Python implementation of:

  - **1-D Kalman filter** for tracking a single vital sign (HR, SpO2,
    BP, etc.) under a constant-velocity model: the underlying state is
    (value, slope), measurements are the observed value, and the
    process + measurement noise are configurable.

  - **Multi-D Kalman filter** for tracking multiple correlated vitals
    in a single state vector. State = [HR, SpO2, BP, RR, T]; we use a
    diagonal process model + diagonal measurement model so the matrix
    operations stay implementable without NumPy.

  - **Change-point detector**: flag any timestep where the post-update
    residual exceeds 2 sigma of the predicted measurement. This
    surfaces clinically important deteriorations earlier than a
    sliding-mean threshold.

Pure-Python, deterministic, stdlib-only. No NumPy / SciPy.
"""

from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# 1-D constant-velocity Kalman filter
# ─────────────────────────────────────────────────────────────────────


class KalmanStep(BaseModel):
    t: float
    measurement: float
    filtered_value: float
    filtered_velocity: float
    posterior_variance: float
    residual: float
    residual_z: float
    is_change_point: bool


class Kalman1DReport(BaseModel):
    n: int
    process_noise_q: float
    measurement_noise_r: float
    change_point_threshold_z: float
    steps: list[KalmanStep]
    n_change_points: int
    rationale: str


def _kalman_1d_step(
    *,
    state: tuple[float, float],
    cov: tuple[tuple[float, float], tuple[float, float]],
    dt: float,
    z: float,
    q: float,
    r: float,
) -> tuple[
    tuple[float, float],
    tuple[tuple[float, float], tuple[float, float]],
    float, float,
]:
    """One predict + update step of a 1-D constant-velocity Kalman.

    State x = (value, velocity).
    F = [[1, dt], [0, 1]] (constant-velocity transition)
    H = [1, 0] (we observe the value)
    """
    x, v = state
    p00, p01 = cov[0]
    p10, p11 = cov[1]
    # Predict
    x_pred = x + v * dt
    v_pred = v
    p00_pred = p00 + dt * (p10 + p01) + dt * dt * p11 + q
    p01_pred = p01 + dt * p11
    p10_pred = p10 + dt * p11
    p11_pred = p11 + q
    # Innovation
    s = p00_pred + r
    y = z - x_pred
    k0 = p00_pred / s
    k1 = p10_pred / s
    # Update
    x_upd = x_pred + k0 * y
    v_upd = v_pred + k1 * y
    p00_upd = (1.0 - k0) * p00_pred
    p01_upd = (1.0 - k0) * p01_pred
    p10_upd = p10_pred - k1 * p00_pred
    p11_upd = p11_pred - k1 * p01_pred
    return (
        (x_upd, v_upd),
        ((p00_upd, p01_upd), (p10_upd, p11_upd)),
        y, s,
    )


def kalman_filter_1d(
    *,
    timestamps: list[float],
    measurements: list[float],
    process_noise_q: float = 0.5,
    measurement_noise_r: float = 4.0,
    change_point_threshold_z: float = 2.0,
    initial_value: float | None = None,
    initial_velocity: float = 0.0,
    initial_variance: float = 100.0,
) -> Kalman1DReport:
    """Run the 1-D Kalman filter over a vital-sign time series.

    Args:
        timestamps: list of timestamps in any monotonic unit (e.g.
            seconds since admission, hours, etc.).
        measurements: aligned list of vital measurements.
        process_noise_q: tunes how much the underlying state can
            drift between updates (higher q -> filter trusts new
            measurements more).
        measurement_noise_r: variance of the measurement error
            (higher r -> filter smooths more).
        change_point_threshold_z: flag |residual / sqrt(S)| above
            this z-score as a change-point.
    """
    if len(timestamps) != len(measurements):
        raise ValueError(
            "timestamps and measurements must align")
    if not timestamps:
        raise ValueError("input series cannot be empty")
    state = (
        measurements[0] if initial_value is None else initial_value,
        initial_velocity,
    )
    cov: tuple[tuple[float, float], tuple[float, float]] = (
        (initial_variance, 0.0),
        (0.0, initial_variance),
    )
    steps: list[KalmanStep] = []
    n_cp = 0
    prev_t = timestamps[0]
    for i, (t, z) in enumerate(zip(timestamps, measurements)):
        dt = max(1e-6, t - prev_t) if i > 0 else 1.0
        state, cov, residual, s = _kalman_1d_step(
            state=state, cov=cov, dt=dt, z=z,
            q=process_noise_q, r=measurement_noise_r,
        )
        rz = (residual / math.sqrt(s)) if s > 0 else 0.0
        is_cp = (
            i > 0
            and abs(rz) >= change_point_threshold_z
        )
        if is_cp:
            n_cp += 1
        steps.append(KalmanStep(
            t=t, measurement=z,
            filtered_value=round(state[0], 6),
            filtered_velocity=round(state[1], 6),
            posterior_variance=round(cov[0][0], 6),
            residual=round(residual, 6),
            residual_z=round(rz, 4),
            is_change_point=is_cp,
        ))
        prev_t = t
    return Kalman1DReport(
        n=len(steps),
        process_noise_q=process_noise_q,
        measurement_noise_r=measurement_noise_r,
        change_point_threshold_z=change_point_threshold_z,
        steps=steps, n_change_points=n_cp,
        rationale=(
            f"1-D Kalman filter over {len(steps)} measurements; "
            f"{n_cp} change-point(s) flagged at z >= "
            f"{change_point_threshold_z}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Multi-dimensional Kalman filter (diagonal model)
# ─────────────────────────────────────────────────────────────────────


class MultiKalmanStep(BaseModel):
    t: float
    measurements: dict[str, float]
    filtered: dict[str, float]
    posterior_variance: dict[str, float]
    residual_z: dict[str, float]
    is_change_point: bool


class MultiKalmanReport(BaseModel):
    n: int
    feature_names: list[str]
    process_noise_q: float
    measurement_noise_r: float
    change_point_threshold_z: float
    steps: list[MultiKalmanStep]
    n_change_points: int
    rationale: str


def kalman_filter_multivariate(
    *,
    feature_names: list[str],
    timestamps: list[float],
    measurements: list[dict[str, float]],
    process_noise_q: float = 0.5,
    measurement_noise_r: float = 4.0,
    change_point_threshold_z: float = 2.0,
) -> MultiKalmanReport:
    """Multi-dimensional Kalman with a diagonal process + measurement
    model. Equivalent to running an independent 1-D Kalman per feature
    + flagging a change-point if any feature exceeds the z-score
    threshold.
    """
    if not feature_names:
        raise ValueError("feature_names cannot be empty")
    if len(timestamps) != len(measurements):
        raise ValueError(
            "timestamps and measurements must align")
    if not timestamps:
        raise ValueError("input series cannot be empty")

    series = {name: [] for name in feature_names}
    for m in measurements:
        for name in feature_names:
            if name not in m:
                raise ValueError(
                    f"measurement missing feature `{name}`")
            series[name].append(float(m[name]))

    per_feature_reports: dict[str, Kalman1DReport] = {}
    for name in feature_names:
        per_feature_reports[name] = kalman_filter_1d(
            timestamps=list(timestamps),
            measurements=series[name],
            process_noise_q=process_noise_q,
            measurement_noise_r=measurement_noise_r,
            change_point_threshold_z=change_point_threshold_z,
        )

    steps: list[MultiKalmanStep] = []
    n_cp = 0
    for i, t in enumerate(timestamps):
        m = measurements[i]
        filtered = {n: per_feature_reports[n].steps[i].filtered_value
                    for n in feature_names}
        var = {n: per_feature_reports[n].steps[i].posterior_variance
               for n in feature_names}
        rz = {n: per_feature_reports[n].steps[i].residual_z
              for n in feature_names}
        is_cp = any(
            per_feature_reports[n].steps[i].is_change_point
            for n in feature_names
        )
        if is_cp:
            n_cp += 1
        steps.append(MultiKalmanStep(
            t=t, measurements={n: float(m[n]) for n in feature_names},
            filtered=filtered, posterior_variance=var, residual_z=rz,
            is_change_point=is_cp,
        ))
    return MultiKalmanReport(
        n=len(steps), feature_names=list(feature_names),
        process_noise_q=process_noise_q,
        measurement_noise_r=measurement_noise_r,
        change_point_threshold_z=change_point_threshold_z,
        steps=steps, n_change_points=n_cp,
        rationale=(
            f"Multi-D Kalman over {len(feature_names)} features and "
            f"{len(steps)} timesteps; {n_cp} change-point(s) flagged."
        ),
    )
