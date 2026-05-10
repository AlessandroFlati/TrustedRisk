"""Phase 17.AW - Neural-ODE-style trajectory model.

Continuous-time patient-state evolution model:

    dx/dt = f_theta(x, t)

integrated forward via fixed-step Euler (and the more accurate
classical RK4 as an option). The vector field ``f_theta`` is a
small linear+ReLU network:

    f_theta(x, t) = W_2 * relu(W_1 * x + b_1) + b_2

and is evaluated entirely in pure Python (no autograd, no NumPy).
The intent is to model a 5-dim patient state vector
(HR, SpO2, MAP, RR, T) over a 24-hour window with hourly
observations + return the integrated trajectory + a residual loss
against ground-truth observations.

References:
- Chen et al. 2018 "Neural Ordinary Differential Equations" (NeurIPS).
- Rubanova et al. 2019 "Latent ODEs for Irregularly-Sampled Time Series".
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Literal

from pydantic import BaseModel, Field


_Method = Literal["euler", "rk4"]


# ─────────────────────────────────────────────────────────────────────
# Tiny linear+ReLU network
# ─────────────────────────────────────────────────────────────────────


class NODEParams(BaseModel):
    state_dim: int = Field(ge=1)
    hidden_dim: int = Field(ge=1)
    W1: list[list[float]]
    b1: list[float]
    W2: list[list[float]]
    b2: list[float]


def _matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(w * v for w, v in zip(row, x)) for row in W]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


def _relu(a: list[float]) -> list[float]:
    return [max(0.0, x) for x in a]


def init_node_params(
    *, state_dim: int, hidden_dim: int, seed: int = 7,
) -> NODEParams:
    """Xavier-style initialisation."""
    rng = random.Random(seed)
    bound1 = math.sqrt(6.0 / (state_dim + hidden_dim))
    bound2 = math.sqrt(6.0 / (hidden_dim + state_dim))
    W1 = [
        [rng.uniform(-bound1, bound1) for _ in range(state_dim)]
        for _ in range(hidden_dim)
    ]
    W2 = [
        [rng.uniform(-bound2, bound2) for _ in range(hidden_dim)]
        for _ in range(state_dim)
    ]
    b1 = [0.0] * hidden_dim
    b2 = [0.0] * state_dim
    return NODEParams(
        state_dim=state_dim, hidden_dim=hidden_dim,
        W1=W1, b1=b1, W2=W2, b2=b2,
    )


def vector_field(
    x: list[float], t: float, params: NODEParams,
) -> list[float]:
    """f_theta(x, t) = W2 * relu(W1 x + b1) + b2.

    Time is unused in this minimal autonomous variant; pass it for
    API symmetry with non-autonomous extensions."""
    h = _add(_matvec(params.W1, x), params.b1)
    h = _relu(h)
    return _add(_matvec(params.W2, h), params.b2)


# ─────────────────────────────────────────────────────────────────────
# Integrators
# ─────────────────────────────────────────────────────────────────────


def euler_step(
    x: list[float], t: float, dt: float, params: NODEParams,
) -> list[float]:
    return _add(x, _scale(vector_field(x, t, params), dt))


def rk4_step(
    x: list[float], t: float, dt: float, params: NODEParams,
) -> list[float]:
    k1 = vector_field(x, t, params)
    k2 = vector_field(_add(x, _scale(k1, dt / 2)), t + dt / 2, params)
    k3 = vector_field(_add(x, _scale(k2, dt / 2)), t + dt / 2, params)
    k4 = vector_field(_add(x, _scale(k3, dt)), t + dt, params)
    inc = [
        (a + 2 * b + 2 * c + d) / 6.0
        for a, b, c, d in zip(k1, k2, k3, k4)
    ]
    return _add(x, _scale(inc, dt))


# ─────────────────────────────────────────────────────────────────────
# Trajectory + report
# ─────────────────────────────────────────────────────────────────────


class TrajectoryStep(BaseModel):
    t: float
    state: list[float]


class NODETrajectoryReport(BaseModel):
    method: _Method
    n_steps: int = Field(ge=0)
    state_dim: int = Field(ge=1)
    feature_names: list[str]
    trajectory: list[TrajectoryStep]
    residual_loss: float | None = None
    rationale: str


def integrate_trajectory(
    *,
    initial_state: list[float],
    timestamps: list[float],
    params: NODEParams,
    method: _Method = "euler",
    feature_names: list[str] | None = None,
    observed_trajectory: list[list[float]] | None = None,
) -> NODETrajectoryReport:
    """Integrate the state from t=timestamps[0] through the timestamps
    using the chosen method. If ``observed_trajectory`` is supplied
    we also report the L2 residual loss vs the integrated path."""
    if not timestamps:
        raise ValueError("timestamps cannot be empty")
    if len(initial_state) != params.state_dim:
        raise ValueError(
            "initial_state length must match params.state_dim")
    if observed_trajectory is not None and (
        len(observed_trajectory) != len(timestamps)
    ):
        raise ValueError(
            "observed_trajectory must align with timestamps")
    feature_names = (
        feature_names or [f"x{i}" for i in range(params.state_dim)]
    )
    if len(feature_names) != params.state_dim:
        raise ValueError(
            "feature_names must align with state_dim")

    step_fn = euler_step if method == "euler" else rk4_step
    trajectory: list[TrajectoryStep] = []
    state = list(initial_state)
    prev_t = timestamps[0]
    trajectory.append(TrajectoryStep(t=prev_t, state=list(state)))
    for i in range(1, len(timestamps)):
        t = timestamps[i]
        dt = t - prev_t
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing")
        state = step_fn(state, prev_t, dt, params)
        trajectory.append(TrajectoryStep(t=t, state=list(state)))
        prev_t = t

    residual: float | None = None
    if observed_trajectory is not None:
        sq = 0.0
        for step, obs in zip(trajectory, observed_trajectory):
            for a, b in zip(step.state, obs):
                sq += (a - b) ** 2
        residual = math.sqrt(sq / len(trajectory))

    return NODETrajectoryReport(
        method=method,
        n_steps=len(trajectory),
        state_dim=params.state_dim,
        feature_names=list(feature_names),
        trajectory=trajectory,
        residual_loss=(round(residual, 6)
                       if residual is not None else None),
        rationale=(
            f"NeuralODE integration via {method} over "
            f"{len(trajectory)} timesteps."
        ),
    )
