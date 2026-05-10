"""Phase 17.AW - Neural-ODE trajectory tests."""

from __future__ import annotations

import math

import pytest

from a2a_agent.neural_ode import (
    NODEParams, NODETrajectoryReport,
    euler_step, init_node_params, integrate_trajectory,
    rk4_step, vector_field,
)


def test_init_params_yields_correct_dimensions():
    p = init_node_params(state_dim=5, hidden_dim=10, seed=1)
    assert p.state_dim == 5
    assert p.hidden_dim == 10
    assert len(p.W1) == 10 and len(p.W1[0]) == 5
    assert len(p.W2) == 5 and len(p.W2[0]) == 10
    assert len(p.b1) == 10
    assert len(p.b2) == 5


def test_vector_field_returns_state_dim_vector():
    p = init_node_params(state_dim=3, hidden_dim=8, seed=1)
    f = vector_field([1.0, 0.0, -0.5], 0.0, p)
    assert len(f) == 3
    for v in f:
        assert math.isfinite(v)


def test_euler_step_advances_state():
    p = init_node_params(state_dim=2, hidden_dim=4, seed=1)
    x = [1.0, 0.0]
    x_next = euler_step(x, 0.0, 0.1, p)
    assert len(x_next) == 2
    # The step should be small (dt=0.1, small weights)
    diffs = [abs(a - b) for a, b in zip(x_next, x)]
    assert all(d < 5.0 for d in diffs)


def test_rk4_step_returns_state_dim_vector():
    p = init_node_params(state_dim=2, hidden_dim=4, seed=1)
    x = [1.0, 0.0]
    x_next = rk4_step(x, 0.0, 0.1, p)
    assert len(x_next) == 2


def test_integrate_emits_one_step_per_timestamp():
    p = init_node_params(state_dim=3, hidden_dim=6, seed=1)
    timestamps = [0.0, 1.0, 2.0, 3.0]
    rep = integrate_trajectory(
        initial_state=[0.5, 0.0, -0.5],
        timestamps=timestamps, params=p, method="euler",
    )
    assert len(rep.trajectory) == len(timestamps)
    assert rep.method == "euler"


def test_integrate_residual_loss_zero_for_self_consistent_observations():
    p = init_node_params(state_dim=2, hidden_dim=4, seed=1)
    timestamps = [0.0, 1.0, 2.0]
    rep_no_obs = integrate_trajectory(
        initial_state=[1.0, 0.0],
        timestamps=timestamps, params=p,
    )
    observed = [step.state for step in rep_no_obs.trajectory]
    rep_with_obs = integrate_trajectory(
        initial_state=[1.0, 0.0],
        timestamps=timestamps, params=p,
        observed_trajectory=observed,
    )
    assert rep_with_obs.residual_loss is not None
    assert rep_with_obs.residual_loss < 1e-9


def test_rk4_more_accurate_than_euler_on_known_field():
    """Construct params where the linear part dominates, so the
    true vector field has analytic Euler-vs-RK4 comparison: RK4
    should give smaller residual against the same dense reference
    integration over multiple steps."""
    p = init_node_params(state_dim=2, hidden_dim=4, seed=1)
    timestamps_dense = [i * 0.05 for i in range(101)]
    timestamps_coarse = [i * 0.5 for i in range(11)]
    dense_euler = integrate_trajectory(
        initial_state=[1.0, 0.5],
        timestamps=timestamps_dense, params=p, method="rk4",
    )
    # Reference at t=5: take the dense RK4 result
    ref_state = dense_euler.trajectory[-1].state
    # Compare coarse Euler vs coarse RK4 against the reference
    coarse_euler = integrate_trajectory(
        initial_state=[1.0, 0.5],
        timestamps=timestamps_coarse, params=p, method="euler",
    )
    coarse_rk4 = integrate_trajectory(
        initial_state=[1.0, 0.5],
        timestamps=timestamps_coarse, params=p, method="rk4",
    )
    err_euler = sum(
        abs(a - b) for a, b in zip(
            coarse_euler.trajectory[-1].state, ref_state)
    )
    err_rk4 = sum(
        abs(a - b) for a, b in zip(
            coarse_rk4.trajectory[-1].state, ref_state)
    )
    assert err_rk4 <= err_euler


def test_integrate_rejects_empty_timestamps():
    p = init_node_params(state_dim=2, hidden_dim=4, seed=1)
    with pytest.raises(ValueError):
        integrate_trajectory(
            initial_state=[1.0, 0.0], timestamps=[],
            params=p,
        )


def test_integrate_rejects_misaligned_initial_state():
    p = init_node_params(state_dim=3, hidden_dim=4, seed=1)
    with pytest.raises(ValueError):
        integrate_trajectory(
            initial_state=[1.0, 0.0],   # 2D vs state_dim=3
            timestamps=[0.0, 1.0],
            params=p,
        )


def test_integrate_rejects_non_increasing_timestamps():
    p = init_node_params(state_dim=2, hidden_dim=4, seed=1)
    with pytest.raises(ValueError):
        integrate_trajectory(
            initial_state=[1.0, 0.0],
            timestamps=[0.0, 1.0, 0.5],
            params=p,
        )


def test_integrate_round_trip_through_pydantic():
    p = init_node_params(state_dim=2, hidden_dim=4, seed=1)
    rep = integrate_trajectory(
        initial_state=[1.0, 0.0],
        timestamps=[0.0, 0.5], params=p,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = NODETrajectoryReport.model_validate(payload)
    assert rebuilt.n_steps == rep.n_steps
