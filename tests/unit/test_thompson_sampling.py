"""Phase 17.AP - Thompson sampling tests."""

from __future__ import annotations

import random

import pytest

from a2a_agent.thompson_sampling import (
    ArmPosterior, BanditTrace, ThompsonBandit,
    _sample_beta, _sample_gamma,
    regret_bound_o_sqrt_n_log_k, simulate_bandit_run,
)


# ─────────────────────────────────────────────────────────────────────
# Sampling primitives
# ─────────────────────────────────────────────────────────────────────

def test_sample_beta_in_unit_interval():
    rng = random.Random(7)
    for _ in range(50):
        v = _sample_beta(2.0, 5.0, rng)
        assert 0.0 <= v <= 1.0


def test_sample_gamma_positive():
    rng = random.Random(7)
    for alpha in (0.5, 1.0, 5.0, 20.0):
        for _ in range(20):
            assert _sample_gamma(alpha, rng) > 0


def test_sample_beta_mean_close_to_alpha_over_alpha_plus_beta():
    rng = random.Random(11)
    samples = [_sample_beta(8.0, 2.0, rng) for _ in range(500)]
    avg = sum(samples) / len(samples)
    # E[Beta(8,2)] = 0.8
    assert abs(avg - 0.8) < 0.05


# ─────────────────────────────────────────────────────────────────────
# ThompsonBandit
# ─────────────────────────────────────────────────────────────────────

def test_bandit_rejects_empty_arms():
    with pytest.raises(ValueError):
        ThompsonBandit(arm_ids=[])


def test_bandit_rejects_non_positive_priors():
    with pytest.raises(ValueError):
        ThompsonBandit(arm_ids=["a"], prior_alpha=0.0)


def test_bandit_select_arm_returns_known_arm():
    bandit = ThompsonBandit(
        arm_ids=["a", "b", "c"], rng=random.Random(1))
    arm = bandit.select_arm()
    assert arm in {"a", "b", "c"}


def test_bandit_update_increments_alpha_for_reward_1():
    bandit = ThompsonBandit(
        arm_ids=["a"], prior_alpha=1.0, prior_beta=1.0,
    )
    bandit.update("a", reward=1)
    post = bandit.arms["a"]
    assert post.alpha == 2.0
    assert post.beta == 1.0
    assert post.n_pulls == 1
    assert post.n_successes == 1


def test_bandit_update_increments_beta_for_reward_0():
    bandit = ThompsonBandit(
        arm_ids=["a"], prior_alpha=1.0, prior_beta=1.0,
    )
    bandit.update("a", reward=0)
    post = bandit.arms["a"]
    assert post.alpha == 1.0
    assert post.beta == 2.0


def test_bandit_update_rejects_unknown_arm():
    bandit = ThompsonBandit(arm_ids=["a"])
    with pytest.raises(ValueError):
        bandit.update("not_an_arm", reward=1)


def test_bandit_update_rejects_invalid_reward():
    bandit = ThompsonBandit(arm_ids=["a"])
    with pytest.raises(ValueError):
        bandit.update("a", reward=2)


# ─────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────

def test_simulation_pulls_best_arm_most_often_eventually():
    """With p_a=0.7, p_b=0.3 over 500 rounds, the bandit must pull
    the better arm a clear majority of the time."""
    trace = simulate_bandit_run(
        arm_ids=["a", "b"],
        true_reward_probabilities={"a": 0.7, "b": 0.3},
        n_rounds=500, seed=17,
    )
    assert trace.pulls_per_arm["a"] > trace.pulls_per_arm["b"]
    assert trace.best_arm == "a"


def test_simulation_cumulative_reward_bounded():
    trace = simulate_bandit_run(
        arm_ids=["a", "b"],
        true_reward_probabilities={"a": 0.5, "b": 0.5},
        n_rounds=200, seed=7,
    )
    assert 0 <= trace.cumulative_reward <= 200


def test_simulation_regret_non_negative():
    trace = simulate_bandit_run(
        arm_ids=["a", "b"],
        true_reward_probabilities={"a": 0.7, "b": 0.3},
        n_rounds=300, seed=3,
    )
    assert trace.cumulative_regret >= 0


def test_simulation_rejects_arm_id_mismatch():
    with pytest.raises(ValueError):
        simulate_bandit_run(
            arm_ids=["a", "b"],
            true_reward_probabilities={"a": 0.5},
            n_rounds=100,
        )


def test_simulation_rejects_zero_rounds():
    with pytest.raises(ValueError):
        simulate_bandit_run(
            arm_ids=["a"],
            true_reward_probabilities={"a": 0.5},
            n_rounds=0,
        )


def test_simulation_round_trip_through_pydantic():
    trace = simulate_bandit_run(
        arm_ids=["a", "b"],
        true_reward_probabilities={"a": 0.5, "b": 0.5},
        n_rounds=50, seed=1,
    )
    payload = trace.model_dump(mode="json")
    rebuilt = BanditTrace.model_validate(payload)
    assert rebuilt.n_rounds == trace.n_rounds


# ─────────────────────────────────────────────────────────────────────
# Regret bound
# ─────────────────────────────────────────────────────────────────────

def test_regret_bound_grows_sublinearly():
    n_100 = regret_bound_o_sqrt_n_log_k(n_rounds=100, n_arms=3)
    n_400 = regret_bound_o_sqrt_n_log_k(n_rounds=400, n_arms=3)
    # sqrt(N) growth -> 4x N -> 2x bound
    assert abs(n_400 / n_100 - 2.0) < 0.1


def test_regret_bound_rejects_zero():
    with pytest.raises(ValueError):
        regret_bound_o_sqrt_n_log_k(n_rounds=0, n_arms=2)
