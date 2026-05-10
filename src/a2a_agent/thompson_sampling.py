"""Phase 17.AP - Multi-armed bandits + Thompson sampling.

Beta-Binomial Thompson sampling over a discrete set of treatment
arms. Each arm carries a Beta(alpha, beta) posterior on its reward
probability ``p_arm``. At each round the algorithm:

  1. Samples ``p_hat_arm ~ Beta(alpha_arm, beta_arm)`` for every arm.
  2. Picks the arm with the highest sampled reward probability.
  3. Observes a Bernoulli reward.
  4. Updates the chosen arm's Beta posterior.

Provides:
  - ``ThompsonBandit`` - stateful bandit object.
  - ``simulate_bandit_run`` - end-to-end run with synthetic rewards.
  - ``regret_bound_o_sqrt_n_log_k`` - the canonical Bayesian regret
    bound (Russo + Van Roy 2016) for instrumentation.

Pure-Python deterministic. Sampling from Beta uses the gamma-ratio
parameterisation: ``X ~ Beta(a, b)`` where
``X = G_a / (G_a + G_b)`` and ``G_x ~ Gamma(x, 1)``.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────


def _sample_gamma(alpha: float, rng: random.Random) -> float:
    """Marsaglia & Tsang 2000 - Gamma sampler for alpha >= 1.
    For alpha < 1 use the boost trick: G(alpha) = G(alpha+1) * U^(1/alpha).
    """
    if alpha < 1.0:
        u = rng.random()
        return _sample_gamma(alpha + 1.0, rng) * (u ** (1.0 / alpha))
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0, 1)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if (
            u < 1.0 - 0.0331 * (x ** 4)
            or math.log(u) < 0.5 * (x ** 2) + d * (1.0 - v + math.log(v))
        ):
            return d * v


def _sample_beta(alpha: float, beta: float,
                 rng: random.Random) -> float:
    a = _sample_gamma(alpha, rng)
    b = _sample_gamma(beta, rng)
    return a / (a + b) if (a + b) > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────
# Stateful bandit
# ─────────────────────────────────────────────────────────────────────


class ArmPosterior(BaseModel):
    arm_id: str
    alpha: float = Field(gt=0.0)
    beta: float = Field(gt=0.0)
    n_pulls: int = Field(ge=0)
    n_successes: int = Field(ge=0)

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ThompsonBandit:
    """Stateful Beta-Binomial Thompson-sampling bandit."""

    def __init__(
        self, *, arm_ids: list[str],
        prior_alpha: float = 1.0, prior_beta: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        if not arm_ids:
            raise ValueError("arm_ids cannot be empty")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("priors must be positive")
        self._rng = rng if rng is not None else random.Random()
        self._arms: dict[str, ArmPosterior] = {
            a: ArmPosterior(
                arm_id=a, alpha=prior_alpha, beta=prior_beta,
                n_pulls=0, n_successes=0,
            )
            for a in arm_ids
        }

    @property
    def arms(self) -> dict[str, ArmPosterior]:
        return {k: v.model_copy() for k, v in self._arms.items()}

    def select_arm(self) -> str:
        """Sample one reward probability per arm + pick the max."""
        best_arm: str | None = None
        best_sample = -1.0
        for arm_id, post in self._arms.items():
            sample = _sample_beta(post.alpha, post.beta, self._rng)
            if sample > best_sample:
                best_sample = sample
                best_arm = arm_id
        assert best_arm is not None
        return best_arm

    def update(self, arm_id: str, reward: int) -> None:
        if arm_id not in self._arms:
            raise ValueError(f"unknown arm {arm_id!r}")
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        post = self._arms[arm_id]
        post.alpha += reward
        post.beta += (1 - reward)
        post.n_pulls += 1
        post.n_successes += reward


# ─────────────────────────────────────────────────────────────────────
# Synthetic-environment simulation
# ─────────────────────────────────────────────────────────────────────


class BanditTrace(BaseModel):
    n_rounds: int
    arm_ids: list[str]
    true_reward_probabilities: dict[str, float]
    pulls_per_arm: dict[str, int]
    successes_per_arm: dict[str, int]
    cumulative_reward: int
    cumulative_regret: float
    best_arm: str
    posteriors: dict[str, ArmPosterior]
    rationale: str


def simulate_bandit_run(
    *,
    arm_ids: list[str],
    true_reward_probabilities: dict[str, float],
    n_rounds: int = 500,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    seed: int = 42,
) -> BanditTrace:
    """Run ``n_rounds`` of Thompson sampling against a synthetic
    Bernoulli environment with the supplied per-arm reward
    probabilities. Reports cumulative reward + regret."""
    if n_rounds <= 0:
        raise ValueError("n_rounds must be > 0")
    if set(arm_ids) != set(true_reward_probabilities.keys()):
        raise ValueError(
            "arm_ids must match true_reward_probabilities keys")
    rng = random.Random(seed)
    bandit = ThompsonBandit(
        arm_ids=arm_ids,
        prior_alpha=prior_alpha, prior_beta=prior_beta,
        rng=rng,
    )
    pulls = {a: 0 for a in arm_ids}
    successes = {a: 0 for a in arm_ids}
    cumulative_reward = 0
    cumulative_regret = 0.0
    best_arm = max(
        true_reward_probabilities, key=true_reward_probabilities.get
    )
    p_star = true_reward_probabilities[best_arm]
    for _ in range(n_rounds):
        arm = bandit.select_arm()
        p_arm = true_reward_probabilities[arm]
        reward = 1 if rng.random() < p_arm else 0
        bandit.update(arm, reward)
        pulls[arm] += 1
        successes[arm] += reward
        cumulative_reward += reward
        cumulative_regret += p_star - p_arm
    posteriors = bandit.arms
    return BanditTrace(
        n_rounds=n_rounds, arm_ids=list(arm_ids),
        true_reward_probabilities=dict(true_reward_probabilities),
        pulls_per_arm=pulls, successes_per_arm=successes,
        cumulative_reward=cumulative_reward,
        cumulative_regret=round(cumulative_regret, 4),
        best_arm=best_arm,
        posteriors=posteriors,
        rationale=(
            f"Thompson sampling over {len(arm_ids)} arms x "
            f"{n_rounds} rounds; cumulative reward "
            f"{cumulative_reward}, regret "
            f"{cumulative_regret:.2f} vs best arm `{best_arm}` "
            f"(p* = {p_star:.3f})."
        ),
    )


def regret_bound_o_sqrt_n_log_k(*, n_rounds: int, n_arms: int
                                ) -> float:
    """Russo & Van Roy 2016 Bayesian regret upper bound for Thompson
    sampling: O(sqrt(N log K)). Returns the constant-free upper
    bound for instrumentation."""
    if n_rounds <= 0 or n_arms <= 0:
        raise ValueError("n_rounds + n_arms must be > 0")
    return math.sqrt(n_rounds * math.log(n_arms + 1))
