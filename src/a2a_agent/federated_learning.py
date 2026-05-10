"""Phase 17.AN - Federated Learning simulator.

Simulates McMahan et al. 2017 (FedAvg) on the W1 spec_002 5-bin
Beta-Binomial readmission model. The premise:

  - N hospital sites each carry a *local* cohort that never leaves
    their walls.
  - Each site fits its own Beta-Binomial posterior on local data.
  - A central server aggregates only the *parameters* (alpha_post,
    beta_post per bin) weighted by site sample size.
  - The aggregated model is broadcast back to the sites for the
    next round.
  - Optional **differentially-private FedAvg**: each site adds
    Laplace noise to its parameters before sending; this honours
    GDPR Article 5(1)(c) data-minimisation + reduces re-
    identification risk on small bins.

Two counterfactuals are computed:
  - **Centralized baseline**: a single Beta-Binomial fit on the
    union of all per-site cohorts (the "if we could pool" upper
    bound).
  - **Local-only baseline**: each site uses its own model with no
    federation (the "no collaboration" lower bound).

Output: per-round + per-site ECE / Brier + global aggregated model
+ privacy-budget tracking. Pure-Python deterministic. No NumPy.

References:
- McMahan et al. 2017 "Communication-Efficient Learning of Deep
  Networks from Decentralized Data" (AISTATS).
- Dwork et al. 2006 "Calibrating Noise to Sensitivity".
- Geyer et al. 2017 "Differentially Private Federated Learning".
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Calibration anchor (matches W1 spec_002)
# ─────────────────────────────────────────────────────────────────────


_LACE_BUCKETS: list[tuple[range, str]] = [
    (range(0, 3),   "lace_0_2"),
    (range(3, 6),   "lace_3_5"),
    (range(6, 10),  "lace_6_9"),
    (range(10, 13), "lace_10_12"),
    (range(13, 60), "lace_13_19"),
]


_BUCKET_PROB_TRUE = {
    "lace_0_2": 0.072, "lace_3_5": 0.103,
    "lace_6_9": 0.158, "lace_10_12": 0.234,
    "lace_13_19": 0.327,
}


_PRIOR_ALPHA = 2.0
_PRIOR_BETA = 18.0


def _bucket_for(lace: int) -> str:
    for r, name in _LACE_BUCKETS:
        if lace in r:
            return name
    return _LACE_BUCKETS[-1][1]


# ─────────────────────────────────────────────────────────────────────
# Site cohort generators
# ─────────────────────────────────────────────────────────────────────


_SITE_PROFILES: dict[str, dict[str, float | str]] = {
    "site_urban_academic": {
        "lace_skew": 0.0,    # neutral
        "outcome_mult": 1.00,
        "bias_label": "balanced",
    },
    "site_rural_community": {
        "lace_skew": -1.0,   # younger / lower acuity
        "outcome_mult": 0.90,
        "bias_label": "low-acuity bias",
    },
    "site_safety_net": {
        "lace_skew": 1.5,    # sicker, more comorbid
        "outcome_mult": 1.20,
        "bias_label": "high-acuity Medicaid-heavy",
    },
    "site_geriatric_specialty": {
        "lace_skew": 2.0,    # 75+ skew
        "outcome_mult": 1.10,
        "bias_label": "geriatric-skewed",
    },
    "site_pediatric_adjacent": {
        "lace_skew": -2.0,   # mixed adolescent / young adult
        "outcome_mult": 0.85,
        "bias_label": "young-adult-skewed",
    },
}


class SiteCohort(BaseModel):
    site_id: str
    n: int = Field(ge=0)
    bias_label: str
    counts_per_bucket: dict[str, int]
    successes_per_bucket: dict[str, int]


def _sample_lace_with_skew(rng: random.Random, skew: float) -> int:
    """Sample a LACE total with a per-site shift."""
    L = rng.choices([1, 2, 3, 4, 5, 6, 7],
                    weights=[8, 14, 20, 22, 18, 12, 6])[0]
    A = rng.choices([0, 3], weights=[40, 60])[0]
    C = rng.choices([0, 1, 2, 3, 4, 5],
                    weights=[20, 22, 22, 18, 12, 6])[0]
    E = rng.choices([0, 1, 2, 3, 4],
                    weights=[35, 28, 18, 12, 7])[0]
    raw = L + A + C + E + int(round(skew))
    return max(0, min(19, raw))


def generate_site_cohort(
    *, site_id: str, n: int,
    seed: int = 42,
) -> SiteCohort:
    """Generate a synthetic per-site cohort respecting the site's
    profile (LACE skew + outcome multiplier)."""
    if site_id not in _SITE_PROFILES:
        raise ValueError(f"unknown site_id: {site_id!r}")
    if n <= 0:
        raise ValueError("n must be > 0")
    profile = _SITE_PROFILES[site_id]
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    successes: dict[str, int] = {}
    for r, name in _LACE_BUCKETS:
        counts[name] = 0
        successes[name] = 0
    for _ in range(n):
        lace = _sample_lace_with_skew(
            rng, float(profile["lace_skew"]))
        bucket = _bucket_for(lace)
        rate = (
            _BUCKET_PROB_TRUE[bucket]
            * float(profile["outcome_mult"])
        )
        rate = min(0.95, max(0.005, rate))
        outcome = 1 if rng.random() < rate else 0
        counts[bucket] += 1
        successes[bucket] += outcome
    return SiteCohort(
        site_id=site_id, n=n,
        bias_label=str(profile["bias_label"]),
        counts_per_bucket=counts,
        successes_per_bucket=successes,
    )


# ─────────────────────────────────────────────────────────────────────
# Beta-Binomial parameters
# ─────────────────────────────────────────────────────────────────────


class BBPosterior(BaseModel):
    alpha_per_bucket: dict[str, float]
    beta_per_bucket: dict[str, float]

    def mean_per_bucket(self) -> dict[str, float]:
        return {
            b: round(
                self.alpha_per_bucket[b]
                / (self.alpha_per_bucket[b] + self.beta_per_bucket[b]),
                6,
            )
            for b in self.alpha_per_bucket
        }


def _prior_posterior() -> BBPosterior:
    return BBPosterior(
        alpha_per_bucket={
            name: _PRIOR_ALPHA for _, name in _LACE_BUCKETS
        },
        beta_per_bucket={
            name: _PRIOR_BETA for _, name in _LACE_BUCKETS
        },
    )


def local_update(
    cohort: SiteCohort, *, prior: BBPosterior | None = None,
) -> BBPosterior:
    """Conjugate Beta-Binomial update on a site's local cohort.

    prior + (k_b, n_b - k_b) -> posterior.
    """
    prior = prior if prior is not None else _prior_posterior()
    alpha: dict[str, float] = {}
    beta: dict[str, float] = {}
    for _, name in _LACE_BUCKETS:
        n_b = cohort.counts_per_bucket.get(name, 0)
        k_b = cohort.successes_per_bucket.get(name, 0)
        alpha[name] = prior.alpha_per_bucket[name] + k_b
        beta[name] = prior.beta_per_bucket[name] + (n_b - k_b)
    return BBPosterior(
        alpha_per_bucket=alpha, beta_per_bucket=beta,
    )


# ─────────────────────────────────────────────────────────────────────
# FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────


def fedavg_aggregate(
    site_posteriors: list[tuple[BBPosterior, int]],
) -> BBPosterior:
    """Sample-size-weighted average of Beta-Binomial parameters.

    Each tuple is (posterior, n_site). Result is the McMahan 2017
    FedAvg of the alpha/beta vectors.
    """
    if not site_posteriors:
        raise ValueError("site_posteriors cannot be empty")
    total_n = sum(n for _, n in site_posteriors)
    if total_n <= 0:
        raise ValueError("sum of site n must be > 0")
    alpha: dict[str, float] = {}
    beta: dict[str, float] = {}
    for _, name in _LACE_BUCKETS:
        a = sum(
            (n / total_n) * p.alpha_per_bucket[name]
            for p, n in site_posteriors
        )
        b = sum(
            (n / total_n) * p.beta_per_bucket[name]
            for p, n in site_posteriors
        )
        alpha[name] = a
        beta[name] = b
    return BBPosterior(
        alpha_per_bucket=alpha, beta_per_bucket=beta,
    )


# ─────────────────────────────────────────────────────────────────────
# Differentially-private FedAvg
# ─────────────────────────────────────────────────────────────────────


def _laplace_noise(scale: float, rng: random.Random) -> float:
    u = rng.random() - 0.5
    if abs(u) >= 0.5:
        u = 0.4999
    sign = 1.0 if u >= 0 else -1.0
    return -scale * sign * math.log(1 - 2 * abs(u))


def add_laplace_noise(
    posterior: BBPosterior,
    *,
    epsilon: float, sensitivity: float = 1.0,
    seed: int = 0,
) -> BBPosterior:
    """Add Laplace(0, sensitivity/epsilon) noise to each alpha + beta
    parameter. Clips to >= 0 (parameters cannot be negative)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    rng = random.Random(seed)
    scale = sensitivity / epsilon
    alpha: dict[str, float] = {}
    beta: dict[str, float] = {}
    for k, v in posterior.alpha_per_bucket.items():
        alpha[k] = max(0.001, v + _laplace_noise(scale, rng))
    for k, v in posterior.beta_per_bucket.items():
        beta[k] = max(0.001, v + _laplace_noise(scale, rng))
    return BBPosterior(
        alpha_per_bucket=alpha, beta_per_bucket=beta,
    )


# ─────────────────────────────────────────────────────────────────────
# ECE on a held-out cohort
# ─────────────────────────────────────────────────────────────────────


def _ece(
    cohort: SiteCohort, posterior: BBPosterior,
) -> float:
    """ECE on the cohort using the posterior's per-bucket means."""
    means = posterior.mean_per_bucket()
    n_total = cohort.n
    if n_total == 0:
        return 0.0
    err = 0.0
    for _, name in _LACE_BUCKETS:
        n_b = cohort.counts_per_bucket.get(name, 0)
        k_b = cohort.successes_per_bucket.get(name, 0)
        if n_b == 0:
            continue
        observed_rate = k_b / n_b
        err += (n_b / n_total) * abs(means[name] - observed_rate)
    return round(err, 6)


def _brier(
    cohort: SiteCohort, posterior: BBPosterior,
) -> float:
    """Approximate Brier score using bucket means + bucket counts."""
    means = posterior.mean_per_bucket()
    n_total = cohort.n
    if n_total == 0:
        return 0.0
    score = 0.0
    for _, name in _LACE_BUCKETS:
        n_b = cohort.counts_per_bucket.get(name, 0)
        k_b = cohort.successes_per_bucket.get(name, 0)
        if n_b == 0:
            continue
        p = means[name]
        # n_b - k_b negatives (y=0) + k_b positives (y=1)
        score += (n_b - k_b) * (p ** 2)
        score += k_b * ((1.0 - p) ** 2)
    return round(score / n_total, 6)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


class FederatedRound(BaseModel):
    round_index: int
    global_posterior: BBPosterior
    per_site_ece: dict[str, float]
    global_ece: float
    privacy_epsilon_used: float


class FederatedTrainingReport(BaseModel):
    n_sites: int
    n_rounds: int
    epsilon_per_round: float
    rounds: list[FederatedRound]
    centralized_baseline_posterior: BBPosterior
    centralized_baseline_ece: float
    local_only_ece: dict[str, float]
    rationale: str


def _centralized_baseline(
    cohorts: list[SiteCohort],
) -> BBPosterior:
    """Train on the union of all per-site cohorts."""
    counts: dict[str, int] = {}
    succ: dict[str, int] = {}
    for _, name in _LACE_BUCKETS:
        counts[name] = sum(
            c.counts_per_bucket.get(name, 0) for c in cohorts
        )
        succ[name] = sum(
            c.successes_per_bucket.get(name, 0) for c in cohorts
        )
    fake = SiteCohort(
        site_id="centralized", n=sum(c.n for c in cohorts),
        bias_label="union",
        counts_per_bucket=counts, successes_per_bucket=succ,
    )
    return local_update(fake)


def run_federated_training(
    *,
    site_ids: list[str],
    per_site_n: int = 1000,
    n_rounds: int = 10,
    epsilon_per_round: float | None = None,
    base_seed: int = 20260430,
) -> FederatedTrainingReport:
    """Run R rounds of FedAvg (or DP-FedAvg if ``epsilon_per_round``
    is set) over N sites and return the per-round trace."""
    if not site_ids:
        raise ValueError("site_ids cannot be empty")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be > 0")
    if per_site_n <= 0:
        raise ValueError("per_site_n must be > 0")

    cohorts = [
        generate_site_cohort(
            site_id=sid, n=per_site_n,
            seed=base_seed + i,
        )
        for i, sid in enumerate(site_ids)
    ]
    centralized = _centralized_baseline(cohorts)
    centralized_ece = _ece(
        SiteCohort(
            site_id="union", n=sum(c.n for c in cohorts),
            bias_label="union",
            counts_per_bucket={
                name: sum(
                    c.counts_per_bucket.get(name, 0) for c in cohorts
                )
                for _, name in _LACE_BUCKETS
            },
            successes_per_bucket={
                name: sum(
                    c.successes_per_bucket.get(name, 0)
                    for c in cohorts
                )
                for _, name in _LACE_BUCKETS
            },
        ),
        centralized,
    )
    local_only_ece = {
        c.site_id: _ece(c, local_update(c))
        for c in cohorts
    }

    global_posterior: BBPosterior = _prior_posterior()
    rounds: list[FederatedRound] = []
    privacy_used = 0.0
    for r in range(n_rounds):
        site_posteriors: list[tuple[BBPosterior, int]] = []
        for i, c in enumerate(cohorts):
            local = local_update(c, prior=global_posterior)
            if epsilon_per_round is not None:
                local = add_laplace_noise(
                    local,
                    epsilon=epsilon_per_round,
                    sensitivity=1.0,
                    seed=base_seed + r * 31 + i,
                )
            site_posteriors.append((local, c.n))
        global_posterior = fedavg_aggregate(site_posteriors)
        ece_per_site = {
            c.site_id: _ece(c, global_posterior) for c in cohorts
        }
        global_ece = _ece(
            SiteCohort(
                site_id="union",
                n=sum(c.n for c in cohorts),
                bias_label="union",
                counts_per_bucket={
                    name: sum(
                        c.counts_per_bucket.get(name, 0)
                        for c in cohorts
                    )
                    for _, name in _LACE_BUCKETS
                },
                successes_per_bucket={
                    name: sum(
                        c.successes_per_bucket.get(name, 0)
                        for c in cohorts
                    )
                    for _, name in _LACE_BUCKETS
                },
            ),
            global_posterior,
        )
        if epsilon_per_round is not None:
            privacy_used += epsilon_per_round
        rounds.append(FederatedRound(
            round_index=r,
            global_posterior=global_posterior,
            per_site_ece=ece_per_site,
            global_ece=global_ece,
            privacy_epsilon_used=round(privacy_used, 4),
        ))

    final_global_ece = rounds[-1].global_ece
    return FederatedTrainingReport(
        n_sites=len(site_ids),
        n_rounds=n_rounds,
        epsilon_per_round=epsilon_per_round or 0.0,
        rounds=rounds,
        centralized_baseline_posterior=centralized,
        centralized_baseline_ece=centralized_ece,
        local_only_ece=local_only_ece,
        rationale=(
            f"FedAvg over {len(site_ids)} sites x {n_rounds} rounds; "
            f"final global ECE {final_global_ece:.4f} vs centralized "
            f"baseline {centralized_ece:.4f} (gap "
            f"{abs(final_global_ece - centralized_ece):.4f}); "
            f"total epsilon spent {privacy_used:.2f}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Privacy-utility curve
# ─────────────────────────────────────────────────────────────────────


class PrivacyUtilityPoint(BaseModel):
    epsilon_per_round: float
    final_global_ece: float
    centralized_baseline_ece: float
    federated_overhead: float


class PrivacyUtilityReport(BaseModel):
    n_sites: int
    epsilon_grid: list[float]
    points: list[PrivacyUtilityPoint]
    rationale: str


def privacy_utility_curve(
    *,
    site_ids: list[str],
    per_site_n: int = 1000,
    n_rounds: int = 10,
    epsilon_grid: list[float] | None = None,
    base_seed: int = 20260430,
) -> PrivacyUtilityReport:
    """Sweep ``epsilon_per_round`` across a grid + report the final
    global ECE for each privacy budget."""
    if not site_ids:
        raise ValueError("site_ids cannot be empty")
    grid = epsilon_grid or [0.1, 0.5, 1.0, 5.0, 10.0]
    points: list[PrivacyUtilityPoint] = []
    for eps in grid:
        rep = run_federated_training(
            site_ids=site_ids,
            per_site_n=per_site_n,
            n_rounds=n_rounds,
            epsilon_per_round=eps,
            base_seed=base_seed,
        )
        final = rep.rounds[-1].global_ece
        points.append(PrivacyUtilityPoint(
            epsilon_per_round=eps,
            final_global_ece=round(final, 6),
            centralized_baseline_ece=rep.centralized_baseline_ece,
            federated_overhead=round(
                final - rep.centralized_baseline_ece, 6
            ),
        ))
    return PrivacyUtilityReport(
        n_sites=len(site_ids),
        epsilon_grid=list(grid),
        points=points,
        rationale=(
            f"Privacy-utility sweep across {len(grid)} epsilon "
            f"values; lowest final ECE "
            f"{min(p.final_global_ece for p in points):.4f}."
        ),
    )
