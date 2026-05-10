"""Phase 17.AO - FedProx for heterogeneous federated clients.

Li et al. 2020 "Federated Optimization in Heterogeneous Networks".
Extends FedAvg with a proximal term that pulls each client's local
update toward the latest global model, bounding client drift on
non-IID cohorts.

For Beta-Binomial parameters the proximal term simplifies to a
weighted average between the conjugate posterior and the broadcast
global model:

    alpha_local^new = (1-eta) * alpha_local_conjugate + eta * alpha_global
    beta_local^new  = (1-eta) * beta_local_conjugate  + eta * beta_global

where ``eta = mu / (mu + 1)`` maps the proximal weight ``mu`` to a
[0, 1] interpolation factor.

Pure-Python deterministic. Reuses the FedAvg machinery from the
``federated_learning`` module.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .federated_learning import (
    BBPosterior, FederatedRound, FederatedTrainingReport,
    SiteCohort, _LACE_BUCKETS, _centralized_baseline,
    _ece, _prior_posterior, add_laplace_noise,
    fedavg_aggregate, generate_site_cohort, local_update,
)


class FedProxRound(BaseModel):
    round_index: int
    global_posterior: BBPosterior
    per_site_ece: dict[str, float]
    global_ece: float
    avg_client_drift_l1: float
    privacy_epsilon_used: float


class FedProxTrainingReport(BaseModel):
    n_sites: int
    n_rounds: int
    proximal_mu: float
    epsilon_per_round: float
    rounds: list[FedProxRound]
    centralized_baseline_posterior: BBPosterior
    centralized_baseline_ece: float
    final_global_ece: float
    rationale: str


def _proximal_local_update(
    cohort: SiteCohort,
    global_posterior: BBPosterior,
    *, mu: float,
) -> tuple[BBPosterior, float]:
    """Local update with FedProx proximal term + return the L1
    drift between the conjugate posterior and the proximal one."""
    if mu < 0:
        raise ValueError("mu must be >= 0")
    conjugate = local_update(cohort, prior=global_posterior)
    eta = mu / (mu + 1.0) if mu > 0 else 0.0
    alpha = {}
    beta = {}
    drift = 0.0
    for _, name in _LACE_BUCKETS:
        a_local = (1 - eta) * conjugate.alpha_per_bucket[name] \
            + eta * global_posterior.alpha_per_bucket[name]
        b_local = (1 - eta) * conjugate.beta_per_bucket[name] \
            + eta * global_posterior.beta_per_bucket[name]
        alpha[name] = a_local
        beta[name] = b_local
        drift += abs(
            conjugate.alpha_per_bucket[name] - a_local
        )
        drift += abs(
            conjugate.beta_per_bucket[name] - b_local
        )
    return BBPosterior(
        alpha_per_bucket=alpha, beta_per_bucket=beta,
    ), round(drift, 6)


def run_fedprox_training(
    *,
    site_ids: list[str],
    per_site_n: int = 1000,
    n_rounds: int = 10,
    proximal_mu: float = 0.1,
    epsilon_per_round: float | None = None,
    base_seed: int = 20260430,
) -> FedProxTrainingReport:
    """Run FedProx training. ``proximal_mu=0`` recovers FedAvg."""
    if not site_ids:
        raise ValueError("site_ids cannot be empty")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be > 0")
    if proximal_mu < 0:
        raise ValueError("proximal_mu must be >= 0")

    cohorts = [
        generate_site_cohort(
            site_id=sid, n=per_site_n, seed=base_seed + i,
        )
        for i, sid in enumerate(site_ids)
    ]
    centralized = _centralized_baseline(cohorts)
    union = SiteCohort(
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
                c.successes_per_bucket.get(name, 0) for c in cohorts
            )
            for _, name in _LACE_BUCKETS
        },
    )
    centralized_ece = _ece(union, centralized)

    global_posterior: BBPosterior = _prior_posterior()
    rounds: list[FedProxRound] = []
    privacy_used = 0.0
    for r in range(n_rounds):
        site_posteriors: list[tuple[BBPosterior, int]] = []
        drifts: list[float] = []
        for i, c in enumerate(cohorts):
            local, drift = _proximal_local_update(
                c, global_posterior, mu=proximal_mu,
            )
            drifts.append(drift)
            if epsilon_per_round is not None:
                local = add_laplace_noise(
                    local, epsilon=epsilon_per_round,
                    sensitivity=1.0,
                    seed=base_seed + r * 31 + i,
                )
            site_posteriors.append((local, c.n))
        global_posterior = fedavg_aggregate(site_posteriors)
        ece_per_site = {
            c.site_id: _ece(c, global_posterior) for c in cohorts
        }
        global_ece = _ece(union, global_posterior)
        if epsilon_per_round is not None:
            privacy_used += epsilon_per_round
        rounds.append(FedProxRound(
            round_index=r,
            global_posterior=global_posterior,
            per_site_ece=ece_per_site,
            global_ece=global_ece,
            avg_client_drift_l1=round(
                sum(drifts) / max(1, len(drifts)), 6
            ),
            privacy_epsilon_used=round(privacy_used, 4),
        ))

    return FedProxTrainingReport(
        n_sites=len(site_ids),
        n_rounds=n_rounds,
        proximal_mu=proximal_mu,
        epsilon_per_round=epsilon_per_round or 0.0,
        rounds=rounds,
        centralized_baseline_posterior=centralized,
        centralized_baseline_ece=centralized_ece,
        final_global_ece=rounds[-1].global_ece,
        rationale=(
            f"FedProx mu={proximal_mu} over {len(site_ids)} sites x "
            f"{n_rounds} rounds; final global ECE "
            f"{rounds[-1].global_ece:.4f} vs centralized "
            f"{centralized_ece:.4f}; avg client drift L1 over "
            f"final round "
            f"{rounds[-1].avg_client_drift_l1:.4f}."
        ),
    )
