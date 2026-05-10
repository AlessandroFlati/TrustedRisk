"""Phase 17.AO - FedProx tests."""

from __future__ import annotations

import pytest

from a2a_agent.fedprox import (
    FedProxTrainingReport, _proximal_local_update,
    run_fedprox_training,
)
from a2a_agent.federated_learning import (
    _prior_posterior, generate_site_cohort,
)


_SITE_IDS = [
    "site_urban_academic",
    "site_rural_community",
    "site_safety_net",
    "site_geriatric_specialty",
    "site_pediatric_adjacent",
]


# ─────────────────────────────────────────────────────────────────────
# Proximal update
# ─────────────────────────────────────────────────────────────────────

def test_proximal_update_with_mu_zero_equals_conjugate():
    cohort = generate_site_cohort(
        site_id="site_urban_academic", n=400, seed=1)
    global_post = _prior_posterior()
    proxed, drift = _proximal_local_update(
        cohort, global_post, mu=0.0)
    # mu=0 -> conjugate; drift relative to conjugate is exactly 0
    assert drift == 0.0


def test_proximal_update_with_high_mu_pulls_toward_global():
    cohort = generate_site_cohort(
        site_id="site_safety_net", n=400, seed=1)
    global_post = _prior_posterior()
    low_mu, low_drift = _proximal_local_update(
        cohort, global_post, mu=0.1)
    high_mu, high_drift = _proximal_local_update(
        cohort, global_post, mu=10.0)
    # Higher mu -> larger drift relative to the conjugate
    assert high_drift > low_drift


def test_proximal_update_rejects_negative_mu():
    cohort = generate_site_cohort(
        site_id="site_urban_academic", n=10, seed=1)
    with pytest.raises(ValueError):
        _proximal_local_update(
            cohort, _prior_posterior(), mu=-1.0)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def test_fedprox_runs_n_rounds():
    rep = run_fedprox_training(
        site_ids=_SITE_IDS, per_site_n=300, n_rounds=5,
        proximal_mu=0.1,
    )
    assert rep.n_rounds == 5
    assert len(rep.rounds) == 5


def test_fedprox_with_mu_zero_matches_fedavg_ece():
    """mu=0 -> FedProx == FedAvg, so the final ECE should be
    identical to the FedAvg path on the same seed."""
    from a2a_agent.federated_learning import (
        run_federated_training,
    )
    fedavg = run_federated_training(
        site_ids=_SITE_IDS, per_site_n=400, n_rounds=4,
        base_seed=11,
    )
    fedprox = run_fedprox_training(
        site_ids=_SITE_IDS, per_site_n=400, n_rounds=4,
        proximal_mu=0.0, base_seed=11,
    )
    assert (
        abs(fedavg.rounds[-1].global_ece -
            fedprox.rounds[-1].global_ece) < 1e-6
    )


def test_fedprox_higher_mu_increases_avg_drift():
    low = run_fedprox_training(
        site_ids=_SITE_IDS, per_site_n=300, n_rounds=2,
        proximal_mu=0.05, base_seed=21,
    )
    high = run_fedprox_training(
        site_ids=_SITE_IDS, per_site_n=300, n_rounds=2,
        proximal_mu=5.0, base_seed=21,
    )
    assert (
        high.rounds[-1].avg_client_drift_l1
        > low.rounds[-1].avg_client_drift_l1
    )


def test_fedprox_rejects_zero_rounds():
    with pytest.raises(ValueError):
        run_fedprox_training(
            site_ids=_SITE_IDS, per_site_n=100, n_rounds=0,
        )


def test_fedprox_rejects_empty_site_ids():
    with pytest.raises(ValueError):
        run_fedprox_training(
            site_ids=[], per_site_n=100, n_rounds=1,
        )


def test_fedprox_dp_increments_privacy_budget():
    rep = run_fedprox_training(
        site_ids=_SITE_IDS, per_site_n=200, n_rounds=3,
        proximal_mu=0.1, epsilon_per_round=1.0,
    )
    assert rep.rounds[-1].privacy_epsilon_used == 3.0


def test_fedprox_round_trip_through_pydantic():
    rep = run_fedprox_training(
        site_ids=_SITE_IDS, per_site_n=200, n_rounds=2,
        proximal_mu=0.1,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = FedProxTrainingReport.model_validate(payload)
    assert rebuilt.n_rounds == rep.n_rounds


def test_fedprox_final_ece_close_to_centralized():
    rep = run_fedprox_training(
        site_ids=_SITE_IDS, per_site_n=600, n_rounds=8,
        proximal_mu=0.1,
    )
    gap = abs(
        rep.final_global_ece - rep.centralized_baseline_ece
    )
    assert gap < 0.05
