"""Phase 17.AN - Federated Learning simulator tests."""

from __future__ import annotations

import pytest

from a2a_agent.federated_learning import (
    BBPosterior, FederatedTrainingReport, PrivacyUtilityReport,
    SiteCohort, add_laplace_noise, fedavg_aggregate,
    generate_site_cohort, local_update, privacy_utility_curve,
    run_federated_training,
)


_SITE_IDS = [
    "site_urban_academic",
    "site_rural_community",
    "site_safety_net",
    "site_geriatric_specialty",
    "site_pediatric_adjacent",
]


# ─────────────────────────────────────────────────────────────────────
# Site cohort generation
# ─────────────────────────────────────────────────────────────────────

def test_generate_cohort_yields_n_observations():
    c = generate_site_cohort(
        site_id="site_urban_academic", n=500, seed=1)
    assert c.n == 500
    assert sum(c.counts_per_bucket.values()) == 500


def test_generate_cohort_rejects_unknown_site():
    with pytest.raises(ValueError):
        generate_site_cohort(
            site_id="not_a_site", n=10, seed=1)


def test_generate_cohort_rejects_zero_n():
    with pytest.raises(ValueError):
        generate_site_cohort(
            site_id="site_urban_academic", n=0, seed=1)


def test_safety_net_site_has_higher_observed_rate_than_rural():
    """Safety-net cohort has outcome_mult=1.20 + sicker LACE skew;
    rural community has outcome_mult=0.90."""
    rural = generate_site_cohort(
        site_id="site_rural_community", n=2000, seed=10)
    safety = generate_site_cohort(
        site_id="site_safety_net", n=2000, seed=11)
    rural_rate = (
        sum(rural.successes_per_bucket.values()) / rural.n
    )
    safety_rate = (
        sum(safety.successes_per_bucket.values()) / safety.n
    )
    assert safety_rate > rural_rate


def test_cohort_seed_is_deterministic():
    a = generate_site_cohort(
        site_id="site_safety_net", n=200, seed=99)
    b = generate_site_cohort(
        site_id="site_safety_net", n=200, seed=99)
    assert a.counts_per_bucket == b.counts_per_bucket
    assert a.successes_per_bucket == b.successes_per_bucket


# ─────────────────────────────────────────────────────────────────────
# Local update
# ─────────────────────────────────────────────────────────────────────

def test_local_update_yields_posterior_with_all_buckets():
    c = generate_site_cohort(
        site_id="site_urban_academic", n=500, seed=1)
    p = local_update(c)
    expected = {"lace_0_2", "lace_3_5", "lace_6_9",
                 "lace_10_12", "lace_13_19"}
    assert set(p.alpha_per_bucket.keys()) == expected


def test_local_update_means_in_unit_interval():
    c = generate_site_cohort(
        site_id="site_urban_academic", n=500, seed=1)
    means = local_update(c).mean_per_bucket()
    for v in means.values():
        assert 0.0 <= v <= 1.0


# ─────────────────────────────────────────────────────────────────────
# FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────

def test_fedavg_recovers_average_for_two_equal_sites():
    c1 = generate_site_cohort(
        site_id="site_urban_academic", n=1000, seed=1)
    c2 = generate_site_cohort(
        site_id="site_urban_academic", n=1000, seed=2)
    p1 = local_update(c1)
    p2 = local_update(c2)
    aggregated = fedavg_aggregate([(p1, c1.n), (p2, c2.n)])
    for bucket in aggregated.alpha_per_bucket:
        avg_alpha = (
            p1.alpha_per_bucket[bucket]
            + p2.alpha_per_bucket[bucket]
        ) / 2
        assert abs(
            aggregated.alpha_per_bucket[bucket] - avg_alpha
        ) < 1e-6


def test_fedavg_rejects_empty():
    with pytest.raises(ValueError):
        fedavg_aggregate([])


def test_fedavg_rejects_zero_total_n():
    c = generate_site_cohort(
        site_id="site_urban_academic", n=10, seed=1)
    p = local_update(c)
    with pytest.raises(ValueError):
        fedavg_aggregate([(p, 0)])


# ─────────────────────────────────────────────────────────────────────
# DP-Laplace noise
# ─────────────────────────────────────────────────────────────────────

def test_laplace_noise_keeps_parameters_non_negative():
    c = generate_site_cohort(
        site_id="site_urban_academic", n=500, seed=1)
    p = local_update(c)
    noised = add_laplace_noise(p, epsilon=1.0, seed=7)
    for v in noised.alpha_per_bucket.values():
        assert v >= 0.0
    for v in noised.beta_per_bucket.values():
        assert v >= 0.0


def test_laplace_noise_rejects_zero_epsilon():
    c = generate_site_cohort(
        site_id="site_urban_academic", n=10, seed=1)
    p = local_update(c)
    with pytest.raises(ValueError):
        add_laplace_noise(p, epsilon=0.0, seed=1)


def test_laplace_noise_seeded_deterministic():
    c = generate_site_cohort(
        site_id="site_urban_academic", n=100, seed=1)
    p = local_update(c)
    a = add_laplace_noise(p, epsilon=1.0, seed=42)
    b = add_laplace_noise(p, epsilon=1.0, seed=42)
    assert a.alpha_per_bucket == b.alpha_per_bucket


# ─────────────────────────────────────────────────────────────────────
# Federated training driver
# ─────────────────────────────────────────────────────────────────────

def test_fedavg_training_runs_n_rounds():
    rep = run_federated_training(
        site_ids=_SITE_IDS, per_site_n=300, n_rounds=5,
    )
    assert rep.n_rounds == 5
    assert len(rep.rounds) == 5


def test_fedavg_final_ece_close_to_centralized_baseline():
    """With N=5 sites + per_site_n=600, the final FedAvg model
    should land within a few percentage points of the centralized
    baseline ECE."""
    rep = run_federated_training(
        site_ids=_SITE_IDS, per_site_n=600, n_rounds=8,
    )
    final_ece = rep.rounds[-1].global_ece
    gap = abs(final_ece - rep.centralized_baseline_ece)
    assert gap < 0.05


def test_fedavg_centralized_baseline_ece_is_low():
    rep = run_federated_training(
        site_ids=_SITE_IDS, per_site_n=600, n_rounds=3,
    )
    # Centralized fit on union has tiny ECE by construction
    assert rep.centralized_baseline_ece < 0.05


def test_fedavg_local_only_baseline_per_site_present():
    rep = run_federated_training(
        site_ids=_SITE_IDS, per_site_n=300, n_rounds=2,
    )
    assert set(rep.local_only_ece.keys()) == set(_SITE_IDS)


def test_fedavg_rejects_empty_site_ids():
    with pytest.raises(ValueError):
        run_federated_training(
            site_ids=[], per_site_n=100, n_rounds=1)


def test_fedavg_rejects_zero_rounds():
    with pytest.raises(ValueError):
        run_federated_training(
            site_ids=_SITE_IDS, per_site_n=100, n_rounds=0)


def test_fedavg_dp_increases_privacy_budget_used():
    rep = run_federated_training(
        site_ids=_SITE_IDS, per_site_n=200, n_rounds=4,
        epsilon_per_round=1.0,
    )
    # Cumulative epsilon grows monotonically
    eps_used = [r.privacy_epsilon_used for r in rep.rounds]
    for a, b in zip(eps_used, eps_used[1:]):
        assert b >= a
    assert rep.rounds[-1].privacy_epsilon_used == 4.0


def test_fedavg_round_trip_through_pydantic():
    rep = run_federated_training(
        site_ids=_SITE_IDS, per_site_n=200, n_rounds=3,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = FederatedTrainingReport.model_validate(payload)
    assert rebuilt.n_rounds == rep.n_rounds


# ─────────────────────────────────────────────────────────────────────
# Privacy-utility curve
# ─────────────────────────────────────────────────────────────────────

def test_privacy_utility_curve_emits_one_point_per_epsilon():
    rep = privacy_utility_curve(
        site_ids=_SITE_IDS, per_site_n=200, n_rounds=3,
        epsilon_grid=[0.5, 1.0, 5.0],
    )
    assert len(rep.points) == 3
    assert [p.epsilon_per_round for p in rep.points] == [0.5, 1.0, 5.0]


def test_privacy_utility_curve_rejects_empty():
    with pytest.raises(ValueError):
        privacy_utility_curve(
            site_ids=[], per_site_n=100, n_rounds=2,
        )


def test_privacy_utility_curve_round_trip():
    rep = privacy_utility_curve(
        site_ids=_SITE_IDS, per_site_n=100, n_rounds=2,
        epsilon_grid=[1.0],
    )
    payload = rep.model_dump(mode="json")
    rebuilt = PrivacyUtilityReport.model_validate(payload)
    assert rebuilt.n_sites == rep.n_sites
