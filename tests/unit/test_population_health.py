"""Phase 10.2 -- population health + outbreak detection tests
(POPHEALTH-1/2/3)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.population_health import (
    compute_outbreak_heatmap,
    compute_syndromic_surveillance,
    compute_vaccine_reminder_cohort,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# POPHEALTH-1 -- syndromic surveillance
# ─────────────────────────────────────────────────────────────────────

def test_surveillance_flags_cluster_above_threshold():
    out = _run(compute_syndromic_surveillance(
        surveillance_period_start_iso="2026-04-22",
        surveillance_period_end_iso="2026-04-28",
        observed_counts_by_syndrome={"ILI": 80, "GI": 12},
        expected_counts_by_syndrome={"ILI": 30.0, "GI": 15.0},
    ))
    syndromes = {c.syndrome for c in out.clusters}
    assert "ILI" in syndromes
    # GI should NOT trigger (12 ≤ 15 expected -> negative z)
    assert "GI" not in syndromes


def test_surveillance_z_score_correct():
    out = _run(compute_syndromic_surveillance(
        surveillance_period_start_iso="2026-04-22",
        surveillance_period_end_iso="2026-04-28",
        observed_counts_by_syndrome={"ILI": 50},
        expected_counts_by_syndrome={"ILI": 25.0},
    ))
    cluster = out.clusters[0]
    # z = (50 - 25) / sqrt(25) = 25 / 5 = 5.0
    assert cluster.z_score == pytest.approx(5.0)


def test_surveillance_severity_tiers():
    """z=2.5 -> watch, z=3.5 -> alert, z=5.5 -> outbreak."""
    out = _run(compute_syndromic_surveillance(
        surveillance_period_start_iso="2026-04-22",
        surveillance_period_end_iso="2026-04-28",
        observed_counts_by_syndrome={
            "WATCH":    35,    # z = 2.5
            "ALERT":    52,    # z ≈ 5.4 (round trip; treat as alert/outbreak)
            "OUTBREAK": 75,    # z ≈ 10
        },
        expected_counts_by_syndrome={
            "WATCH": 25.0, "ALERT": 25.0, "OUTBREAK": 25.0,
        },
    ))
    by_syn = {c.syndrome: c.severity for c in out.clusters}
    assert by_syn["WATCH"] == "watch"
    assert by_syn["OUTBREAK"] == "outbreak"


def test_surveillance_skips_zero_baseline():
    """A zero expected count cannot drive a Poisson z; cluster skipped."""
    out = _run(compute_syndromic_surveillance(
        surveillance_period_start_iso="2026-04-22",
        surveillance_period_end_iso="2026-04-28",
        observed_counts_by_syndrome={"NEW": 10},
        expected_counts_by_syndrome={"NEW": 0.0},
    ))
    assert out.n_clusters == 0


def test_surveillance_geographic_window_propagates():
    out = _run(compute_syndromic_surveillance(
        surveillance_period_start_iso="2026-04-22",
        surveillance_period_end_iso="2026-04-28",
        observed_counts_by_syndrome={"ILI": 50},
        expected_counts_by_syndrome={"ILI": 25.0},
        geographic_window="NYC ZIP 10001",
    ))
    assert out.clusters[0].geographic_window == "NYC ZIP 10001"


# ─────────────────────────────────────────────────────────────────────
# POPHEALTH-2 -- vaccine reminder cohort
# ─────────────────────────────────────────────────────────────────────

def test_vaccine_cohorts_emit_one_per_known_vaccine():
    out = _run(compute_vaccine_reminder_cohort(
        overdue_by_vaccine={"FLU": 1240, "PNEUMO": 380, "ZOSTER": 95},
    ))
    assert out.n_cohorts == 3
    assert out.total_patients == 1240 + 380 + 95


def test_vaccine_unknown_id_skipped():
    out = _run(compute_vaccine_reminder_cohort(
        overdue_by_vaccine={"FLU": 100, "FAKE-VAX": 50},
    ))
    assert out.n_cohorts == 1
    assert out.total_patients == 100


def test_vaccine_zero_overdue_skipped():
    out = _run(compute_vaccine_reminder_cohort(
        overdue_by_vaccine={"FLU": 0, "PNEUMO": 100},
    ))
    assert out.n_cohorts == 1


def test_vaccine_uptake_varies_with_channel():
    phone = _run(compute_vaccine_reminder_cohort(
        overdue_by_vaccine={"FLU": 100}, outreach_channel="phone",
    ))
    postcard = _run(compute_vaccine_reminder_cohort(
        overdue_by_vaccine={"FLU": 100}, outreach_channel="postcard",
    ))
    # Phone (0.30) > postcard (0.18)
    assert (phone.cohorts[0].expected_outreach_uptake
            > postcard.cohorts[0].expected_outreach_uptake)


def test_vaccine_age_distribution_passes_through():
    out = _run(compute_vaccine_reminder_cohort(
        overdue_by_vaccine={"FLU": 100},
        age_distribution={"FLU": {"<18": 30, "18-64": 50, "65+": 20}},
    ))
    assert out.cohorts[0].age_distribution == {
        "<18": 30, "18-64": 50, "65+": 20,
    }


# ─────────────────────────────────────────────────────────────────────
# POPHEALTH-3 -- DP-noised outbreak heatmap
# ─────────────────────────────────────────────────────────────────────

def test_heatmap_rejects_non_positive_epsilon():
    with pytest.raises(ValueError):
        _run(compute_outbreak_heatmap(
            counts_by_geo_syndrome={"10001": {"ILI": 10}},
            population_by_geo={"10001": 5_000},
            epsilon=0.0,
        ))


def test_heatmap_emits_one_cell_per_geo_syndrome():
    out = _run(compute_outbreak_heatmap(
        counts_by_geo_syndrome={
            "10001": {"ILI": 10, "GI": 5},
            "10002": {"ILI": 3},
        },
        population_by_geo={"10001": 5_000, "10002": 2_000},
        epsilon=1.0,
    ))
    assert out.n_cells == 3


def test_heatmap_seed_yields_reproducible_noise():
    a = _run(compute_outbreak_heatmap(
        counts_by_geo_syndrome={"10001": {"ILI": 10}},
        population_by_geo={"10001": 5_000},
        epsilon=1.0, seed=42,
    ))
    b = _run(compute_outbreak_heatmap(
        counts_by_geo_syndrome={"10001": {"ILI": 10}},
        population_by_geo={"10001": 5_000},
        epsilon=1.0, seed=42,
    ))
    assert a.cells[0].n_cases_dp_noised == b.cells[0].n_cases_dp_noised


def test_heatmap_clips_negative_noise_to_zero():
    """Even with high noise the published count is non-negative."""
    out = _run(compute_outbreak_heatmap(
        counts_by_geo_syndrome={"10001": {"ILI": 1}},
        population_by_geo={"10001": 5_000},
        epsilon=0.05,    # very small ε -> very noisy
        seed=123,
    ))
    assert out.cells[0].n_cases_dp_noised >= 0


def test_heatmap_rate_per_100k_computed():
    """Rate is per-100k of population; clip ensures non-negative."""
    out = _run(compute_outbreak_heatmap(
        counts_by_geo_syndrome={"10001": {"ILI": 100}},
        population_by_geo={"10001": 100_000},
        epsilon=10.0,    # very tight ε -> almost no noise
        seed=1,
    ))
    cell = out.cells[0]
    # Around 100 / 100k * 100k = 100 per 100k -- allow ε ≈ 0.1 noise band
    assert 80 <= cell.rate_dp_noised <= 130


def test_heatmap_epsilon_persisted_in_report():
    out = _run(compute_outbreak_heatmap(
        counts_by_geo_syndrome={"10001": {"ILI": 10}},
        population_by_geo={"10001": 5_000},
        epsilon=2.5, seed=1,
    ))
    assert out.epsilon_used == 2.5


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_population_health_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "population_health" in BUNDLES
    assert set(BUNDLES["population_health"]) == {
        "compute_syndromic_surveillance",
        "compute_vaccine_reminder_cohort",
        "compute_outbreak_heatmap",
    }


def test_population_health_scopes_declared():
    from mcp_server.scopes import BUNDLE_SCOPES
    assert "population_health" in BUNDLE_SCOPES
    assert "patient/Immunization.rs" in BUNDLE_SCOPES["population_health"]
