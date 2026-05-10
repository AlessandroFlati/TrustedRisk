"""Phase 10.1 -- HEDIS / CMS Stars quality specialist tests (STARS-1/2/3)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.quality_stars import (
    compute_care_gap_priority_ranking,
    compute_quality_measures_aggregate,
    compute_stars_rating_forecast,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# STARS-1 -- compute_quality_measures_aggregate
# ─────────────────────────────────────────────────────────────────────

def test_aggregate_produces_one_measure_per_input_key():
    cohort = {
        "BCS": {"numerator": 1400, "denominator": 1968},
        "COL": {"numerator": 3300, "denominator": 5000},
        "CDC-HBA1C": {"numerator": 440, "denominator": 2000},
    }
    out = _run(compute_quality_measures_aggregate(
        measurement_year=2025, cohort_summary=cohort,
        n_eligible_patients=10_000,
    ))
    assert out.n_measures == 3
    by_id = {m.measure_id: m for m in out.measures}
    assert set(by_id) == {"BCS", "COL", "CDC-HBA1C"}


def test_aggregate_rate_matches_numerator_over_denominator():
    out = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 1500, "denominator": 2000}},
    ))
    assert out.measures[0].rate == 0.75


def test_aggregate_inverted_measure_lower_is_better():
    """CDC-HBA1C is poor-control rate -- lower is better. A 5 % rate
    must hit 5-star (cuts: 0.13 / 0.18 / 0.25)."""
    out = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"CDC-HBA1C": {"numerator": 50, "denominator": 1000}},
    ))
    assert out.measures[0].current_stars == 5


def test_aggregate_non_inverted_measure_high_is_better():
    """BCS is screening rate -- higher is better. A 90 % rate must hit
    5-star (cuts: 0.79 / 0.74 / 0.69)."""
    out = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 900, "denominator": 1000}},
    ))
    assert out.measures[0].current_stars == 5


def test_aggregate_unmet_count_only_for_non_inverted():
    """For a non-inverted measure (BCS), unmet = denominator - numerator."""
    out = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 700, "denominator": 1000}},
    ))
    assert out.measures[0].eligible_unmet_count == 300


def test_aggregate_unknown_measure_skipped_silently():
    """Unknown measure ids are not in _MEASURE_TABLE; the aggregator
    skips them rather than raising -- workspace admins may pre-filter."""
    out = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={
            "BCS": {"numerator": 700, "denominator": 1000},
            "FAKE-MEASURE": {"numerator": 1, "denominator": 2},
        },
    ))
    assert out.n_measures == 1


def test_aggregate_zero_denominator_yields_rate_zero():
    out = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 0, "denominator": 0}},
    ))
    assert out.measures[0].rate == 0.0


# ─────────────────────────────────────────────────────────────────────
# STARS-2 -- compute_stars_rating_forecast
# ─────────────────────────────────────────────────────────────────────

def test_forecast_emits_one_row_per_distinct_domain():
    """Mixing preventive_care (BCS) + chronic_conditions (CBP) measures
    yields a 2-row domain forecast."""
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={
            "BCS": {"numerator": 1500, "denominator": 2000},
            "CBP": {"numerator": 1500, "denominator": 2000},
        },
    ))
    fc = _run(compute_stars_rating_forecast(agg))
    domains = {d.domain for d in fc.domains}
    assert domains == {"preventive_care", "chronic_conditions"}


def test_forecast_overall_rises_with_better_rates():
    low = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 400, "denominator": 1000}},
    ))
    hi = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 850, "denominator": 1000}},
    ))
    fc_low = _run(compute_stars_rating_forecast(low))
    fc_hi = _run(compute_stars_rating_forecast(hi))
    assert fc_hi.overall_current > fc_low.overall_current


def test_forecast_qbp_below_4_star_is_zero():
    """Per Avalere: contracts under 4.0 Stars get $0 QBP."""
    poor = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 400, "denominator": 1000}},
    ))
    fc = _run(compute_stars_rating_forecast(poor))
    if fc.overall_projected_eom < 4.0:
        assert fc.estimated_qbp_dollars_at_overall == 0.0


def test_forecast_qbp_scales_with_contract_size():
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 900, "denominator": 1000}},
    ))
    fc_1k = _run(compute_stars_rating_forecast(
        agg, contract_size_thousand_members=1.0,
    ))
    fc_50k = _run(compute_stars_rating_forecast(
        agg, contract_size_thousand_members=50.0,
    ))
    if fc_1k.estimated_qbp_dollars_at_overall > 0:
        assert (fc_50k.estimated_qbp_dollars_at_overall
                == pytest.approx(fc_1k.estimated_qbp_dollars_at_overall * 50.0))


def test_forecast_accepts_dict_input():
    """The tool dehydrates a dict via Pydantic -- useful when an upstream
    A2A call hands the aggregate over the wire."""
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 800, "denominator": 1000}},
    ))
    fc = _run(compute_stars_rating_forecast(agg.model_dump()))
    assert fc.measurement_year == 2025


# ─────────────────────────────────────────────────────────────────────
# STARS-3 -- compute_care_gap_priority_ranking
# ─────────────────────────────────────────────────────────────────────

def test_ranking_skips_measures_without_unmet_gaps():
    """A 5-star measure with eligible_unmet_count == 0 must NOT appear
    in the ranking (no actionable lift)."""
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 1000, "denominator": 1000}},
    ))
    rk = _run(compute_care_gap_priority_ranking(agg))
    assert rk.n_actions == 0


def test_ranking_orders_by_expected_qbp_lift_descending():
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={
            # Many open BCS gaps + few CBP gaps
            "BCS": {"numerator": 200, "denominator": 1000},
            "CBP": {"numerator": 990, "denominator": 1000},
        },
    ))
    rk = _run(compute_care_gap_priority_ranking(agg))
    qbps = [a.expected_qbp_lift_dollars for a in rk.actions]
    assert qbps == sorted(qbps, reverse=True)


def test_ranking_top_n_caps_actions():
    cohort = {
        mid: {"numerator": 100, "denominator": 1000}
        for mid in ("BCS", "CCS", "COL", "CDC-EYE", "CBP", "MPM-ACE")
    }
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025, cohort_summary=cohort,
    ))
    rk = _run(compute_care_gap_priority_ranking(agg, top_n=3))
    assert rk.n_actions == 3


def test_ranking_difficulty_label_attached_per_measure():
    """Every emitted action must carry a closure_difficulty in the
    {easy, moderate, hard, very_hard} closed set."""
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 200, "denominator": 1000}},
    ))
    rk = _run(compute_care_gap_priority_ranking(agg))
    assert rk.actions[0].closure_difficulty in (
        "easy", "moderate", "hard", "very_hard",
    )


def test_ranking_intervention_text_non_empty():
    agg = _run(compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={"BCS": {"numerator": 200, "denominator": 1000}},
    ))
    rk = _run(compute_care_gap_priority_ranking(agg))
    assert rk.actions[0].suggested_intervention.strip() != ""


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_quality_stars_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "quality_stars" in BUNDLES
    assert set(BUNDLES["quality_stars"]) == {
        "compute_quality_measures_aggregate",
        "compute_stars_rating_forecast",
        "compute_care_gap_priority_ranking",
    }


def test_quality_stars_scopes_declared():
    from mcp_server.scopes import BUNDLE_SCOPES
    assert "quality_stars" in BUNDLE_SCOPES
    assert "patient/Patient.rs" in BUNDLE_SCOPES["quality_stars"]
