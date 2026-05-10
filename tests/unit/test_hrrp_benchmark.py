"""Phase 17.S - HRRP benchmark tests."""

from __future__ import annotations

from a2a_agent.hrrp_benchmark import (
    HRRPBenchmarkReport, benchmark_against_hrrp, render_hrrp_md,
)


def test_per_bin_has_five_rows():
    report = benchmark_against_hrrp()
    assert report.n_bins == 5


def test_all_bins_within_published_ci():
    """The W1 calibration was anchored on HRRP rates, so the gap
    should be < 1 pp on every bin."""
    report = benchmark_against_hrrp()
    assert report.overall_within_published_ci
    assert report.overall_max_abs_gap < 0.01


def test_per_subgroup_empty_when_no_rates_supplied():
    report = benchmark_against_hrrp()
    assert report.per_subgroup == []


def test_per_subgroup_aligns_with_literature_when_supplied():
    report = benchmark_against_hrrp(
        trustedrisk_subgroup_rates={
            "race": {"black": 0.165, "white": 0.146},
            "insurance": {"medicare": 0.158},
        },
    )
    races = {r.subgroup_value for r in report.per_subgroup
             if r.subgroup_axis == "race"}
    assert "black" in races
    black_row = next(
        r for r in report.per_subgroup
        if r.subgroup_value == "black"
    )
    assert black_row.literature_rate > 0
    assert black_row.abs_gap >= 0


def test_render_md_includes_per_bin_table():
    report = benchmark_against_hrrp()
    md = render_hrrp_md(report)
    assert "## Per-LACE bin" in md
    for label in ("lace_0_2", "lace_3_5", "lace_6_9",
                   "lace_10_12", "lace_13_19"):
        assert label in md


def test_render_md_includes_subgroup_table_when_data_supplied():
    report = benchmark_against_hrrp(
        trustedrisk_subgroup_rates={
            "race": {"black": 0.18}, "insurance": {"medicaid": 0.20},
        },
    )
    md = render_hrrp_md(report)
    assert "## Per-subgroup vs literature" in md
    assert "black" in md
    assert "medicaid" in md


def test_round_trip_through_pydantic():
    report = benchmark_against_hrrp()
    payload = report.model_dump(mode="json")
    rebuilt = HRRPBenchmarkReport.model_validate(payload)
    assert rebuilt.n_bins == report.n_bins
