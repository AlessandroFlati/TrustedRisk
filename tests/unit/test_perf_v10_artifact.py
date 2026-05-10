"""Phase 11.7 -- performance benchmark artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT = REPO_ROOT / "docs" / "performance" / "v10_benchmarks.json"


def test_v10_benchmark_artifact_present():
    assert ARTIFACT.exists(), (
        "Run scripts/perf_benchmark_v10.py first."
    )


def test_v10_benchmark_has_three_layers():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for key in ("per_tool_phase_10_11", "per_scenario",
                    "federation_concurrency"):
        assert key in raw, f"Missing layer: {key}"


def test_v10_per_tool_covers_phase_10_11_surface():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    labels = {row["label"] for row in raw["per_tool_phase_10_11"]}
    expected = {
        "compute_quality_measures_aggregate",
        "compute_stars_rating_forecast",
        "compute_care_gap_priority_ranking",
        "compute_syndromic_surveillance",
        "compute_vaccine_reminder_cohort",
        "compute_outbreak_heatmap",
        "compute_denial_letter_parse",
        "compute_appeal_escalation_path",
        "compute_ecg_qt_analyzer",
        "compute_dicom_sr_ingest",
        "plan_tool_use",
    }
    assert expected <= labels


def test_v10_p99_floor_under_100ms_for_pure_python_tools():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for row in raw["per_tool_phase_10_11"]:
        assert row["p99"] < 100.0, (
            f"{row['label']}: p99 {row['p99']} ms exceeds 100 ms floor"
        )


def test_v10_no_errors_in_per_tool_layer():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for row in raw["per_tool_phase_10_11"]:
        assert row["errors"] == 0, (
            f"{row['label']}: {row['errors']} error(s)"
        )


def test_v10_scenarios_complete_under_500ms_at_p99():
    """All 5 scenarios must finish at p99 in under 500 ms -- keeps the
    end-to-end demo interactive."""
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for row in raw["per_scenario"]:
        assert row["p99"] < 500.0, (
            f"{row['label']}: p99 {row['p99']} ms exceeds 500 ms"
        )


def test_v10_federation_concurrency_no_errors():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fed = raw["federation_concurrency"]
    assert fed["errors"] == 0
    assert fed["n"] >= 50
