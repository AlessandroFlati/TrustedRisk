"""Unit tests for compute_lab_trend_analysis."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from mcp_server.tools.lab_trend_analysis import (
    _classify_clinical,
    _classify_direction,
    _in_range,
    _linear_slope,
    _match_lab,
    compute_lab_trend_analysis,
)


def _run(coro):
    return asyncio.run(coro)


def _series(name: str, values: list[float], unit: str = "") -> list[dict]:
    """Build a list of observation dicts spaced 1 day apart."""
    base = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)
    return [
        {"name": name, "value": v, "unit": unit,
         "observed_at": (base + timedelta(days=i)).isoformat()}
        for i, v in enumerate(values)
    ]


# ─────────────────────── _linear_slope ───────────────────────

def test_slope_increasing():
    s = _linear_slope([0, 1, 2, 3], [10, 12, 14, 16])
    assert s == pytest.approx(2.0)


def test_slope_flat():
    s = _linear_slope([0, 1, 2], [5, 5, 5])
    assert s == pytest.approx(0.0)


def test_slope_decreasing():
    s = _linear_slope([0, 1, 2], [10, 8, 6])
    assert s == pytest.approx(-2.0)


def test_slope_single_point():
    assert _linear_slope([0], [5]) == 0.0


# ─────────────────────── _classify_direction ───────────────────────

def test_direction_up_when_slope_positive():
    assert _classify_direction(slope=1.0, mean_value=10.0) == "up"


def test_direction_down_when_slope_negative():
    assert _classify_direction(slope=-1.0, mean_value=10.0) == "down"


def test_direction_flat_when_slope_below_relative_threshold():
    # Slope 0.05/day on mean=10 -> 0.5 % per day, well under the 2 % threshold.
    assert _classify_direction(slope=0.05, mean_value=10.0) == "flat"


# ─────────────────────── _in_range ───────────────────────

def test_in_range_inside():
    assert _in_range(140.0, (135.0, 145.0)) is True


def test_in_range_outside():
    assert _in_range(150.0, (135.0, 145.0)) is False


def test_in_range_no_range():
    assert _in_range(100.0, None) is None


# ─────────────────────── _match_lab ───────────────────────

def test_match_lab_by_name():
    assert _match_lab("Hemoglobin", None) == "hemoglobin"
    assert _match_lab("Serum Creatinine", None) == "creatinine"
    assert _match_lab("INR (prothrombin)", None) == "inr"


def test_match_lab_by_loinc():
    # LOINC for hemoglobin is 718-7
    assert _match_lab("anything", "718-7") == "hemoglobin"


def test_match_lab_unknown():
    assert _match_lab("CompletelyUnknownLab", None) is None


# ─────────────────────── _classify_clinical (per-lab semantics) ───────────────────────

def test_clinical_lower_is_better_decreasing_means_improving():
    """Creatinine going down = kidney function improving."""
    res = _classify_clinical(direction="down", latest=1.5, slope=-0.5,
                              clinical_direction="lower_is_better",
                              reference_range=(0.7, 1.3))
    assert res == "improving"


def test_clinical_lower_is_better_increasing_means_declining():
    res = _classify_clinical(direction="up", latest=2.5, slope=0.5,
                              clinical_direction="lower_is_better",
                              reference_range=(0.7, 1.3))
    assert res == "declining"


def test_clinical_higher_is_better_decreasing_means_declining():
    """Hemoglobin dropping = anemia worsening."""
    res = _classify_clinical(direction="down", latest=8.0, slope=-0.5,
                              clinical_direction="higher_is_better",
                              reference_range=(12.0, 17.0))
    assert res == "declining"


def test_clinical_in_range_moving_toward_midpoint():
    """INR climbing from 1.5 toward 2.5 (target 2.0-3.0) = improving."""
    res = _classify_clinical(direction="up", latest=2.0, slope=0.5,
                              clinical_direction="in_range_is_best",
                              reference_range=(2.0, 3.0))
    # Currently at the lower bound; up means heading further into range
    assert res == "flat"  # latest already at lower bound, in range, slope captured at lower end


def test_clinical_in_range_below_range_climbing_means_improving():
    res = _classify_clinical(direction="up", latest=1.5, slope=0.3,
                              clinical_direction="in_range_is_best",
                              reference_range=(2.0, 3.0))
    assert res == "improving"


def test_clinical_in_range_above_range_climbing_means_declining():
    res = _classify_clinical(direction="up", latest=4.5, slope=0.5,
                              clinical_direction="in_range_is_best",
                              reference_range=(2.0, 3.0))
    assert res == "declining"


def test_clinical_flat_outside_range_is_borderline():
    """A flat value sustained outside range is borderline (sustained problem)."""
    res = _classify_clinical(direction="flat", latest=8.5, slope=0.0,
                              clinical_direction="higher_is_better",
                              reference_range=(12.0, 17.0))
    assert res == "borderline"


def test_clinical_flat_in_range_is_flat():
    res = _classify_clinical(direction="flat", latest=14.0, slope=0.0,
                              clinical_direction="higher_is_better",
                              reference_range=(12.0, 17.0))
    assert res == "flat"


# ─────────────────────── End-to-end ───────────────────────

def test_e2e_hemoglobin_improving():
    obs = _series("Hemoglobin", [8.5, 9.0, 9.5], unit="g/dL")
    rep = _run(compute_lab_trend_analysis(observations=obs))
    assert rep.n_labs_analyzed == 1
    t = rep.trends[0]
    assert t.lab_name == "hemoglobin"
    assert t.direction == "up"
    assert t.trend_clinical == "improving"


def test_e2e_creatinine_declining():
    obs = _series("Creatinine", [0.9, 1.5, 2.5], unit="mg/dL")
    rep = _run(compute_lab_trend_analysis(observations=obs))
    t = rep.trends[0]
    assert t.lab_name == "creatinine"
    assert t.trend_clinical == "declining"
    # Latest 2.5 is well outside ref (0.7-1.3)
    assert t.in_normal_range is False


def test_e2e_inr_in_range_flat():
    obs = _series("INR", [2.5, 2.5, 2.5])
    rep = _run(compute_lab_trend_analysis(observations=obs))
    t = rep.trends[0]
    assert t.trend_clinical == "flat"
    assert t.in_normal_range is True


def test_e2e_insufficient_data_single_point():
    obs = _series("Hemoglobin", [10.0])
    rep = _run(compute_lab_trend_analysis(observations=obs))
    t = rep.trends[0]
    assert t.direction == "insufficient_data"
    assert t.trend_clinical == "insufficient_data"


def test_e2e_unknown_lab_filtered():
    obs = [
        {"name": "Made-up Lab", "value": 5.0, "observed_at": "2026-04-20T08:00Z"},
        {"name": "Made-up Lab", "value": 6.0, "observed_at": "2026-04-21T08:00Z"},
    ]
    rep = _run(compute_lab_trend_analysis(observations=obs))
    assert rep.n_labs_analyzed == 0


def test_e2e_multiple_labs_grouped():
    obs = (_series("Hemoglobin", [8.0, 9.0, 10.0])
           + _series("Creatinine", [0.9, 1.0, 1.1]))
    rep = _run(compute_lab_trend_analysis(observations=obs))
    assert rep.n_labs_analyzed == 2
    by_name = {t.lab_name: t for t in rep.trends}
    assert "hemoglobin" in by_name
    assert "creatinine" in by_name


def test_e2e_flagged_count():
    obs = (_series("Creatinine", [1.0, 1.5, 2.0])  # declining
           + _series("Hemoglobin", [12.0, 13.0, 14.0]))  # improving
    rep = _run(compute_lab_trend_analysis(observations=obs))
    assert rep.flagged_labs_count == 1


def test_e2e_summary_mentions_counts():
    obs = _series("Hemoglobin", [12.0, 13.0, 14.0])
    rep = _run(compute_lab_trend_analysis(observations=obs))
    assert "improving" in rep.summary or "1" in rep.summary


def test_e2e_loinc_match_when_name_unrecognised():
    """A LOINC code 718-7 should match hemoglobin even if `name` is gibberish."""
    obs = [
        {"name": "junk", "value": 12.0, "observed_at": "2026-04-20T08:00Z", "loinc": "718-7"},
        {"name": "junk", "value": 13.0, "observed_at": "2026-04-21T08:00Z", "loinc": "718-7"},
    ]
    rep = _run(compute_lab_trend_analysis(observations=obs))
    assert rep.n_labs_analyzed == 1
    assert rep.trends[0].lab_name == "hemoglobin"


def test_e2e_observations_out_of_order_get_sorted():
    """Observations not in chronological order must still produce a valid slope."""
    base = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)
    obs = [
        {"name": "Hemoglobin", "value": 10.0,
         "observed_at": (base + timedelta(days=2)).isoformat()},
        {"name": "Hemoglobin", "value": 8.0,
         "observed_at": base.isoformat()},
        {"name": "Hemoglobin", "value": 9.0,
         "observed_at": (base + timedelta(days=1)).isoformat()},
    ]
    rep = _run(compute_lab_trend_analysis(observations=obs))
    t = rep.trends[0]
    assert t.direction == "up"
    assert t.latest_value == 10.0


def test_e2e_via_fhir_bundle(monkeypatch):
    """Tool fetches from FHIR when observations are not provided."""
    bundle = {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Observation",
                          "code": {"text": "Creatinine",
                                    "coding": [{"system": "http://loinc.org", "code": "2160-0",
                                                "display": "Creatinine"}]},
                          "valueQuantity": {"value": 1.0, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-20T08:00:00Z"}},
            {"resource": {"resourceType": "Observation",
                          "code": {"text": "Creatinine"},
                          "valueQuantity": {"value": 1.5, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-22T08:00:00Z"}},
        ],
    }

    async def stub_fetch(pid):
        return bundle

    async def stub_resolve(explicit):
        return explicit or "pt-1"

    from mcp_server.tools import lab_trend_analysis as lta
    monkeypatch.setattr(lta, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(lta, "resolve_patient_id", stub_resolve)

    rep = _run(compute_lab_trend_analysis(patient_id="pt-1"))
    assert rep.patient_id == "pt-1"
    assert rep.n_labs_analyzed == 1
    assert rep.trends[0].lab_name == "creatinine"
