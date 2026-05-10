"""SCALE-1 unit tests for the Synthea evaluator (no JAR / no Java needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from a2a_agent.synthea_evaluator import (
    derive_readmission_label,
    evaluate_synthea_cohort,
)


def _bundle(*, encounters: list[dict]) -> dict:
    """Build a minimal FHIR Bundle from a list of Encounter resources."""
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [{"resource": e} for e in encounters],
    }


def _enc(*, kind: str, start: str, end: str) -> dict:
    """Build an Encounter -- kind ∈ {'IMP', 'AMB'}."""
    return {
        "resourceType": "Encounter",
        "class": {"code": kind, "system": "http://hl7.org/v3/ActCode"},
        "period": {"start": start, "end": end},
    }


# ─────────────────────── Outcome label derivation ───────────────────────

def test_no_inpatient_returns_none():
    bundle = _bundle(encounters=[
        _enc(kind="AMB", start="2024-01-01T08:00:00Z",
                end="2024-01-01T09:00:00Z"),
    ])
    assert derive_readmission_label(bundle) is None


def test_single_inpatient_no_subsequent_returns_zero():
    bundle = _bundle(encounters=[
        _enc(kind="IMP", start="2024-01-01T08:00:00Z",
                end="2024-01-04T09:00:00Z"),
    ])
    assert derive_readmission_label(bundle) == 0


def test_readmission_within_30d_returns_one():
    bundle = _bundle(encounters=[
        _enc(kind="IMP", start="2024-01-01T08:00:00Z",
                end="2024-01-04T09:00:00Z"),
        _enc(kind="IMP", start="2024-01-15T08:00:00Z",
                end="2024-01-18T09:00:00Z"),
    ])
    assert derive_readmission_label(bundle, horizon_days=30) == 1


def test_readmission_outside_30d_returns_zero():
    bundle = _bundle(encounters=[
        _enc(kind="IMP", start="2024-01-01T08:00:00Z",
                end="2024-01-04T09:00:00Z"),
        _enc(kind="IMP", start="2024-04-01T08:00:00Z",
                end="2024-04-04T09:00:00Z"),
    ])
    assert derive_readmission_label(bundle, horizon_days=30) == 0


def test_emergency_encounter_counts_as_readmission():
    bundle = _bundle(encounters=[
        _enc(kind="IMP", start="2024-01-01T08:00:00Z",
                end="2024-01-04T09:00:00Z"),
        _enc(kind="EMER", start="2024-01-10T08:00:00Z",
                end="2024-01-10T18:00:00Z"),
    ])
    assert derive_readmission_label(bundle) == 1


def test_outpatient_followup_does_not_count():
    """Ambulatory encounters within the window must NOT count as readmissions."""
    bundle = _bundle(encounters=[
        _enc(kind="IMP", start="2024-01-01T08:00:00Z",
                end="2024-01-04T09:00:00Z"),
        _enc(kind="AMB", start="2024-01-15T08:00:00Z",
                end="2024-01-15T09:00:00Z"),
    ])
    assert derive_readmission_label(bundle) == 0


def test_index_is_most_recent_inpatient_encounter():
    """When multiple inpatient encounters exist, the most recent one is the
    'index' against which we measure readmission."""
    bundle = _bundle(encounters=[
        _enc(kind="IMP", start="2023-06-01T08:00:00Z",
                end="2023-06-05T09:00:00Z"),
        _enc(kind="IMP", start="2024-01-01T08:00:00Z",
                end="2024-01-04T09:00:00Z"),
        # readmission within 30d of the SECOND inpatient
        _enc(kind="IMP", start="2024-01-20T08:00:00Z",
                end="2024-01-22T09:00:00Z"),
    ])
    assert derive_readmission_label(bundle) == 1


def test_malformed_dates_ignored():
    bundle = _bundle(encounters=[
        {"resourceType": "Encounter",
         "class": {"code": "IMP"},
         "period": {"start": "not-a-date"}},
    ])
    # No usable period -> no inpatient discovered -> label is None
    assert derive_readmission_label(bundle) is None


# ─────────────────────── Cohort evaluation ───────────────────────

@pytest.fixture
def coefficients_path() -> Path:
    """Use the real shipped coefficients."""
    return Path("data/coefficients.json")


def test_evaluate_empty_cohort(coefficients_path):
    if not coefficients_path.exists():
        pytest.skip("coefficients.json not present -- run from repo root")
    r = evaluate_synthea_cohort(
        [], cohort_label="empty",
        coefficients_path=coefficients_path,
    )
    assert r.n_patients == 0
    assert r.n_with_readmission_label == 0
    assert r.ece is None
    assert r.auroc is None


def test_evaluate_synthetic_cohort_produces_metrics(coefficients_path):
    """Build a small synthetic cohort with known LACE features + outcomes."""
    if not coefficients_path.exists():
        pytest.skip("coefficients.json not present")

    bundles = []
    # 5 high-risk patients (long LOS + many ED visits) -> high LACE
    for i in range(5):
        encs = [_enc(kind="IMP",
                          start="2024-01-01T08:00:00Z",
                          end="2024-01-12T09:00:00Z")]
        # Simulate readmission for 3 of them
        if i < 3:
            encs.append(_enc(kind="IMP",
                                start="2024-01-20T08:00:00Z",
                                end="2024-01-22T09:00:00Z"))
        bundles.append(_bundle(encounters=encs))

    # 5 low-risk patients (short LOS, no readmissions)
    for _ in range(5):
        encs = [_enc(kind="IMP",
                          start="2024-02-01T08:00:00Z",
                          end="2024-02-03T09:00:00Z")]
        bundles.append(_bundle(encounters=encs))

    r = evaluate_synthea_cohort(
        bundles, cohort_label="synthetic_test_n10",
        coefficients_path=coefficients_path,
    )
    assert r.n_patients == 10
    # Each bundle has at least one inpatient -> labels derivable
    assert r.n_with_readmission_label == 10
    assert r.n_readmitted == 3
    assert 0.0 <= r.base_rate_readmission <= 1.0
    assert r.ece is not None
    # Calibration table populated (5 bins; some may be empty, but at least 1 non-empty)
    assert len(r.calibration_buckets) >= 1


def test_brier_score_in_unit_interval(coefficients_path):
    if not coefficients_path.exists():
        pytest.skip("coefficients.json not present")

    bundles = [
        _bundle(encounters=[
            _enc(kind="IMP",
                  start="2024-01-01T08:00:00Z",
                  end="2024-01-05T09:00:00Z"),
        ]) for _ in range(20)
    ]
    r = evaluate_synthea_cohort(
        bundles, cohort_label="x",
        coefficients_path=coefficients_path,
    )
    if r.brier_score is not None:
        assert 0.0 <= r.brier_score <= 1.0


def test_references_include_synthea_and_lace(coefficients_path):
    if not coefficients_path.exists():
        pytest.skip("coefficients.json not present")
    r = evaluate_synthea_cohort(
        [], cohort_label="x",
        coefficients_path=coefficients_path,
    )
    refs = " ".join(r.references)
    assert "Synthea" in refs or "Walonoski" in refs
    assert "LACE" in refs or "Walraven" in refs


def test_missing_coefficients_raises():
    with pytest.raises(FileNotFoundError):
        evaluate_synthea_cohort(
            [], coefficients_path=Path("data/does_not_exist.json"),
        )
