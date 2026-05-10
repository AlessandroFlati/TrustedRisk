"""Phase 11.1 -- conformal prediction surface in compute_readmission_risk."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp_server.tools.readmission_risk import compute_readmission_risk


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run(coro):
    return asyncio.run(coro)


def _bundle(los_days: float, n_conditions: int, n_ed: int,
                   is_acute: bool = True, observations: int = 0):
    """Minimal FHIR Bundle stub for the LACE extractor."""
    entries = []
    if is_acute or los_days > 0:
        entries.append({
            "resource": {
                "resourceType": "Encounter",
                "class": {"code": "IMP"},
                "period": {
                    "start": "2026-04-01T08:00:00Z",
                    "end": (f"2026-04-{int(los_days) + 1:02d}"
                                "T08:00:00Z"),
                },
            },
        })
    for i in range(n_ed):
        entries.append({
            "resource": {
                "resourceType": "Encounter",
                "class": {"code": "EMER"},
            },
        })
    for i in range(n_conditions):
        entries.append({
            "resource": {
                "resourceType": "Condition",
                "id": f"cond-{i}",
                "code": {"text": f"cond {i}"},
            },
        })
    for i in range(observations):
        entries.append({
            "resource": {
                "resourceType": "Observation", "id": f"obs-{i}",
            },
        })
    return {"resourceType": "Bundle", "entry": entries}


def _stub_module(monkeypatch, bundle):
    from mcp_server.tools import readmission_risk as rr

    async def stub_fetch(pid):
        return bundle

    async def stub_resolve(explicit):
        return explicit or "pt-conformal"

    monkeypatch.setattr(rr, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(rr, "resolve_patient_id", stub_resolve)
    # Reset the conformal cache between tests to isolate env-var paths
    monkeypatch.setattr(rr, "_CONFORMAL", None)
    monkeypatch.setattr(rr, "_CONFORMAL_LOAD_ATTEMPTED", False)


# ─────────────────────────────────────────────────────────────────────
# When artefact is present
# ─────────────────────────────────────────────────────────────────────

def test_risk_carries_conformal_interval_when_artifact_present(monkeypatch):
    art = REPO_ROOT / "data" / "conformal_readmission.json"
    if not art.exists():
        pytest.skip("data/conformal_readmission.json not present yet")
    monkeypatch.setenv("TRUSTEDRISK_CONFORMAL_PATH", str(art))
    _stub_module(monkeypatch, _bundle(los_days=5, n_conditions=4, n_ed=2))
    risk = _run(compute_readmission_risk(patient_id="pt-conformal"))

    assert risk.conformal_target_coverage is not None
    assert 0.0 <= risk.conformal_target_coverage <= 1.0
    assert risk.conformal_interval_lower is not None
    assert risk.conformal_interval_upper is not None
    assert (risk.conformal_interval_lower
            <= risk.probability_mean
            <= risk.conformal_interval_upper)


def test_risk_carries_conformal_prediction_set_when_artifact_present(monkeypatch):
    art = REPO_ROOT / "data" / "conformal_readmission.json"
    if not art.exists():
        pytest.skip("data/conformal_readmission.json not present yet")
    monkeypatch.setenv("TRUSTEDRISK_CONFORMAL_PATH", str(art))
    _stub_module(monkeypatch, _bundle(los_days=5, n_conditions=4, n_ed=2))
    risk = _run(compute_readmission_risk(patient_id="pt-conformal"))

    assert risk.conformal_prediction_set is not None
    assert set(risk.conformal_prediction_set) <= {0, 1}


def test_conformal_interval_clipped_to_unit_box(monkeypatch):
    """Lower bound floors to 0, upper ceils to 1."""
    art = REPO_ROOT / "data" / "conformal_readmission.json"
    if not art.exists():
        pytest.skip("data/conformal_readmission.json not present yet")
    monkeypatch.setenv("TRUSTEDRISK_CONFORMAL_PATH", str(art))
    _stub_module(monkeypatch, _bundle(los_days=1, n_conditions=1, n_ed=0))
    risk = _run(compute_readmission_risk(patient_id="pt-low"))
    assert 0.0 <= risk.conformal_interval_lower <= 1.0  # type: ignore[arg-type]
    assert 0.0 <= risk.conformal_interval_upper <= 1.0  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────
# When artefact is missing -- graceful fallback
# ─────────────────────────────────────────────────────────────────────

def test_risk_carries_no_conformal_fields_when_artifact_missing(
    monkeypatch, tmp_path,
):
    bogus = tmp_path / "absent.json"
    monkeypatch.setenv("TRUSTEDRISK_CONFORMAL_PATH", str(bogus))
    _stub_module(monkeypatch, _bundle(los_days=5, n_conditions=4, n_ed=2))
    risk = _run(compute_readmission_risk(patient_id="pt-noart"))

    assert risk.conformal_interval_lower is None
    assert risk.conformal_interval_upper is None
    assert risk.conformal_prediction_set is None
    assert risk.conformal_target_coverage is None


def test_risk_carries_no_conformal_fields_when_artifact_corrupt(
    monkeypatch, tmp_path,
):
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("TRUSTEDRISK_CONFORMAL_PATH", str(bad))
    _stub_module(monkeypatch, _bundle(los_days=5, n_conditions=4, n_ed=2))
    risk = _run(compute_readmission_risk(patient_id="pt-bad"))
    assert risk.conformal_interval_lower is None


# ─────────────────────────────────────────────────────────────────────
# Artefact contract (validates the calibration script's output shape)
# ─────────────────────────────────────────────────────────────────────

def test_calibration_artifact_has_expected_shape():
    art = REPO_ROOT / "data" / "conformal_readmission.json"
    if not art.exists():
        pytest.skip("data/conformal_readmission.json not present yet")
    raw = json.loads(art.read_text(encoding="utf-8"))
    assert "target_coverage" in raw
    for track in ("binary_one_minus_p", "abs_residual"):
        assert track in raw
        assert "quantile_threshold" in raw[track]
        assert "empirical_coverage_validation" in raw[track]


def test_calibration_validation_coverage_within_target_band():
    art = REPO_ROOT / "data" / "conformal_readmission.json"
    if not art.exists():
        pytest.skip("data/conformal_readmission.json not present yet")
    raw = json.loads(art.read_text(encoding="utf-8"))
    target = float(raw["target_coverage"])
    for track in ("binary_one_minus_p", "abs_residual"):
        emp = float(raw[track]["empirical_coverage_validation"])
        # Allow ± 0.04 slack on n=2500 binomial std
        assert target - 0.04 <= emp <= 1.0, (
            f"{track}: emp {emp} outside band around {target}"
        )
