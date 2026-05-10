"""Regression suite: assert TrustedRisk output on the 3 demo fixtures matches expected_signals.json.

Per OUTPUT_SCHEMA_demo-cohort-casting.md §3 (strict + loose tolerance modes) and
design doc §5.6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
EXPECTED_SIGNALS_PATH = FIXTURES_DIR / "expected_signals.json"


def _load_expected() -> dict | None:
    if not EXPECTED_SIGNALS_PATH.exists():
        return None
    with EXPECTED_SIGNALS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def expected_signals():
    exp = _load_expected()
    if exp is None:
        pytest.skip("expected_signals.json not staged; run `make port` first")
    return exp


@pytest.fixture(scope="module")
def tolerance_mode(expected_signals):
    return expected_signals.get("tolerance_mode", "strict")


def _load_bundle(profile_id: str) -> dict:
    filename = {
        "patient_01_clean": "patient_01_clean.json",
        "patient_02_abstain": "patient_02_abstain.json",
        "patient_03_complex": "patient_03_complex.json",
    }[profile_id]
    fp = FIXTURES_DIR / filename
    if not fp.exists():
        pytest.skip(f"{filename} not staged; run `make port` first")
    with fp.open(encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# These are skeleton tests -- they verify the fixtures LOAD correctly and
# that the expected_signals.json has the right shape. End-to-end invocation
# of the MCP tools requires a running FHIR server + the calibration artifacts
# staged, which is a separate integration concern.
# ─────────────────────────────────────────────────────────────────────

def test_expected_signals_has_three_profiles(expected_signals):
    for key in ("patient_01_clean", "patient_02_abstain", "patient_03_complex"):
        assert key in expected_signals, f"expected_signals.json missing {key}"


def test_patient_01_bundle_loads():
    bundle = _load_bundle("patient_01_clean")
    assert bundle.get("resourceType") == "Bundle"
    assert len(bundle.get("entry", [])) > 0


def test_patient_02_bundle_loads():
    bundle = _load_bundle("patient_02_abstain")
    assert bundle.get("resourceType") == "Bundle"


def test_patient_03_bundle_loads():
    bundle = _load_bundle("patient_03_complex")
    assert bundle.get("resourceType") == "Bundle"


def test_patient_02_expected_abstain_shape(expected_signals):
    """Regardless of tolerance_mode, patient_02 must expect an abstain trigger."""
    p02 = expected_signals["patient_02_abstain"]
    abstain = p02.get("expected_abstain")
    assert abstain is not None, "patient_02 must expect abstain"
    assert abstain.get("triggered") is True


def test_patient_01_expected_no_abstain(expected_signals):
    p01 = expected_signals["patient_01_clean"]
    assert p01.get("expected_abstain") is None


def test_patient_03_expected_no_abstain(expected_signals):
    p03 = expected_signals["patient_03_complex"]
    assert p03.get("expected_abstain") is None


def test_tolerance_mode_known(tolerance_mode):
    assert tolerance_mode in ("strict", "loose"), f"unknown tolerance_mode: {tolerance_mode!r}"


# ─────────────────────────────────────────────────────────────────────
# End-to-end tests (skipped until MCP server + FHIR integration is wired).
# When available, these would:
#   1. Load the fixture Bundle onto a FHIR server.
#   2. POST to TrustedRisk MCP's compute_readmission_risk with X-Patient-ID.
#   3. Assert probability_mean / ci_width / abstain triggers match expected.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skip(reason="End-to-end test requires running MCP server + FHIR instance")
def test_e2e_patient_01_produces_recommendation(expected_signals):
    pass


@pytest.mark.integration
@pytest.mark.skip(reason="End-to-end test requires running MCP server + FHIR instance")
def test_e2e_patient_02_triggers_abstain(expected_signals):
    pass


@pytest.mark.integration
@pytest.mark.skip(reason="End-to-end test requires running MCP server + FHIR instance")
def test_e2e_patient_03_recommends_snf_or_home_with_care(expected_signals):
    pass
