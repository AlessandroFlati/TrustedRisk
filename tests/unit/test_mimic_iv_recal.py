"""Phase 11.2 -- MIMIC-IV demo recalibration artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT = REPO_ROOT / "data" / "mimic_iv_recalibration.json"
DOC = REPO_ROOT / "docs" / "validation" / "MIMIC_IV_RECAL.md"


def test_recalibration_artifact_present():
    assert ARTIFACT.exists(), (
        "Run scripts/mimic_iv_recalibrate.py first."
    )


def test_recalibration_artifact_has_per_bin_posteriors():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    posteriors = raw.get("per_bin")
    assert isinstance(posteriors, dict)
    expected_bins = {"lace_0_2", "lace_3_5", "lace_6_9",
                          "lace_10_12", "lace_13_19"}
    assert set(posteriors.keys()) == expected_bins
    for name, body in posteriors.items():
        assert "p_mean" in body and 0.0 < body["p_mean"] < 1.0
        assert "n" in body and body["n"] >= 0


def test_recalibration_metrics_reasonable():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    metrics = raw["metrics_overall"]
    # ECE under the preferred gate (0.05) -- same gate the W1 calibration
    # had to pass at promotion.
    assert metrics["ece"] < 0.05, (
        f"MIMIC ECE {metrics['ece']:.4f} above 0.05 preferred gate"
    )
    # AUROC at least matches random; in practice ~ 0.60+
    assert metrics["auroc"] > 0.55


def test_doc_lists_w1_baseline():
    """The narrative comparison vs W1 must be present in the markdown."""
    if not DOC.exists():
        pytest.skip("MIMIC_IV_RECAL.md not yet written")
    text = DOC.read_text(encoding="utf-8")
    assert "W1" in text
    assert "0.0078" in text  # W1 published ECE


def test_recalibration_cohort_non_trivial():
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert raw["cohort_n"] >= 100
