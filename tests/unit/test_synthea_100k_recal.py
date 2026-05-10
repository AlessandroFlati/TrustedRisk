"""Phase 12.1 -- Synthea-100k recalibration artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT = REPO_ROOT / "data" / "synthea_100k_recalibration.json"
DOC = REPO_ROOT / "docs" / "validation" / "SYNTHEA_100K.md"


def _load_or_skip():
    if not ARTIFACT.exists():
        pytest.skip("Run scripts/synthea_100k_recalibrate.py first")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_present_and_well_formed():
    raw = _load_or_skip()
    assert raw["cohort_n"] == 100_000
    for key in ("per_bin", "metrics_overall", "comparison"):
        assert key in raw


def test_per_bin_posteriors_cover_all_5_bins():
    raw = _load_or_skip()
    assert set(raw["per_bin"].keys()) == {
        "lace_0_2", "lace_3_5", "lace_6_9",
        "lace_10_12", "lace_13_19",
    }


def test_metrics_under_preferred_gates():
    raw = _load_or_skip()
    m = raw["metrics_overall"]
    assert m["ece"] < 0.05, f"ECE {m['ece']} ≥ 0.05 preferred gate"
    assert m["auroc"] > 0.55


def test_per_bin_minimum_n_at_100k_cohort():
    """Every bin should carry at least n=200 even on the smallest tail."""
    raw = _load_or_skip()
    for name, body in raw["per_bin"].items():
        assert body["n"] >= 200, (
            f"Bin {name!r}: n={body['n']} (< 200, calibration brittle)"
        )


def test_comparison_block_lists_all_three_predecessor_cohorts():
    raw = _load_or_skip()
    cmp = raw["comparison"]
    assert {"w1_published", "synthea_10k_validation", "mimic_iv_demo"} <= set(cmp)


def test_doc_contains_w1_baseline_row():
    if not DOC.exists():
        pytest.skip("SYNTHEA_100K.md not yet written")
    text = DOC.read_text(encoding="utf-8")
    assert "0.0078" in text  # W1 ECE
    assert "MIMIC-IV demo" in text
