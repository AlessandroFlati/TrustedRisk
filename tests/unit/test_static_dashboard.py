"""Phase 14.11 N2 -- Static dashboard tests."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD = REPO_ROOT / "docs" / "dashboard.html"


def test_dashboard_artifact_present():
    if not DASHBOARD.exists():
        pytest.skip("Run scripts/render_static_dashboard.py first")
    assert DASHBOARD.stat().st_size > 1000


def test_dashboard_has_calibration_section():
    if not DASHBOARD.exists():
        pytest.skip("Run scripts/render_static_dashboard.py first")
    text = DASHBOARD.read_text(encoding="utf-8")
    assert "Calibration metrics" in text
    assert "W1 internal calibration" in text


def test_dashboard_lists_all_seven_sections():
    if not DASHBOARD.exists():
        pytest.skip("Run scripts/render_static_dashboard.py first")
    text = DASHBOARD.read_text(encoding="utf-8")
    for h in (
        "Calibration metrics",
        "Conformal calibration thresholds",
        "Adversarial robustness",
        "Performance benchmarks v10",
        "MedQA-USMLE-style bench",
        "Scenario counterfactuals",
        "Cloud Run deployment dry-run",
    ):
        assert h in text, f"missing section {h!r}"


def test_dashboard_has_no_javascript():
    """Static HTML -- no <script> tags allowed."""
    if not DASHBOARD.exists():
        pytest.skip("Run scripts/render_static_dashboard.py first")
    text = DASHBOARD.read_text(encoding="utf-8").lower()
    assert "<script" not in text
