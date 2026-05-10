"""Phase 12.8 C1 -- Cloud Run deploy dry-run validator tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_SH = REPO_ROOT / "scripts" / "deploy_cloudrun.sh"
ARTIFACT = REPO_ROOT / "docs" / "deployment" / "deploy_dry_run.json"


def test_deploy_script_lists_at_least_15_services():
    text = DEPLOY_SH.read_text(encoding="utf-8")
    # Count lines in the SERVICES=() block that contain a pipe
    import re
    m = re.search(r"SERVICES=\(\s*([\s\S]*?)\)", text)
    assert m, "SERVICES=(...) block not found"
    body = m.group(1)
    n = sum(
        1 for ln in body.splitlines()
        if "|" in ln and not ln.strip().startswith("#")
    )
    assert n >= 15, f"deploy_cloudrun.sh lists only {n} services"


def test_deploy_script_lists_phase_10_11_specialists():
    text = DEPLOY_SH.read_text(encoding="utf-8")
    for must_have in (
        "trustedrisk-quality", "trustedrisk-pophealth",
        "trustedrisk-appeals", "trustedrisk-multimodal",
    ):
        assert must_have in text, f"deploy script missing {must_have}"


def test_deploy_script_includes_cds_hooks():
    text = DEPLOY_SH.read_text(encoding="utf-8")
    assert "trustedrisk-cds-hooks" in text


def test_dry_run_artifact_exists():
    if not ARTIFACT.exists():
        pytest.skip("Run scripts/deploy_dry_run.py first")
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert raw["n_services"] >= 15


def test_dry_run_no_failures():
    if not ARTIFACT.exists():
        pytest.skip("Run scripts/deploy_dry_run.py first")
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert raw["n_fail"] == 0, (
        f"Dry-run failures: {raw['n_fail']} -- see DEPLOY_DRY_RUN.md"
    )


def test_dry_run_no_duplicate_service_names():
    if not ARTIFACT.exists():
        pytest.skip("Run scripts/deploy_dry_run.py first")
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert raw["duplicate_service_names"] == [], (
        f"Duplicate names: {raw['duplicate_service_names']}"
    )


def test_dry_run_every_service_has_healthz_200():
    if not ARTIFACT.exists():
        pytest.skip("Run scripts/deploy_dry_run.py first")
    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for row in raw["rows"]:
        assert row["healthz_status"] == 200, (
            f"{row['service_name']}: /healthz returned "
            f"{row['healthz_status']}"
        )
