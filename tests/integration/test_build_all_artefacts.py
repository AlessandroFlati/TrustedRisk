"""Phase 17.CON5 - Build-pipeline smoke test.

Runs ``scripts/build_all_artefacts.py`` end-to-end + verifies
every expected artefact exists with non-trivial size. The pipeline
is fast enough (~10 s) to run in CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent


_EXPECTED_ARTEFACTS: list[tuple[Path, int]] = [
    (ROOT / "docs" / "fairness" / "subgroup_audit.json", 5_000),
    (ROOT / "docs" / "fairness" / "SUBGROUP_AUDIT.md", 1_000),
    (ROOT / "docs" / "prospective" / "prospective_eval.json", 2_000),
    (ROOT / "docs" / "prospective" / "PROSPECTIVE_EVAL.md", 1_000),
    (ROOT / "docs" / "validation" / "HRRP_BENCHMARK.md", 500),
    (ROOT / "docs" / "research" / "MODEL_CARD.md", 1_000),
    (ROOT / "docs" / "research" / "DATASHEET.md", 1_000),
    (ROOT / "docs" / "regulatory" / "REGULATORY_PACK.md", 5_000),
    (ROOT / "docs" / "regulatory" / "regulatory_pack.json", 5_000),
    (ROOT / "docs" / "regulatory" / "FRAMEWORK_CROSSWALKS.md", 1_000),
    (ROOT / "docs" / "guidelines" / "GUIDELINE_CROSSWALK.md", 5_000),
    (ROOT / "docs" / "federation" / "FEDERATION_REGISTRY.md", 500),
    (ROOT / "docs" / "federation" / "marketplace_manifest.json", 1_000),
    (ROOT / "docs" / "e2e" / "v7" / "index.html", 20_000),
    (ROOT / "docs" / "showcase" / "STORYMODE.html", 15_000),
    (ROOT / "docs" / "economics" / "COST_SIMULATION.md", 500),
    (ROOT / "docs" / "federated" / "FEDERATED_LEARNING.md", 1_000),
    (ROOT / "docs" / "adversarial" / "red_team_v4.json", 1_000),
    (ROOT / "docs" / "ui" / "counterfactual.html", 5_000),
    (ROOT / "docs" / "api" / "openapi.json", 3_000),
    (ROOT / "docs" / "MODULE_CATALOG.md", 5_000),
]


def _run_build_pipeline() -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(
            ROOT / "scripts" / "build_all_artefacts.py")],
        cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=900,
    )


@pytest.fixture(scope="module")
def _pipeline_result() -> subprocess.CompletedProcess:
    return _run_build_pipeline()


def test_pipeline_exit_code_is_zero(_pipeline_result):
    assert _pipeline_result.returncode == 0, (
        f"build_all_artefacts.py exited with "
        f"{_pipeline_result.returncode};\n"
        f"stdout: {_pipeline_result.stdout[-500:]}\n"
        f"stderr: {_pipeline_result.stderr[-500:]}"
    )


def test_pipeline_summary_reports_zero_failures(_pipeline_result):
    out = _pipeline_result.stdout
    assert "0 FAIL" in out


@pytest.mark.parametrize(
    "path,min_size",
    [(str(p.relative_to(ROOT)), s)
     for p, s in _EXPECTED_ARTEFACTS],
)
def test_artefact_exists_and_above_min_size(
    _pipeline_result, path: str, min_size: int,
):
    assert _pipeline_result.returncode == 0
    full = ROOT / path
    assert full.exists(), f"missing artefact: {path}"
    assert full.stat().st_size >= min_size, (
        f"{path} smaller than {min_size} bytes "
        f"(actual: {full.stat().st_size})"
    )


def test_module_catalog_records_at_least_seventy_modules():
    catalog = ROOT / "docs" / "MODULE_CATALOG.md"
    text = catalog.read_text(encoding="utf-8")
    n_module_headings = sum(
        1 for ln in text.splitlines()
        if ln.startswith("## `a2a_agent.")
    )
    assert n_module_headings >= 70


def test_regulatory_pack_marks_full_artefact_coverage():
    pack_md = (
        ROOT / "docs" / "regulatory" / "REGULATORY_PACK.md"
    ).read_text(encoding="utf-8")
    assert "Artefact coverage**: 100.0%" in pack_md


def test_federation_registry_at_one_hundred_percent_coverage():
    reg = (
        ROOT / "docs" / "federation" / "FEDERATION_REGISTRY.md"
    ).read_text(encoding="utf-8")
    assert "100.0%" in reg
