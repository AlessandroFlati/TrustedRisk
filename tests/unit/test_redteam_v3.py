"""Phase 12.7 B2 -- Adversarial v3 multi-target campaign tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from a2a_agent.redteam_v3 import (
    MultiTargetReport,
    ToolRobustnessRow,
    run_redteam_multi_target,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO_ROOT / "data" / "redteam" / "redteam_corpus_v2.json"


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Smoke
# ─────────────────────────────────────────────────────────────────────

def test_v3_runs_against_at_least_10_tools():
    rep = _run(run_redteam_multi_target(CORPUS))
    assert isinstance(rep, MultiTargetReport)
    assert rep.n_tools_evaluated >= 10


def test_v3_every_row_has_pass_count_consistent_with_pass_rate():
    rep = _run(run_redteam_multi_target(CORPUS))
    for row in rep.rows:
        if row.n_cases > 0:
            expected = round(row.n_passed / row.n_cases, 4)
            assert abs(row.pass_rate - expected) < 1e-3


def test_v3_detect_phi_passes_all_prompts():
    """The cite-back-aware scorer applied to detect_phi (the v2
    target) must keep its 100 % pass-rate when re-run via v3."""
    rep = _run(run_redteam_multi_target(CORPUS))
    row = next(r for r in rep.rows if r.tool_name == "detect_phi")
    assert row.posture == "pass"
    assert row.pass_rate == 1.0


def test_v3_overall_pass_rate_above_random():
    """The deterministic floor must beat random across the surface."""
    rep = _run(run_redteam_multi_target(CORPUS))
    assert rep.overall_avg_pass_rate >= 0.50


# ─────────────────────────────────────────────────────────────────────
# Posture taxonomy
# ─────────────────────────────────────────────────────────────────────

def test_v3_postures_are_in_known_set():
    rep = _run(run_redteam_multi_target(CORPUS))
    for row in rep.rows:
        assert row.posture in {"pass", "warn", "fail"}


def test_v3_posture_pass_iff_zero_failures():
    rep = _run(run_redteam_multi_target(CORPUS))
    for row in rep.rows:
        if row.posture == "pass":
            assert row.n_failed == 0


# ─────────────────────────────────────────────────────────────────────
# Doc artefact
# ─────────────────────────────────────────────────────────────────────

def test_red_team_map_artefact_present_after_driver_run():
    """Run the driver and assert the RED_TEAM_MAP.md exists."""
    md_path = REPO_ROOT / "docs" / "adversarial" / "RED_TEAM_MAP.md"
    json_path = REPO_ROOT / "docs" / "adversarial" / "red_team_map.json"
    if not (md_path.exists() and json_path.exists()):
        pytest.skip("Run scripts/run_redteam_v3.py first")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["n_tools_evaluated"] >= 10


def test_artefact_passes_invariant_overall_avg_in_unit_interval():
    json_path = REPO_ROOT / "docs" / "adversarial" / "red_team_map.json"
    if not json_path.exists():
        pytest.skip("Run scripts/run_redteam_v3.py first")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert 0.0 <= raw["overall_avg_pass_rate"] <= 1.0
