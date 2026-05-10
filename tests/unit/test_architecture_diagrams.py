"""Phase 13.1 J1 -- Architecture diagram contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOC = REPO_ROOT / "docs" / "ARCHITECTURE.md"


def test_architecture_doc_present():
    assert DOC.exists()


def test_architecture_doc_has_three_mermaid_blocks():
    """Federation topology + sequence diagram + calibration pipeline."""
    text = DOC.read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid", text)
    assert len(blocks) >= 3, (
        f"Expected ≥3 Mermaid blocks, got {len(blocks)}"
    )


def test_architecture_doc_lists_all_16_specialists():
    text = DOC.read_text(encoding="utf-8")
    expected = [
        "discharge :8770", "acute :8771", "evidence :8772",
        "population :8773", "pediatric :8774", "pa :8775",
        "scribe :8776", "patient :8777", "coder :8778",
        "pgx :8779", "preadmit :8781", "quality :8782",
        "pophealth :8783", "appeals :8784", "mental-health :8785",
        "multimodal :8786",
    ]
    for fragment in expected:
        assert fragment in text, (
            f"ARCHITECTURE.md does not list {fragment!r}"
        )


def test_architecture_snapshot_reflects_145_tools_47_bundles():
    text = DOC.read_text(encoding="utf-8")
    assert "145 MCP tools" in text or "145 tools" in text
    assert "47 thematic" in text or "47 bundles" in text


def test_architecture_calibration_block_lists_all_four_cohorts():
    text = DOC.read_text(encoding="utf-8")
    for cohort in ("W1", "Synthea-10k", "Synthea-100k", "MIMIC-IV"):
        assert cohort in text, (
            f"Calibration diagram missing cohort {cohort!r}"
        )


def test_architecture_doc_renders_known_mermaid_class_defs():
    """Mermaid `classDef` markers -- guard against accidentally
    deleting the styling on a copy-paste edit."""
    text = DOC.read_text(encoding="utf-8")
    for cls in ("classDef user", "classDef spec", "classDef governance"):
        assert cls in text


def test_white_paper_references_architecture():
    wp = REPO_ROOT / "docs" / "WHITE_PAPER.md"
    if not wp.exists():
        pytest.skip("WHITE_PAPER.md not found")
    text = wp.read_text(encoding="utf-8")
    # White paper §2 already covers federation topology in prose;
    # this test just makes sure the prose mentions the key terms.
    assert "SHARP-on-MCP" in text
    assert "A2A v1" in text or "A2A" in text
