"""Phase 16.G2 - Framework crosswalks tests."""

from __future__ import annotations

import pytest

from a2a_agent.framework_crosswalks import (
    FrameworkCrosswalkReport, build_framework_crosswalks,
    render_crosswalks_md,
)


def test_report_carries_nist_and_oecd_rows():
    r = build_framework_crosswalks()
    assert len(r.nist_ai_rmf_rows) >= 14
    assert len(r.oecd_ai_principles_rows) == 5


def test_n_rows_equals_sum_of_subreports():
    r = build_framework_crosswalks()
    assert r.n_rows == (
        len(r.nist_ai_rmf_rows) + len(r.oecd_ai_principles_rows)
    )


def test_nist_rows_cover_four_functions():
    r = build_framework_crosswalks()
    sections = [row.framework_section for row in r.nist_ai_rmf_rows]
    for fn in ("GOVERN", "MAP", "MEASURE", "MANAGE"):
        assert any(s.startswith(fn) for s in sections), (
            f"NIST function {fn} missing from crosswalk"
        )


def test_oecd_rows_cover_five_principles():
    r = build_framework_crosswalks()
    for n in range(1, 6):
        assert any(
            row.framework_section.startswith(f"Principle 1.{n}")
            for row in r.oecd_ai_principles_rows
        )


def test_render_emits_markdown_tables():
    r = build_framework_crosswalks()
    md = render_crosswalks_md(r)
    assert md.startswith("# TrustedRisk - NIST AI RMF")
    assert "## NIST AI RMF 1.0" in md
    assert "## OECD AI Principles" in md
    assert "| Section | Requirement |" in md


def test_round_trip_serialises():
    r = build_framework_crosswalks()
    payload = r.model_dump(mode="json")
    rebuilt = FrameworkCrosswalkReport.model_validate(payload)
    assert rebuilt.n_rows == r.n_rows


def test_every_row_has_evidence_field():
    r = build_framework_crosswalks()
    for row in r.nist_ai_rmf_rows + r.oecd_ai_principles_rows:
        assert row.requirement.strip()
        assert row.trustedrisk_evidence.strip()
