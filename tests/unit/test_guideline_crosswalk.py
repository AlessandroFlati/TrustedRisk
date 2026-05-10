"""Phase 17.P - Guideline crosswalk tests."""

from __future__ import annotations

from a2a_agent.guideline_crosswalk import (
    GuidelineCrosswalkReport, build_guideline_crosswalk,
    render_guideline_crosswalk_md,
)


def test_crosswalk_covers_every_registered_tool():
    from mcp_server.tools import BUNDLES
    seen: set[str] = set()
    for tools in BUNDLES.values():
        seen.update(tools)
    rep = build_guideline_crosswalk()
    mapped = {r.tool_name for r in rep.rows}
    assert mapped == seen


def test_crosswalk_at_least_50_tools_have_specific_override():
    rep = build_guideline_crosswalk()
    assert rep.n_with_specific_guideline >= 50


def test_every_row_has_year_in_range():
    rep = build_guideline_crosswalk()
    for r in rep.rows:
        assert 1900 <= r.year <= 2030


def test_every_row_has_non_empty_source():
    rep = build_guideline_crosswalk()
    for r in rep.rows:
        assert r.guideline_source.strip()
        assert r.level_of_evidence.strip()


def test_render_emits_per_bundle_sections():
    rep = build_guideline_crosswalk()
    md = render_guideline_crosswalk_md(rep)
    bundles_in_md = sum(
        1 for line in md.splitlines() if line.startswith("## ")
    )
    bundles_unique = {r.bundle for r in rep.rows}
    assert bundles_in_md == len(bundles_unique)


def test_round_trip_through_pydantic():
    rep = build_guideline_crosswalk()
    payload = rep.model_dump(mode="json")
    rebuilt = GuidelineCrosswalkReport.model_validate(payload)
    assert rebuilt.n_rows == rep.n_rows


def test_specific_known_tools_have_canonical_year():
    rep = build_guideline_crosswalk()
    by_tool = {r.tool_name: r for r in rep.rows}
    assert by_tool["compute_heart_score"].year == 2008
    assert by_tool["compute_apache_ii_score"].year == 1985
    assert by_tool["compute_meld_score"].year == 2007
    assert by_tool["compute_aki_kdigo_stage"].year == 2012
