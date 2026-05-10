"""Phase 14.17 P1 -- Regulatory pack generator tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from a2a_agent.regulatory_pack import (
    RegulatoryPack, build_regulatory_pack,
    render_regulatory_pack_md,
)


# ─────────────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────────────

def test_pack_has_fourteen_sections():
    pack = build_regulatory_pack()
    assert pack.n_sections == 14


def test_pack_section_titles_unique():
    pack = build_regulatory_pack()
    titles = [s.title for s in pack.sections]
    assert len(set(titles)) == len(titles)


def test_pack_covers_three_frameworks():
    pack = build_regulatory_pack()
    fw_text = " ".join(s.framework for s in pack.sections)
    assert "EU AI Act" in fw_text
    assert "FDA 510(k)" in fw_text
    assert "ISO 13485" in fw_text


def test_overall_artefact_coverage_in_unit_interval():
    pack = build_regulatory_pack()
    assert 0.0 <= pack.overall_artefact_coverage <= 1.0


def test_pack_carries_generated_timestamp():
    pack = build_regulatory_pack()
    assert "UTC" in pack.generated_at


# ─────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────

def test_render_includes_title_and_metadata():
    pack = build_regulatory_pack()
    md = render_regulatory_pack_md(pack)
    assert md.startswith("# TrustedRisk -- Regulatory Pack")
    assert "Generated" in md
    assert "Artefact coverage" in md


def test_render_includes_each_section_heading():
    pack = build_regulatory_pack()
    md = render_regulatory_pack_md(pack)
    for section in pack.sections:
        assert f"## " in md
        assert section.title in md


def test_render_includes_framework_metadata_line():
    pack = build_regulatory_pack()
    md = render_regulatory_pack_md(pack)
    for section in pack.sections:
        assert f"*Framework*: **{section.framework}**" in md


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────

def test_pack_render_deterministic_modulo_timestamp():
    """Two consecutive runs produce identical content except for the
    generated_at line."""
    pack_a = build_regulatory_pack()
    pack_b = build_regulatory_pack()
    md_a = render_regulatory_pack_md(pack_a).splitlines()
    md_b = render_regulatory_pack_md(pack_b).splitlines()
    a_minus_ts = [ln for ln in md_a if "Generated" not in ln]
    b_minus_ts = [ln for ln in md_b if "Generated" not in ln]
    assert a_minus_ts == b_minus_ts


# ─────────────────────────────────────────────────────────────────────
# Schema invariants
# ─────────────────────────────────────────────────────────────────────

def test_section_body_non_empty():
    pack = build_regulatory_pack()
    for s in pack.sections:
        assert s.body_md.strip() != ""


def test_pack_serialises_round_trip():
    pack = build_regulatory_pack()
    payload = pack.model_dump(mode="json")
    rebuilt = RegulatoryPack.model_validate(payload)
    assert rebuilt.n_sections == pack.n_sections
    assert (
        rebuilt.overall_artefact_coverage
        == pack.overall_artefact_coverage
    )


# ─────────────────────────────────────────────────────────────────────
# Specific section presence
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "expected_title",
    [
        "Intended purpose + risk classification",
        "Calibration + external validation",
        "Fairness + bias governance",
        "Audit trail + record-keeping",
        "Adversarial robustness + design verification",
        "Performance + scalability",
        "Quality Management System",
        "Post-market monitoring + incident reporting",
        "FDA 510(k) predicate device comparison",
        "GDPR Art. 35 Data Protection Impact Assessment",
        "HIPAA §164 Privacy + Security Rule crosswalk",
        "Cybersecurity + SBOM",
        "Human oversight + emergency override",
        "EU AI Act conformity assessment readiness",
    ],
)
def test_specific_section_present(expected_title: str):
    pack = build_regulatory_pack()
    titles = [s.title for s in pack.sections]
    assert expected_title in titles
