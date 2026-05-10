"""Phase 17.AI - Federation registry tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from a2a_agent.federation_registry import (
    FederationRegistry, build_federation_registry,
    render_marketplace_manifest, render_registry_md,
)


ROOT = Path(__file__).resolve().parent.parent.parent


def test_registry_aggregates_at_least_fifteen_specialists():
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    assert reg.n_specialists >= 15


def test_registry_assigns_canonical_ports_to_known_specialists():
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    by_name = {e.name: e for e in reg.specialists}
    assert by_name["trustedrisk-discharge"].port == 8770
    assert by_name["trustedrisk-acute"].port == 8771
    assert by_name["trustedrisk-evidence"].port == 8772
    assert by_name["trustedrisk-population"].port == 8773
    assert by_name["trustedrisk-pediatric"].port == 8774
    assert by_name["trustedrisk-mental-health"].port == 8785


def test_coverage_includes_phase_13_14_bundles():
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    covered = set(reg.bundles_covered)
    for required in ("rheumatology", "peri_op_risk", "infectious_disease",
                      "gi_hepatology_depth", "neurology_depth",
                      "ob_peds_advanced", "model_research",
                      "legacy_ehr_parsers", "cardiology_depth",
                      "heme_onc_depth", "endocrinology_advanced",
                      "sleep_pain", "transplant", "specialty_clinics",
                      "critical_care"):
        assert required in covered, (
            f"missing Phase-13/14 bundle in federation: {required}"
        )


def test_full_runtime_coverage_after_extension():
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    assert reg.coverage_percent == 100.0
    assert reg.bundles_missing == []


def test_marketplace_manifest_includes_all_specialists():
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    manifest = render_marketplace_manifest(reg)
    assert manifest["n_specialists"] == reg.n_specialists
    assert len(manifest["specialists"]) == reg.n_specialists
    for entry in manifest["specialists"]:
        assert entry["name"].startswith("trustedrisk-")
        assert "agent_card_url" in entry


def test_registry_render_md_emits_summary_table():
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    md = render_registry_md(reg)
    assert "# TrustedRisk - Federation Registry" in md
    assert "| Name | Port | Bundles | Skills |" in md


def test_round_trip_through_pydantic():
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    payload = reg.model_dump(mode="json")
    rebuilt = FederationRegistry.model_validate(payload)
    assert rebuilt.n_specialists == reg.n_specialists


def test_registry_rejects_missing_apps_dir():
    with pytest.raises(ValueError):
        build_federation_registry(
            apps_dir=ROOT / "definitely_does_not_exist")
