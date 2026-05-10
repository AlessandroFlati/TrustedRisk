"""Phase 17.AI - Federation marketplace registry + bundle coverage audit.

Aggregates every ``apps/specialist_*/agent_card.json`` into a single
federation registry document. Verifies that the union of specialist
``_bundles`` covers every entry in ``mcp_server.tools.BUNDLES`` and
flags any gap. Also emits a marketplace-style discovery manifest
for A2A v1 chat clients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# Bundle -> proposed specialist owner (used to fill coverage gaps).
_BUNDLE_TO_OWNER: dict[str, str] = {
    "core_discharge":           "trustedrisk-discharge",
    "ed_acute":                 "trustedrisk-acute",
    "pediatric":                "trustedrisk-pediatric",
    "mental_health":            "trustedrisk-mental-health",
    "antimicrobial":            "trustedrisk-discharge",
    "oncology":                 "trustedrisk-discharge",
    "stroke_acs":               "trustedrisk-acute",
    "obstetric_geriatric":      "trustedrisk-acute",
    "trauma_critical":          "trustedrisk-acute",
    "endocrine_acute":          "trustedrisk-acute",
    "imaging":                  "trustedrisk-acute",
    "nephrology":               "trustedrisk-acute",
    "economics":                "trustedrisk-population",
    "context_resolution":       "trustedrisk-evidence",
    "diagnosis":                "trustedrisk-evidence",
    "patient_facing":           "trustedrisk-patient",
    "data_normalization":       "trustedrisk-discharge",
    "clinical_workflow":        "trustedrisk-discharge",
    "external_knowledge":       "trustedrisk-evidence",
    "chart_intelligence":       "trustedrisk-evidence",
    "prior_authorization":      "trustedrisk-pa",
    "clinical_documentation":   "trustedrisk-scribe",
    "patient_qa":               "trustedrisk-patient",
    "auto_coding":              "trustedrisk-coder",
    "pharmacogenomics":         "trustedrisk-pgx",
    "preadmit_triage":          "trustedrisk-preadmit",
    "research_design":          "trustedrisk-evidence",
    "quality_stars":            "trustedrisk-quality",
    "population_health":        "trustedrisk-pophealth",
    "insurance_appeals":        "trustedrisk-appeals",
    "multimodal":               "trustedrisk-multimodal",
    "critical_care":            "trustedrisk-acute",
    "specialty_clinics":        "trustedrisk-evidence",
    "cardiology_depth":         "trustedrisk-acute",
    "heme_onc_depth":           "trustedrisk-discharge",
    "endocrinology_advanced":   "trustedrisk-acute",
    "sleep_pain":               "trustedrisk-evidence",
    "transplant":               "trustedrisk-acute",
    "fhir_writeback":           "trustedrisk-discharge",
    "rheumatology":             "trustedrisk-acute",
    "peri_op_risk":             "trustedrisk-acute",
    "infectious_disease":       "trustedrisk-acute",
    "gi_hepatology_depth":      "trustedrisk-acute",
    "neurology_depth":          "trustedrisk-acute",
    "ob_peds_advanced":         "trustedrisk-acute",
    "model_research":           "trustedrisk-evidence",
    "legacy_ehr_parsers":       "trustedrisk-evidence",
}


# Canonical port assignments per the apps/specialist_*/server.py declarations.
_SPECIALIST_PORTS: dict[str, int] = {
    "trustedrisk-discharge":     8770,
    "trustedrisk-acute":         8771,
    "trustedrisk-evidence":      8772,
    "trustedrisk-population":    8773,
    "trustedrisk-pediatric":     8774,
    "trustedrisk-pa":            8775,
    "trustedrisk-scribe":        8776,
    "trustedrisk-patient":       8777,
    "trustedrisk-coder":         8778,
    "trustedrisk-pgx":           8779,
    "trustedrisk-preadmit":      8781,
    "trustedrisk-quality":       8782,
    "trustedrisk-pophealth":     8783,
    "trustedrisk-appeals":       8784,
    "trustedrisk-mental-health": 8785,
    "trustedrisk-multimodal":    8786,
}


class SpecialistEntry(BaseModel):
    name: str
    description_first_line: str
    bundles: list[str]
    n_skills: int = Field(ge=0)
    port: int | None = None
    card_path: str


class FederationRegistry(BaseModel):
    n_specialists: int = Field(ge=0)
    specialists: list[SpecialistEntry]
    bundles_covered: list[str]
    bundles_missing: list[str]
    coverage_percent: float = Field(ge=0.0, le=100.0)
    rationale: str


def build_federation_registry(
    apps_dir: Path | None = None,
) -> FederationRegistry:
    if apps_dir is None:
        apps_dir = (
            Path(__file__).resolve().parent.parent.parent / "apps"
        )
    if not apps_dir.exists():
        raise ValueError(
            f"apps directory not found: {apps_dir}")
    entries: list[SpecialistEntry] = []
    bundles_seen: set[str] = set()
    for card_path in sorted(apps_dir.glob("specialist_*/agent_card.json")):
        try:
            data = json.loads(card_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Use the canonical slug (`_legacy_name`, e.g. trustedrisk-discharge)
        # for routing + registry identity. The human-readable `name` field
        # ("Discharge Planner") is reserved for the chat-client dropdown.
        name = data.get("_legacy_name") or data.get("name", "")
        bundles_dict = data.get("_bundles", {}) or {}
        bundle_names = [
            k for k in bundles_dict.keys() if not k.startswith("_")
        ]
        bundles_seen.update(bundle_names)
        skills = data.get("skills", []) or []
        desc_full = data.get("description", "") or ""
        desc_first = desc_full.split(".")[0] if desc_full else ""
        port = _SPECIALIST_PORTS.get(name)
        entries.append(SpecialistEntry(
            name=name,
            description_first_line=desc_first,
            bundles=sorted(bundle_names),
            n_skills=len(skills),
            port=port,
            card_path=str(card_path.relative_to(apps_dir.parent)),
        ))
    from mcp_server.tools import BUNDLES   # type: ignore
    runtime_bundles = set(BUNDLES.keys())
    missing = sorted(runtime_bundles - bundles_seen)
    covered = sorted(runtime_bundles & bundles_seen)
    coverage_pct = 100.0 * len(covered) / max(1, len(runtime_bundles))
    return FederationRegistry(
        n_specialists=len(entries),
        specialists=entries,
        bundles_covered=covered,
        bundles_missing=missing,
        coverage_percent=round(coverage_pct, 2),
        rationale=(
            f"{len(entries)} specialist agent cards aggregated; "
            f"{len(covered)}/{len(runtime_bundles)} runtime bundles "
            f"covered ({coverage_pct:.1f}%); "
            f"{len(missing)} gap(s)."
        ),
    )


def render_marketplace_manifest(
    registry: FederationRegistry,
    *, base_url: str = "https://trustedrisk.local",
) -> dict[str, Any]:
    """Emit a marketplace manifest for the federation."""
    return {
        "schemaVersion": "1.0.0",
        "name": "trustedrisk-federation",
        "description": (
            "TrustedRisk federation - 15 specialist agents sharing a "
            "common 145-tool / 47-bundle MCP backend."
        ),
        "marketplaceVersion": "v1",
        "n_specialists": registry.n_specialists,
        "n_bundles_covered": len(registry.bundles_covered),
        "n_bundles_missing": len(registry.bundles_missing),
        "specialists": [
            {
                "name": e.name,
                # Force POSIX separators on the URL even when the
                # generator runs on Windows -- agent_card.json paths in
                # the manifest are HTTP URLs, not filesystem paths.
                "agent_card_url": (
                    f"{base_url}/{e.card_path}".replace("\\", "/")
                ),
                "port": e.port,
                "n_bundles": len(e.bundles),
                "n_skills": e.n_skills,
                "summary": e.description_first_line[:200],
            }
            for e in registry.specialists
        ],
        "bundle_coverage": {
            "covered": registry.bundles_covered,
            "missing": registry.bundles_missing,
            "coverage_percent": registry.coverage_percent,
        },
        "proposed_owner_for_missing": {
            b: _BUNDLE_TO_OWNER.get(b, "unassigned")
            for b in registry.bundles_missing
        },
    }


def render_registry_md(registry: FederationRegistry) -> str:
    lines = [
        "# TrustedRisk - Federation Registry", "",
        f"**Specialists**: {registry.n_specialists} - "
        f"**Bundle coverage**: "
        f"{len(registry.bundles_covered)}/"
        f"{len(registry.bundles_covered) + len(registry.bundles_missing)} "
        f"({registry.coverage_percent:.1f}%)",
        "",
        "## Specialists",
        "",
        "| Name | Port | Bundles | Skills |",
        "| --- | ---: | ---: | ---: |",
    ]
    for e in registry.specialists:
        lines.append(
            f"| `{e.name}` | "
            f"{e.port if e.port is not None else '-'} | "
            f"{len(e.bundles)} | {e.n_skills} |"
        )
    lines.append("")
    if registry.bundles_missing:
        lines.append("## Bundle gaps")
        lines.append("")
        lines.append("| Bundle | Proposed owner |")
        lines.append("| --- | --- |")
        for b in registry.bundles_missing:
            lines.append(
                f"| `{b}` | `{_BUNDLE_TO_OWNER.get(b, 'unassigned')}` |"
            )
        lines.append("")
    else:
        lines.append("All runtime bundles are covered.")
    return "\n".join(lines)
