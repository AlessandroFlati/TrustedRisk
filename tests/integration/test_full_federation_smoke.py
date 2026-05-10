"""Phase 10.10 -- Full-federation smoke (S3).

Cross-specialist integration: builds the in-memory ASGI app for every
specialist + composer + master agent, hits each /healthz and
/.well-known/agent-card.json, and asserts the federation hangs
together under one shared MCP backend.

Does NOT spawn subprocesses or open external ports. The Makefile
target `make demo-up-full` is the corresponding ops-side surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PO_FHIR_CONTEXT_URI = (
    "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"
)


# (specialist_id, expected_port) -- one source of truth that any future
# Phase-10 expansion must update in lock-step with the agent-card
# generator + the demo_up_full.sh roster.
EXPECTED_PORTS = {
    "specialist_discharge":  8770,
    "specialist_acute":      8771,
    "specialist_evidence":   8772,
    "specialist_population": 8773,
    "specialist_pediatric":     8774,
    "specialist_mental_health": 8785,
    "specialist_pa":         8775,
    "specialist_scribe":     8776,
    "specialist_patient":    8777,
    "specialist_coder":      8778,
    "specialist_pgx":        8779,
    "specialist_preadmit":   8781,
    "specialist_quality":    8782,
    "specialist_pophealth":  8783,
    "specialist_appeals":    8784,
    "specialist_multimodal": 8786,
}


@pytest.fixture(scope="module")
def federation_apps():
    """Build all 16 specialist apps once."""
    from apps.specialist_acute.server import app as acute_app
    from apps.specialist_appeals.server import app as appeals_app
    from apps.specialist_coder.server import app as coder_app
    from apps.specialist_discharge.server import app as discharge_app
    from apps.specialist_evidence.server import app as evidence_app
    from apps.specialist_multimodal.server import app as multimodal_app
    from apps.specialist_pa.server import app as pa_app
    from apps.specialist_patient.server import app as patient_app
    from apps.specialist_pediatric.server import app as pediatric_app
    from apps.specialist_mental_health.server import app as mental_health_app
    from apps.specialist_pgx.server import app as pgx_app
    from apps.specialist_pophealth.server import app as pophealth_app
    from apps.specialist_population.server import app as population_app
    from apps.specialist_preadmit.server import app as preadmit_app
    from apps.specialist_quality.server import app as quality_app
    from apps.specialist_scribe.server import app as scribe_app
    return {
        "specialist_discharge":  discharge_app,
        "specialist_acute":      acute_app,
        "specialist_evidence":   evidence_app,
        "specialist_population": population_app,
        "specialist_pediatric":     pediatric_app,
        "specialist_mental_health": mental_health_app,
        "specialist_pa":         pa_app,
        "specialist_scribe":     scribe_app,
        "specialist_patient":    patient_app,
        "specialist_coder":      coder_app,
        "specialist_pgx":        pgx_app,
        "specialist_preadmit":   preadmit_app,
        "specialist_quality":    quality_app,
        "specialist_pophealth":  pophealth_app,
        "specialist_appeals":    appeals_app,
        "specialist_multimodal": multimodal_app,
    }


def test_all_16_specialists_build(federation_apps):
    assert len(federation_apps) == len(EXPECTED_PORTS) == 16


def test_every_specialist_serves_healthz(federation_apps):
    for spec_id, app in federation_apps.items():
        client = TestClient(app)
        r = client.get("/healthz")
        assert r.status_code == 200, f"{spec_id} /healthz returned {r.status_code}"


def test_every_specialist_reports_same_shared_backend(federation_apps):
    """The shared MCP backend must report identical tool / bundle
    counts across every specialist deployment. A drift here means a
    specialist accidentally registered a different tool surface."""
    counts = set()
    for spec_id, app in federation_apps.items():
        client = TestClient(app)
        body = client.get("/healthz").json()
        counts.add((body["tools_registered"], body["bundles_registered"]))
    assert len(counts) == 1, (
        f"Tool/bundle counts diverge across specialists: {counts}"
    )


def test_phase_13_expansion_is_visible_in_shared_counts(federation_apps):
    """After Phase 13 H expansion (critical_care + specialty_clinics
    + cardiology + heme/onc + endocrinology + sleep_pain + transplant
    + fairness + Phase-14 K bundles + model_research + legacy_ehr_parsers)
    we expect 145 tools / 47 bundles."""
    client = TestClient(next(iter(federation_apps.values())))
    body = client.get("/healthz").json()
    assert body["tools_registered"] == 145
    assert body["bundles_registered"] == 47


def test_specialist_directories_carry_agent_card_json(federation_apps):
    """Every specialist directory must ship its generated agent_card.json
    so the local demo can serve it directly without re-running the
    generator script."""
    for spec_id in federation_apps:
        path = REPO_ROOT / "apps" / spec_id / "agent_card.json"
        assert path.exists(), f"Missing card at {path}"


def test_phase_10_specialists_advertise_only_their_own_bundle(federation_apps):
    """Tight scope check -- the 3 new Phase-10 specialists must each
    surface exactly one Phase-10 bundle and nothing else."""
    expected_bundle = {
        "specialist_quality":   "quality_stars",
        "specialist_pophealth": "population_health",
        "specialist_appeals":   "insurance_appeals",
    }
    for spec_id, expected in expected_bundle.items():
        client = TestClient(federation_apps[spec_id])
        card = client.get("/.well-known/agent-card.json").json()
        bundles = {b for b in card["_bundles"] if not b.startswith("_")}
        assert bundles == {expected}, (
            f"{spec_id}: expected exactly {{{expected!r}}}, got {bundles}"
        )


def test_makefile_lists_demo_up_full_target():
    mk = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "demo-up-full" in mk
    assert "demo-status" in mk


def test_demo_up_full_script_present_and_lists_all_specialists():
    """The shell script bundled with the demo must enumerate every
    Phase-10 / Phase-11 specialist; tests/integration smoke would
    otherwise miss a missing process at deploy time."""
    script = (REPO_ROOT / "scripts" / "demo_up_full.sh").read_text(
        encoding="utf-8"
    )
    for new_specialist in (
        "quality]=8782", "pophealth]=8783", "appeals]=8784",
        "multimodal]=8786",
    ):
        assert new_specialist in script, (
            f"demo_up_full.sh missing {new_specialist}"
        )


def test_master_agent_card_lists_phase_10_bundles():
    raw = json.loads(
        (REPO_ROOT / "src" / "a2a_agent" / "agent-card.json").read_text(
            encoding="utf-8"
        )
    )
    bundles = set(raw["_bundles"]) - {"_doc"}
    for new in ("quality_stars", "population_health", "insurance_appeals"):
        assert new in bundles, f"Master card missing bundle {new!r}"


def test_planner_brain_routes_phase_10_intents_through_federation():
    """Phase 10.8's planner must route the new Phase 10.1/2/3 intents
    to the new Phase 10.x specialists -- ties the brain to the
    federation surface explicitly."""
    import asyncio

    from a2a_agent.planner import plan_tool_use

    async def _do(query):
        return await plan_tool_use(query)

    intents = [
        ("HEDIS Stars rating forecast for this MA contract.",
         "trustedrisk-quality"),
        ("Detect syndromic clusters in the ED chief complaints.",
         "trustedrisk-pophealth"),
        ("Parse this Aetna denial letter.",
         "trustedrisk-appeals"),
    ]
    for query, expected_specialist in intents:
        plan = asyncio.run(_do(query))
        specialists = {s.specialist for s in plan.steps}
        assert expected_specialist in specialists, (
            f"Planner did not route {query!r} to {expected_specialist}; "
            f"got {specialists}"
        )
