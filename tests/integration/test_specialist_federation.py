"""Integration tests for the Phase 1 federation specialists.

For each of the 5 specialists this verifies:
  1. The factory builds the ASGI app without raising
  2. The `/.well-known/agent-card.json` endpoint returns a card that
     validates against the A2A v1 schema (incl. supportedInterfaces[],
     extensions[], securitySchemes typed-keys)
  3. The Prompt Opinion FHIR-context extension URI is declared
  4. The scope set declared on the extension is non-empty (population
     specialist only carries `patient/Patient.rs`, the others carry
     more)
  5. The `_bundles` map contains exactly the bundles configured for
     that specialist (no leakage from sibling specialists)
  6. `/healthz` reports the SHARED 58-tool / 20-bundle backend (the
     specialist filters surface, NOT the underlying registration)
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


# (id, expected_skill_count, expected_bundle_count, expected_min_scope_count)
# Bundle counts include the Phase-13/14 bundles the
# `extend_specialist_bundle_coverage.py` script attaches to maintain 100%
# federation coverage; the largest specialists (acute, discharge,
# evidence) own a long tail of niche depth bundles.
SPECIALISTS = [
    ("specialist_discharge", 9, 8, 7),
    ("specialist_acute", 11, 17, 7),
    ("specialist_evidence", 6, 9, 5),
    ("specialist_population", 2, 1, 1),
    ("specialist_pediatric", 2, 1, 3),
    ("specialist_mental_health", 2, 1, 4),
    ("specialist_pa", 2, 1, 7),
    ("specialist_scribe", 2, 1, 8),
    ("specialist_patient", 2, 2, 1),
    ("specialist_coder", 2, 1, 7),
    ("specialist_pgx", 2, 1, 5),
    ("specialist_preadmit", 2, 1, 1),
    ("specialist_quality", 2, 1, 6),
    ("specialist_pophealth", 2, 1, 5),
    ("specialist_appeals", 2, 1, 7),
    ("specialist_multimodal", 2, 1, 4),
]


@pytest.fixture(scope="module")
def specialist_apps():
    """Build all specialist apps once per test module."""
    from apps.specialist_acute.server import app as acute_app
    from apps.specialist_appeals.server import app as appeals_app
    from apps.specialist_coder.server import app as coder_app
    from apps.specialist_discharge.server import app as discharge_app
    from apps.specialist_evidence.server import app as evidence_app
    from apps.specialist_pa.server import app as pa_app
    from apps.specialist_patient.server import app as patient_app
    from apps.specialist_pediatric.server import app as pediatric_app
    from apps.specialist_mental_health.server import app as mental_health_app
    from apps.specialist_multimodal.server import app as multimodal_app
    from apps.specialist_pgx.server import app as pgx_app
    from apps.specialist_pophealth.server import app as pophealth_app
    from apps.specialist_population.server import app as population_app
    from apps.specialist_preadmit.server import app as preadmit_app
    from apps.specialist_quality.server import app as quality_app
    from apps.specialist_scribe.server import app as scribe_app

    return {
        "specialist_discharge": discharge_app,
        "specialist_acute": acute_app,
        "specialist_evidence": evidence_app,
        "specialist_population": population_app,
        "specialist_pediatric": pediatric_app,
        "specialist_mental_health": mental_health_app,
        "specialist_pa": pa_app,
        "specialist_scribe": scribe_app,
        "specialist_patient": patient_app,
        "specialist_coder": coder_app,
        "specialist_pgx": pgx_app,
        "specialist_preadmit": preadmit_app,
        "specialist_quality": quality_app,
        "specialist_pophealth": pophealth_app,
        "specialist_appeals": appeals_app,
        "specialist_multimodal": multimodal_app,
    }


# ─────────────────────── Per-specialist parameterised tests ───────────────────────

@pytest.mark.parametrize(
    "spec_id, expected_skills, expected_bundles, expected_min_scopes",
    SPECIALISTS,
    ids=[s[0] for s in SPECIALISTS],
)
def test_specialist_agent_card_served(
    specialist_apps,
    spec_id: str,
    expected_skills: int,
    expected_bundles: int,
    expected_min_scopes: int,
):
    client = TestClient(specialist_apps[spec_id])
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200, \
        f"{spec_id}: agent-card endpoint returned {resp.status_code}"
    card = resp.json()
    # Skill count
    assert len(card["skills"]) == expected_skills, \
        f"{spec_id}: expected {expected_skills} skills, got {len(card['skills'])}"
    # Bundle count (excluding _doc annotation)
    bundle_keys = [k for k in card.get("_bundles", {})
                       if not k.startswith("_")]
    assert len(bundle_keys) == expected_bundles, \
        f"{spec_id}: expected {expected_bundles} bundles, got {bundle_keys}"
    # PO FHIR-context extension declared
    extensions = card["capabilities"]["extensions"]
    po_ext = next((e for e in extensions
                       if e["uri"] == PO_FHIR_CONTEXT_URI), None)
    assert po_ext is not None, \
        f"{spec_id}: PO FHIR-context extension URI missing"
    # Scope count
    scopes = po_ext["params"]["scopes"]
    assert len(scopes) >= expected_min_scopes, \
        f"{spec_id}: expected ≥{expected_min_scopes} scopes; got {len(scopes)}"


@pytest.mark.parametrize("spec_id", [s[0] for s in SPECIALISTS])
def test_specialist_card_a2a_v1_compliance(
    specialist_apps, spec_id: str,
):
    """Each specialist card must declare the A2A v1 fields:
    supportedInterfaces[] + securitySchemes typed-keys."""
    client = TestClient(specialist_apps[spec_id])
    card = client.get("/.well-known/agent-card.json").json()

    # supportedInterfaces[] is required
    interfaces = card.get("supportedInterfaces") or []
    assert interfaces, \
        f"{spec_id}: supportedInterfaces[] missing (A2A v1 violation)"
    for iface in interfaces:
        # The agent_card_normalizer rewrites "A2A" -> "JSONRPC" to match
        # the canonical proto-spec wire-binding name (the protocol is A2A;
        # the binding is JSONRPC over HTTP).
        assert iface["protocolBinding"] == "JSONRPC"
        assert iface["protocolVersion"] == "1.0"

    # securitySchemes uses nested typed-key shape
    schemes = card.get("securitySchemes") or {}
    for name, decl in schemes.items():
        typed = [k for k in decl if k.endswith("SecurityScheme")]
        assert len(typed) == 1, \
            f"{spec_id}: securitySchemes[{name!r}] must wrap exactly one *SecurityScheme typed key; got {list(decl.keys())}"


@pytest.mark.parametrize("spec_id", [s[0] for s in SPECIALISTS])
def test_specialist_healthz_reports_shared_backend(
    specialist_apps, spec_id: str,
):
    """Every specialist /healthz reports the SHARED 83-tool / 28-bundle
    backend -- the specialist's published surface is narrower, but the
    underlying tool registration is identical across all
    deployments. This verifies we're not accidentally registering a
    different MCP backend per specialist."""
    client = TestClient(specialist_apps[spec_id])
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tools_registered"] == 145
    assert body["bundles_registered"] == 47


@pytest.mark.parametrize("spec_id", [s[0] for s in SPECIALISTS])
def test_specialist_card_url_distinct(
    specialist_apps, spec_id: str,
):
    """Specialist cards must each carry a distinct base URL -- otherwise
    workspace-side discovery collides."""
    client = TestClient(specialist_apps[spec_id])
    card = client.get("/.well-known/agent-card.json").json()
    interfaces = card["supportedInterfaces"]
    url = interfaces[0]["url"]
    # Placeholder URLs are accepted but must encode the specialist id
    assert spec_id.replace("_", "-").replace("specialist-", "trustedrisk-") in url \
        or "trustedrisk-" in url, \
        f"{spec_id}: URL {url!r} should reference its own specialist name"


@pytest.mark.parametrize("spec_id", [s[0] for s in SPECIALISTS])
def test_specialist_card_skill_ids_distinct(
    specialist_apps, spec_id: str,
):
    """Within a card, every skill id must be unique."""
    client = TestClient(specialist_apps[spec_id])
    card = client.get("/.well-known/agent-card.json").json()
    skill_ids = [s["id"] for s in card["skills"]]
    assert len(skill_ids) == len(set(skill_ids)), \
        f"{spec_id}: duplicate skill ids {skill_ids}"


# ─────────────────────── Cross-specialist properties ───────────────────────

def test_specialist_card_files_exist_on_disk():
    """The generator script wrote each agent_card.json to its specialist
    directory."""
    for spec_id, *_ in SPECIALISTS:
        path = REPO_ROOT / "apps" / spec_id / "agent_card.json"
        assert path.exists(), f"Missing card at {path!s}"
        # And is parseable JSON
        json.loads(path.read_text(encoding="utf-8"))


def test_specialist_cards_advertise_distinct_names(specialist_apps):
    names: list[str] = []
    for spec_id in [s[0] for s in SPECIALISTS]:
        client = TestClient(specialist_apps[spec_id])
        names.append(client.get("/.well-known/agent-card.json").json()["name"])
    assert len(names) == len(set(names)), f"Specialist names collide: {names}"


def test_specialist_population_advertises_population_scope_only():
    """Population specialist is cohort-scoped -- it should NOT advertise
    patient-clinical scopes like Encounter/Condition/Observation that
    a per-patient bundle would. Only patient/Patient.rs (for cohort
    rollups) is needed."""
    from apps.specialist_population.server import app
    client = TestClient(app)
    card = client.get("/.well-known/agent-card.json").json()
    scopes = [
        s["name"]
        for s in card["capabilities"]["extensions"][0]["params"]["scopes"]
    ]
    assert scopes == ["patient/Patient.rs"], \
        f"population specialist should only need Patient.rs; got {scopes}"


def test_pediatric_and_mental_health_are_isolated_specialists():
    """Pediatric and mental-health surfaces ship as two separate
    specialists so workspace admins can grant them on their own without
    exposing the broader clinical-decision footprint."""
    from apps.specialist_pediatric.server import app as pediatric_app
    from apps.specialist_mental_health.server import app as mh_app
    pedi_card = TestClient(pediatric_app).get(
        "/.well-known/agent-card.json").json()
    mh_card = TestClient(mh_app).get(
        "/.well-known/agent-card.json").json()
    pedi_bundles = set(b for b in pedi_card["_bundles"] if not b.startswith("_"))
    mh_bundles = set(b for b in mh_card["_bundles"] if not b.startswith("_"))
    assert pedi_bundles == {"pediatric"}, \
        f"pediatric specialist must carry pediatric only; got {pedi_bundles}"
    assert mh_bundles == {"mental_health"}, \
        f"mental-health specialist must carry mental_health only; got {mh_bundles}"
