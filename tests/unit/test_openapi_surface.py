"""Phase 12.9 A2 -- OpenAPI/Swagger surface tests.

Validates that every specialist serves a /openapi.json and /docs page,
and that the spec lists each tool the specialist's `_bundles`
catalog advertises."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PHASE_10_11_SPECIALISTS = [
    "specialist_quality",
    "specialist_pophealth",
    "specialist_appeals",
    "specialist_multimodal",
]


def _all_specialists() -> dict:
    """Return {specialist_id: ASGI app} for every specialist."""
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


@pytest.fixture(scope="module")
def specialist_apps():
    return _all_specialists()


# ─────────────────────────────────────────────────────────────────────
# /openapi.json
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec_id", [
    "specialist_quality", "specialist_pophealth",
    "specialist_appeals", "specialist_multimodal",
    "specialist_discharge", "specialist_pa",
])
def test_openapi_endpoint_returns_200(specialist_apps, spec_id):
    client = TestClient(specialist_apps[spec_id])
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["openapi"].startswith("3.")
    assert "paths" in spec


@pytest.mark.parametrize("spec_id", [
    "specialist_quality", "specialist_pophealth",
    "specialist_appeals", "specialist_multimodal",
])
def test_openapi_lists_each_advertised_tool(specialist_apps, spec_id):
    """Every tool in the specialist's `_bundles` map must appear as
    a `POST /tools/{tool_name}` operation in the OpenAPI spec."""
    client = TestClient(specialist_apps[spec_id])
    card = client.get("/.well-known/agent-card.json").json()
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for bundle_id, tools in card.get("_bundles", {}).items():
        if bundle_id.startswith("_") or not isinstance(tools, list):
            continue
        for tool in tools:
            path = f"/tools/{tool}"
            assert path in paths, (
                f"{spec_id}: spec missing {path} for bundle {bundle_id}"
            )


@pytest.mark.parametrize("spec_id", PHASE_10_11_SPECIALISTS)
def test_openapi_paths_carry_bundle_tags(specialist_apps, spec_id):
    client = TestClient(specialist_apps[spec_id])
    spec = client.get("/openapi.json").json()
    for path, ops in spec["paths"].items():
        if not path.startswith("/tools/"):
            continue
        post = ops.get("post") or {}
        assert isinstance(post.get("tags"), list) and post["tags"], (
            f"{spec_id}: path {path} has no tags"
        )


# ─────────────────────────────────────────────────────────────────────
# /docs
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec_id", [
    "specialist_quality", "specialist_pophealth",
    "specialist_appeals", "specialist_multimodal",
])
def test_docs_endpoint_serves_swagger_ui(specialist_apps, spec_id):
    client = TestClient(specialist_apps[spec_id])
    r = client.get("/docs")
    assert r.status_code == 200
    body = r.text
    assert "swagger-ui" in body.lower()
    assert "/openapi.json" in body


# ─────────────────────────────────────────────────────────────────────
# Public-paths allow-list
# ─────────────────────────────────────────────────────────────────────

def test_openapi_endpoint_does_not_require_sharp_context(specialist_apps):
    """The OpenAPI surface is documentation -- it must NOT trigger
    SHARP-on-MCP enforcement."""
    client = TestClient(specialist_apps["specialist_quality"])
    r = client.get("/openapi.json")
    assert r.status_code == 200
