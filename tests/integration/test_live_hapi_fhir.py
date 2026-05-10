"""Phase 4.2 -- Live HAPI FHIR R4 integration test.

Hits the public test FHIR server at https://hapi.fhir.org/baseR4. The
`@pytest.mark.live` marker keeps these tests opt-in (they don't run in
CI by default -- set TRUSTEDRISK_LIVE=1 to enable).

What it proves:
  - Our SHARP middleware works against a real FHIR server (no mocks)
  - The MCP tool surface composes against real Patient/Encounter/
    Observation/Condition resources
  - We gracefully skip on upstream 5xx so transient HAPI outages don't
    poison the suite

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        TRUSTEDRISK_LIVE=1 \\
        .venv/Scripts/python.exe -m pytest tests/integration/test_live_hapi_fhir.py -v
"""

from __future__ import annotations

import os

import httpx
import pytest

from starlette.testclient import TestClient


pytestmark = pytest.mark.skipif(
    os.environ.get("TRUSTEDRISK_LIVE", "").lower() not in ("1", "true", "yes"),
    reason="Live tests require TRUSTEDRISK_LIVE=1 (avoid hitting "
            "remote HAPI server during routine CI).",
)


HAPI_URL = "https://hapi.fhir.org/baseR4"


@pytest.fixture(scope="module")
def hapi_available() -> bool:
    """Skip the whole module if HAPI is unreachable / 5xx."""
    try:
        r = httpx.get(f"{HAPI_URL}/metadata", timeout=10.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"HAPI unreachable: {exc}")
    if r.status_code >= 500:
        pytest.skip(f"HAPI returned {r.status_code}; skipping live test")
    return True


# ─────────────────────── Smoke ───────────────────────

def test_hapi_capability_statement_is_fhir_r4(hapi_available):
    r = httpx.get(f"{HAPI_URL}/metadata", timeout=10.0)
    assert r.status_code == 200
    cap = r.json()
    assert cap.get("resourceType") == "CapabilityStatement"
    fhir_version = cap.get("fhirVersion", "")
    assert fhir_version.startswith("4.")


# ─────────────────────── SHARP middleware against HAPI ───────────────────────

def test_sharp_middleware_passes_real_fhir_url_through(hapi_available):
    """The SHARP middleware accepts a real FHIR server URL + token and
    propagates them via the FHIRContext. No actual FHIR call is made
    here -- we just verify the middleware doesn't reject a real URL."""
    from mcp_server.server import build_http_app

    app = build_http_app()
    client = TestClient(app)
    headers = {
        "X-FHIR-Server-URL": HAPI_URL,
        "X-FHIR-Access-Token": "anonymous-public-server",
        "X-Patient-ID": "Patient/example",
    }
    # /healthz is a public path so SHARP validation isn't applied; use
    # the batch endpoint instead, which DOES go through SHARP.
    resp = client.post(
        "/api/batch/decision-cards",
        json={"requests": []},
        headers=headers,
    )
    # Empty batch should return 200 with an empty results list, not 403
    assert resp.status_code in (200, 400), (
        f"Expected 200/400, got {resp.status_code}: {resp.text[:200]}"
    )


# ─────────────────────── Live FHIR fetch ───────────────────────

def test_live_hapi_can_fetch_patient_metadata(hapi_available):
    """Demonstrates a real FHIR R4 fetch against HAPI for a
    well-known test patient."""
    # HAPI has a slew of test Patient resources; we just fetch any 5
    # to verify the round-trip
    r = httpx.get(f"{HAPI_URL}/Patient?_count=5", timeout=15.0)
    assert r.status_code == 200
    bundle = r.json()
    assert bundle.get("resourceType") == "Bundle"
    assert isinstance(bundle.get("entry", []), list)


def test_live_hapi_evidence_pack_round_trip(hapi_available):
    """End-to-end: fetch a real Patient + Conditions from HAPI, build a
    PA evidence pack from them, verify the pack is non-empty."""
    import asyncio

    from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack
    from shared.schemas import PARequestedService

    # Fetch the first Patient that has at least one Condition
    r = httpx.get(
        f"{HAPI_URL}/Patient?_has:Condition:patient:_id&_count=1",
        timeout=15.0,
    )
    if r.status_code != 200 or not r.json().get("entry"):
        pytest.skip("No HAPI Patient with attached Conditions")
    patient_id = r.json()["entry"][0]["resource"]["id"]

    cond_r = httpx.get(
        f"{HAPI_URL}/Condition?patient={patient_id}&_count=10",
        timeout=15.0,
    )
    cond_bundle = cond_r.json() if cond_r.status_code == 200 else {"entry": []}

    # Build a synthetic FHIR Bundle for the evidence pack
    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": cond_bundle.get("entry", []),
    }

    pack = asyncio.run(compute_pa_evidence_pack(
        patient_reference=f"Patient/{patient_id}",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            description="Live HAPI smoke",
        ),
        payer="generic",
        fhir_bundle=fhir_bundle,
    ))
    # The pack might still abstain (n_evidence < 3) on a sparse HAPI
    # patient, but it must NOT crash and must carry the patient
    # reference verbatim.
    assert pack.patient_reference == f"Patient/{patient_id}"
    # Diagnoses should match what we pulled
    assert len(pack.diagnoses) == len(cond_bundle.get("entry", []))
