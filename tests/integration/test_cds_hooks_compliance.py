"""INT-1 -- CDS Hooks v1.1 compliance test against our cds_hooks server.

Asserts the server adheres to the spec invariants documented in
https://cds-hooks.hl7.org/1.1/. Specifically:

  - GET /cds-services returns {"services": [...]} where each service has
    {hook, name, id, description}, plus optional `prefetch`.
  - POST /cds-services/{id} requires `hook` + `hookInstance`.
  - Response is {"cards": [...]} where each card has {summary, indicator,
    source.label}; indicator ∈ {info, warning, critical}.
  - SMART-on-FHIR launch info (`fhirServer`, `fhirAuthorization`) is
    accepted but optional.
  - Mismatched hook for a service yields 400.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cds_app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "cds_hooks_server",
        str(ROOT / "apps" / "cds_hooks" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cds_hooks_server"] = mod
    spec.loader.exec_module(mod)
    return mod.app


# ─────────────────────── Discovery endpoint ───────────────────────

def test_discovery_returns_services_shape(cds_app):
    from fastapi.testclient import TestClient
    with TestClient(cds_app) as c:
        r = c.get("/cds-services")
    assert r.status_code == 200
    body = r.json()
    assert "services" in body
    assert isinstance(body["services"], list)
    assert len(body["services"]) >= 1
    for svc in body["services"]:
        # Required per CDS Hooks v1.1: hook, id, description
        assert "hook" in svc and svc["hook"]
        assert "id" in svc and svc["id"]
        assert "description" in svc and svc["description"]


def test_discovery_hook_values_are_v1_1(cds_app):
    """v1.1 supports patient-view, medication-prescribe, order-review/select, encounter-*."""
    from fastapi.testclient import TestClient
    valid_hooks = {"patient-view", "medication-prescribe",
                       "order-review", "order-select", "order-sign",
                       "encounter-start", "encounter-discharge",
                       "appointment-book"}
    with TestClient(cds_app) as c:
        services = c.get("/cds-services").json()["services"]
    for svc in services:
        assert svc["hook"] in valid_hooks


def test_discovery_includes_trustedrisk_services(cds_app):
    """We declared 3 services; ensure they all surface."""
    from fastapi.testclient import TestClient
    with TestClient(cds_app) as c:
        services = c.get("/cds-services").json()["services"]
    ids = {s["id"] for s in services}
    assert {"trustedrisk-patient-view",
              "trustedrisk-medication-prescribe",
              "trustedrisk-order-review"} <= ids


# ─────────────────────── Service invocation ───────────────────────

def test_invoke_missing_hook_returns_400(cds_app):
    from fastapi.testclient import TestClient
    with TestClient(cds_app) as c:
        r = c.post("/cds-services/trustedrisk-patient-view", json={})
    assert r.status_code == 400


def test_invoke_unknown_service_returns_404(cds_app):
    from fastapi.testclient import TestClient
    with TestClient(cds_app) as c:
        r = c.post("/cds-services/unknown-service",
                     json={"hook": "patient-view", "hookInstance": "x"})
    assert r.status_code == 404


def test_invoke_mismatched_hook_returns_400(cds_app):
    """A service registered for patient-view must reject medication-prescribe payloads."""
    from fastapi.testclient import TestClient
    with TestClient(cds_app) as c:
        r = c.post("/cds-services/trustedrisk-patient-view",
                     json={"hook": "medication-prescribe",
                            "hookInstance": "x",
                            "context": {"patientId": "pt-1"}})
    assert r.status_code == 400


# ─────────────────────── Card response shape ───────────────────────

_VALID_INDICATORS = {"info", "warning", "critical"}


def test_patient_view_returns_cards_shape(cds_app):
    from fastapi.testclient import TestClient
    payload = {
        "hook": "patient-view",
        "hookInstance": "abc-123",
        "context": {"userId": "Practitioner/p-1",
                       "patientId": "pt-001"},
        "prefetch": {},
    }
    with TestClient(cds_app) as c:
        r = c.post("/cds-services/trustedrisk-patient-view", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cards" in body
    for card in body["cards"]:
        assert "summary" in card and card["summary"]
        assert "indicator" in card and card["indicator"] in _VALID_INDICATORS
        assert "source" in card and "label" in card["source"]


def test_medication_prescribe_with_smart_on_fhir_launch(cds_app):
    """SMART launch info must be accepted (optional)."""
    from fastapi.testclient import TestClient
    payload = {
        "hook": "medication-prescribe",
        "hookInstance": "med-instance-1",
        "context": {
            "userId": "Practitioner/p-1",
            "patientId": "pt-001",
            "medications": {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [{"resource": {
                    "resourceType": "MedicationRequest",
                    "id": "mr-1",
                    "status": "draft",
                    "intent": "order",
                    "subject": {"reference": "Patient/pt-001"},
                    "medicationCodeableConcept": {
                        "text": "warfarin 5 mg"},
                }}],
            },
        },
        "fhirServer": "https://hapi.fhir.org/baseR4",
        "fhirAuthorization": {
            "access_token": "test-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "patient/*.read",
            "subject": "client-id-x",
        },
        "prefetch": {},
    }
    with TestClient(cds_app) as c:
        r = c.post("/cds-services/trustedrisk-medication-prescribe",
                     json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cards" in body


def test_order_review_basic_invocation(cds_app):
    from fastapi.testclient import TestClient
    payload = {
        "hook": "order-review",
        "hookInstance": "order-instance-1",
        "context": {
            "userId": "Practitioner/p-1",
            "patientId": "pt-001",
            "draftOrders": {
                "resourceType": "Bundle",
                "entry": [{"resource": {
                    "resourceType": "ServiceRequest",
                    "id": "sr-1",
                    "status": "draft",
                    "intent": "order",
                    "subject": {"reference": "Patient/pt-001"},
                    "code": {"text": "CT abdomen with contrast"},
                }}],
            },
        },
    }
    with TestClient(cds_app) as c:
        r = c.post("/cds-services/trustedrisk-order-review",
                     json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cards" in body


# ─────────────────────── Optional: card detail / links structure ───────────────────────

def test_cards_with_detail_or_links_well_formed(cds_app):
    """When cards include `detail` or `links`, those fields conform to spec."""
    from fastapi.testclient import TestClient
    payload = {
        "hook": "patient-view",
        "hookInstance": "x",
        "context": {"userId": "Practitioner/p-1", "patientId": "pt-001"},
        "prefetch": {},
    }
    with TestClient(cds_app) as c:
        body = c.post("/cds-services/trustedrisk-patient-view",
                         json=payload).json()
    for card in body["cards"]:
        if "detail" in card:
            assert isinstance(card["detail"], str)
        if "links" in card:
            assert isinstance(card["links"], list)
            for link in card["links"]:
                assert "label" in link
                assert "url" in link
                if "type" in link:
                    assert link["type"] in {"absolute", "smart"}


# ─────────────────────── CORS posture (CDS clients are browser-side) ───────────────────────

def test_post_cds_service_accepts_json_content(cds_app):
    """Idempotent invocation should not reject application/json explicitly."""
    from fastapi.testclient import TestClient
    payload = {"hook": "patient-view", "hookInstance": "x",
                 "context": {"userId": "p", "patientId": "pt-1"}}
    with TestClient(cds_app) as c:
        r = c.post("/cds-services/trustedrisk-patient-view",
                     json=payload,
                     headers={"Content-Type": "application/json"})
    assert r.status_code == 200
