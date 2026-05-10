"""Unit tests for the CDS Hooks server (CDS-1 + CDS-2)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_cds():
    spec = importlib.util.spec_from_file_location(
        "cds_hooks_server",
        str(ROOT / "apps" / "cds_hooks" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cds_hooks_server"] = mod
    spec.loader.exec_module(mod)
    return mod


_cds = _load_cds()


# ─────────────────────── Discovery (CDS Hooks v1.1 spec) ───────────────────────

def test_discovery_returns_services_array():
    """Per spec: GET /cds-services returns {"services": [...]} with each
    service having required fields hook, title, description, id."""
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.get("/cds-services")
    assert r.status_code == 200
    data = r.json()
    assert "services" in data
    assert isinstance(data["services"], list)
    assert len(data["services"]) >= 3   # at least patient-view, prescribe, order-review
    for s in data["services"]:
        assert "id" in s
        assert "hook" in s
        assert "title" in s
        assert "description" in s


def test_discovery_includes_patient_view():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.get("/cds-services")
    services = r.json()["services"]
    pv = next((s for s in services if s["hook"] == "patient-view"), None)
    assert pv is not None
    # prefetch is recommended
    assert "prefetch" in pv


# ─────────────────────── Service invocation contract ───────────────────────

def test_invoke_unknown_service_404():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/totally-fake-service",
        json={"hook": "patient-view", "hookInstance": "x",
              "context": {"patientId": "p1"}},
    )
    assert r.status_code == 404


def test_invoke_missing_hook_400():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-patient-view",
        json={"context": {}},  # missing hook + hookInstance
    )
    assert r.status_code == 400


def test_invoke_wrong_hook_for_service_400():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-patient-view",
        json={"hook": "medication-prescribe",  # wrong hook for this service
              "hookInstance": "x", "context": {}},
    )
    assert r.status_code == 400


# ─────────────────────── patient-view hook ───────────────────────

def test_patient_view_returns_cards():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-patient-view",
        json={
            "hook": "patient-view",
            "hookInstance": "test-1",
            "context": {"userId": "Practitioner/dr-smith",
                         "patientId": "Patient/p1"},
            "prefetch": {
                "patient": {"resourceType": "Patient", "id": "p1",
                              "birthDate": "1950-01-01"},
                "conditions": {"resourceType": "Bundle", "entry": [
                    {"resource": {"resourceType": "Condition",
                                    "code": {"text": "Heart failure"}}},
                ]},
                "medications": {"resourceType": "Bundle", "entry": []},
            },
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "cards" in data
    assert isinstance(data["cards"], list)
    # Each card must have summary + indicator + source
    for card in data["cards"]:
        assert "summary" in card
        assert "indicator" in card
        assert card["indicator"] in ("info", "warning", "critical")
        assert "source" in card
        assert "label" in card["source"]


def test_patient_view_summary_under_140_chars():
    """CDS Hooks v1.1 spec mandates summary ≤ 140 chars."""
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-patient-view",
        json={
            "hook": "patient-view", "hookInstance": "x",
            "context": {"patientId": "Patient/p1"},
            "prefetch": {"patient": {"resourceType": "Patient", "id": "p1"}},
        },
    )
    for card in r.json()["cards"]:
        assert len(card["summary"]) <= 140


# ─────────────────────── medication-prescribe hook ───────────────────────

def test_medication_prescribe_no_meds_returns_info_card():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-medication-prescribe",
        json={
            "hook": "medication-prescribe", "hookInstance": "x",
            "context": {"patientId": "Patient/p1",
                          "medications": {"resourceType": "Bundle", "entry": []}},
            "prefetch": {"active_meds": {"resourceType": "Bundle", "entry": []}},
        },
    )
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert any("no medications" in c["summary"].lower() for c in cards)


def test_medication_prescribe_warfarin_plus_nsaid_high_severity():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-medication-prescribe",
        json={
            "hook": "medication-prescribe", "hookInstance": "x",
            "context": {
                "patientId": "Patient/p1",
                "medications": {"resourceType": "Bundle", "entry": [
                    {"resource": {"medicationCodeableConcept":
                                    {"text": "ibuprofen 400mg PO TID"},
                                    "status": "draft"}},
                ]},
            },
            "prefetch": {
                "active_meds": {"resourceType": "Bundle", "entry": [
                    {"resource": {"medicationCodeableConcept":
                                    {"text": "warfarin 5mg PO daily"},
                                    "status": "active"}},
                ]},
            },
        },
    )
    cards = r.json()["cards"]
    # Should detect the high-severity warfarin + NSAID DDI
    critical = [c for c in cards if c["indicator"] == "critical"]
    assert len(critical) >= 1
    # Critical card should mention bleeding risk or warfarin
    text = " ".join(c["summary"] + " " + c.get("detail", "") for c in critical).lower()
    assert "warfarin" in text or "bleeding" in text or "ibuprofen" in text


def test_medication_prescribe_card_has_suggestions():
    """When a high-severity DDI is detected, the card should offer a
    structured suggestion the EHR can act on."""
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-medication-prescribe",
        json={
            "hook": "medication-prescribe", "hookInstance": "x",
            "context": {
                "patientId": "Patient/p1",
                "medications": {"resourceType": "Bundle", "entry": [
                    {"resource": {"medicationCodeableConcept":
                                    {"text": "ibuprofen"}, "status": "draft"}},
                ]},
            },
            "prefetch": {
                "active_meds": {"resourceType": "Bundle", "entry": [
                    {"resource": {"medicationCodeableConcept":
                                    {"text": "warfarin"}, "status": "active"}},
                ]},
            },
        },
    )
    cards = r.json()["cards"]
    # Find any card that carries suggestions (the high-severity DDI one)
    cards_with_suggestions = [c for c in cards if "suggestions" in c]
    assert len(cards_with_suggestions) >= 1
    sug = cards_with_suggestions[0]["suggestions"][0]
    # Per spec, each suggestion has a label + uuid + actions
    assert "label" in sug
    assert "uuid" in sug
    assert "actions" in sug


# ─────────────────────── order-review hook ───────────────────────

def test_order_review_contrast_with_low_egfr_blocks():
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-order-review",
        json={
            "hook": "order-review", "hookInstance": "x",
            "context": {
                "patientId": "Patient/p1",
                "draftOrders": {"resourceType": "Bundle", "entry": [
                    {"resource": {"resourceType": "ServiceRequest",
                                    "code": {"text": "CT abdomen with IV contrast"}}},
                ]},
            },
            "prefetch": {
                "egfr": {"resourceType": "Bundle", "entry": [
                    {"resource": {"valueQuantity": {"value": 20},
                                    "effectiveDateTime": "2026-04-26T08:00:00Z"}},
                ]},
            },
        },
    )
    cards = r.json()["cards"]
    critical_or_warning = [c for c in cards
                              if c["indicator"] in ("critical", "warning")]
    assert len(critical_or_warning) >= 1


# ─────────────────────── SMART-on-FHIR launch context ───────────────────────

def test_smart_fhir_authorization_accepted():
    """The fhirAuthorization block + fhirServer carry SMART context per
    spec §SMART-on-FHIR. The server must accept them without error."""
    from fastapi.testclient import TestClient
    client = TestClient(_cds.app)
    r = client.post(
        "/cds-services/trustedrisk-patient-view",
        json={
            "hook": "patient-view", "hookInstance": "x",
            "fhirServer": "https://hapi.example.com/fhir",
            "fhirAuthorization": {
                "access_token": "fake-bearer-token",
                "token_type": "Bearer",
                "expires_in": 300,
                "scope": "patient/*.read",
                "subject": "client-id",
            },
            "context": {"userId": "Practitioner/x",
                          "patientId": "Patient/p1"},
            "prefetch": {"patient": {"resourceType": "Patient", "id": "p1"}},
        },
    )
    assert r.status_code == 200
    # The hook must succeed and return cards
    assert "cards" in r.json()


# ─────────────────────── Indicator mapping ───────────────────────

@pytest.mark.parametrize("severity,abstain,expected", [
    (None, 0, "info"),
    ("low", 0, "info"),
    ("medium", 0, "warning"),
    ("moderate", 0, "warning"),
    ("high", 0, "critical"),
    ("severe", 0, "critical"),
    ("imminent", 0, "critical"),
    ("low", 1, "critical"),    # any abstain -> critical
    ("medium", 2, "critical"),
])
def test_indicator_mapping(severity, abstain, expected):
    assert _cds._indicator_for(severity, abstain) == expected
