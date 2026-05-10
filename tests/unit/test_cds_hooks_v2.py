"""Phase 11.5 -- CDS Hooks 2.0 service tests.

Validates the upgrade from v1.1 to v2.0:
  - new services (`order-select`, `order-sign`) advertised in discovery
  - every emitted card carries a `uuid`
  - `overrideReasons[]` present on warning/critical cards
  - `systemActions` returned by `order-sign` on critical DDIs
  - `feedbackEndpoint` honoured at /cds-services/{id}/feedback
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.cds_hooks.server import (
    app,
    clear_feedback_log,
    get_feedback_log,
)


client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────

def test_discovery_lists_v2_services():
    r = client.get("/cds-services")
    assert r.status_code == 200
    services = r.json()["services"]
    ids = {s["id"] for s in services}
    assert "trustedrisk-order-select" in ids
    assert "trustedrisk-order-sign" in ids


def test_v2_services_advertise_feedback_endpoint():
    services = client.get("/cds-services").json()["services"]
    for sid in ("trustedrisk-order-select", "trustedrisk-order-sign"):
        s = next(svc for svc in services if svc["id"] == sid)
        assert s["feedbackEndpoint"].startswith("/cds-services/")
        assert s["feedbackEndpoint"].endswith("/feedback")


def test_v2_services_carry_calibration_extension():
    services = client.get("/cds-services").json()["services"]
    s = next(svc for svc in services
                if svc["id"] == "trustedrisk-order-select")
    assert s.get("extension", {}).get(
        "com.trustedrisk.calibration_version") == "spec_002"


def test_v1_services_still_advertised_for_backcompat():
    services = client.get("/cds-services").json()["services"]
    ids = {s["id"] for s in services}
    assert "trustedrisk-patient-view" in ids
    assert "trustedrisk-medication-prescribe" in ids


# ─────────────────────────────────────────────────────────────────────
# Card 2.0 wiring -- uuid + overrideReasons
# ─────────────────────────────────────────────────────────────────────

def _critical_ddi_request(hook_id: str = "order-select") -> dict:
    return {
        "hook": hook_id,
        "hookInstance": "test-instance-1",
        "fhirServer": "http://example.com/fhir",
        "context": {
            "userId": "Practitioner/test-user",
            "patientId": "Patient/test-pt",
            "medications": {
                "resourceType": "Bundle",
                "entry": [{
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "status": "draft",
                        "medicationCodeableConcept": {"text": "warfarin"},
                    },
                }],
            },
            "selections": ["MedicationRequest/draft-1"],
        },
        "prefetch": {
            "active_meds": {
                "resourceType": "Bundle",
                "entry": [
                    {"resource": {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "medicationCodeableConcept": {
                            "text": "ibuprofen 600 mg",
                        },
                    }},
                    {"resource": {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "medicationCodeableConcept": {
                            "text": "amiodarone 200 mg",
                        },
                    }},
                ],
            },
        },
    }


def test_order_select_returns_card_with_uuid():
    body = _critical_ddi_request("order-select")
    r = client.post("/cds-services/trustedrisk-order-select", json=body)
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert cards
    for card in cards:
        assert "uuid" in card and len(card["uuid"]) >= 16


def test_order_select_warning_card_carries_override_reasons():
    body = _critical_ddi_request("order-select")
    r = client.post("/cds-services/trustedrisk-order-select", json=body)
    cards = r.json()["cards"]
    elevated = [c for c in cards if c["indicator"] in ("warning", "critical")]
    assert elevated, "Expected at least one warning/critical card"
    for c in elevated:
        assert "overrideReasons" in c
        assert any(o.get("code") == "clinically-justified"
                       for o in c["overrideReasons"])


# ─────────────────────────────────────────────────────────────────────
# order-sign systemActions
# ─────────────────────────────────────────────────────────────────────

def test_order_sign_emits_system_actions_for_critical_ddi():
    body = _critical_ddi_request("order-sign")
    r = client.post("/cds-services/trustedrisk-order-sign", json=body)
    payload = r.json()
    elevated = [c for c in payload["cards"]
                    if c["indicator"] == "critical"]
    if elevated:
        assert "systemActions" in payload
        assert any(a["type"] == "update"
                       for a in payload["systemActions"])


def test_order_sign_omits_system_actions_when_no_critical():
    """A clean med list should not trigger systemActions."""
    body = {
        "hook": "order-sign",
        "hookInstance": "test-instance-clean",
        "fhirServer": "http://example.com/fhir",
        "context": {
            "userId": "Practitioner/test-user",
            "patientId": "Patient/test-pt",
            "medications": {
                "resourceType": "Bundle",
                "entry": [{
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "status": "draft",
                        "medicationCodeableConcept": {
                            "text": "vitamin D 1000 IU",
                        },
                    },
                }],
            },
        },
        "prefetch": {
            "active_meds": {"resourceType": "Bundle", "entry": []},
        },
    }
    r = client.post("/cds-services/trustedrisk-order-sign", json=body)
    payload = r.json()
    # systemActions only present when at least one critical card emitted
    if not any(c["indicator"] == "critical" for c in payload["cards"]):
        assert "systemActions" not in payload


# ─────────────────────────────────────────────────────────────────────
# Feedback channel
# ─────────────────────────────────────────────────────────────────────

def test_feedback_endpoint_records_payload():
    clear_feedback_log()
    fb = {
        "feedback": [{
            "card": "fa6f9d12-card-uuid",
            "outcome": "overridden",
            "outcomeTimestamp": "2026-04-30T10:00:00Z",
            "overrideReason": {
                "reason": {"code": "clinically-justified",
                              "system": "https://cds-hooks.trustedrisk.local/override-reason"},
                "userComment": "Patient stable on warfarin INR 2.4 last week.",
            },
        }],
    }
    r = client.post(
        "/cds-services/trustedrisk-order-select/feedback", json=fb,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["service_id"] == "trustedrisk-order-select"
    assert body["n_recorded"] == 1
    log = get_feedback_log()
    assert len(log) == 1
    clear_feedback_log()


def test_feedback_endpoint_rejects_unknown_service():
    r = client.post("/cds-services/no-such-service/feedback",
                       json={"feedback": []})
    assert r.status_code == 404


def test_feedback_endpoint_rejects_malformed_payload():
    r = client.post(
        "/cds-services/trustedrisk-order-select/feedback",
        json={"not_feedback": []},
    )
    assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────
# Hook-name mismatch must 400
# ─────────────────────────────────────────────────────────────────────

def test_invoke_with_wrong_hook_name_returns_400():
    body = _critical_ddi_request("patient-view")  # wrong hook
    r = client.post("/cds-services/trustedrisk-order-select", json=body)
    assert r.status_code == 400


def test_invoke_unknown_service_returns_404():
    body = _critical_ddi_request("order-select")
    r = client.post("/cds-services/no-such-service", json=body)
    assert r.status_code == 404
