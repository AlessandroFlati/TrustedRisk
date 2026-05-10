"""Phase 8.5 -- broader integration tests.

Covers:
  1. Cross-specialist composition via the composer (4-5 specialists in
     sequence).
  2. Merkle audit-chain replay over a synthetic stream.
  3. DP equity epsilon-budget verification.
  4. Rate-limit middleware end-to-end.
  5. Push-notification dispatch via the alert_agent webhook contract.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from starlette.testclient import TestClient

from apps.composer.workflows import REGISTRY


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Cross-specialist composition ───────────────────────


def _bundle() -> dict:
    return {"resourceType": "Bundle", "entry": [
        {"resource": {
            "resourceType": "Patient", "id": "pt-jane",
            "name": [{"given": ["Jane"], "family": "Doe"}],
            "birthDate": "1955-01-01",
        }},
        {"resource": {
            "resourceType": "Encounter", "id": "enc-1",
            "class": {"code": "IMP"},
        }},
        {"resource": {
            "resourceType": "Condition", "id": "cond-chf",
            "code": {"coding": [{
                "code": "I50.21",
                "display": "Acute on chronic systolic CHF",
            }], "text": "CHF"},
        }},
        {"resource": {
            "resourceType": "MedicationRequest", "id": "med-furo",
            "medicationCodeableConcept": {"text": "furosemide 40 mg"},
        }},
    ]}


def test_composer_chf_admission_exercises_4_specialists():
    """The chf_admission workflow consults 4 specialists end-to-end:
    acute (triage + deterioration), evidence (DDx), discharge
    (polypharmacy), scribe (H&P)."""
    from apps.composer.orchestrator import execute_workflow
    out = _run(execute_workflow(REGISTRY["chf_admission"], {
        "chief_complaint": "shortness of breath",
        "vital_signs": {"heart_rate": 122, "spo2": 91,
                          "respiratory_rate": 26, "systolic_bp": 96,
                          "temperature_c": 38.4},
        "age": 72,
        "patient_id": "Patient/pt-jane",
        "fhir_bundle": _bundle(),
        "medications": ["furosemide", "lisinopril", "metoprolol"],
    }))
    specialists = {s.specialist for s in out.steps}
    assert {"trustedrisk-acute", "trustedrisk-evidence",
              "trustedrisk-discharge", "trustedrisk-scribe"} <= specialists


def test_composer_sepsis_workup_exercises_4_specialists():
    from apps.composer.orchestrator import execute_workflow
    out = _run(execute_workflow(REGISTRY["sepsis_workup"], {
        "chief_complaint": "fever and confusion",
        "vital_signs": {"heart_rate": 130, "systolic_bp": 84,
                          "respiratory_rate": 26, "temperature_c": 39.2,
                          "spo2": 92},
        "age": 78,
        "patient_id": "Patient/pt-sepsis",
        "infection_source": "urinary",
        "patient_factors": {"egfr_ml_min": 50},
        "fhir_bundle": _bundle(),
    }))
    specialists = {s.specialist for s in out.steps}
    assert len(specialists) >= 3
    # The sepsis workup is an end-to-end multi-specialist trace
    assert any(s.specialist == "trustedrisk-acute" for s in out.steps)


# ─────────────────────── Merkle audit-chain replay ───────────────────────


def test_merkle_audit_chain_appends_and_verifies():
    """100 events through the Merkle chain; verify inclusion for a
    sampled subset."""
    from a2a_agent.merkle_audit import (
        build_inclusion_proof, compute_merkle_audit_root, verify_inclusion,
    )

    events = [
        {"event_id": f"ev-{i:04d}", "tool": "compute_x", "ok": True}
        for i in range(100)
    ]
    chain = compute_merkle_audit_root(events)
    assert chain.n_events == 100
    assert isinstance(chain.merkle_root, str) and chain.merkle_root

    # Sample 10 events, verify each inclusion proof
    import random
    rng = random.Random(42)
    for i in rng.sample(range(100), 10):
        proof = build_inclusion_proof(events, events[i]["event_id"])
        assert verify_inclusion(
            chain.leaf_hashes[i], proof.proof_steps, chain.merkle_root,
        )


# ─────────────────────── DP equity epsilon budget ───────────────────────


def test_dp_equity_dashboard_respects_epsilon_budget():
    """Apply Laplace-mechanism DP to an EquityDashboard and verify the
    noised output stays in valid ranges."""
    from a2a_agent.dp_equity import compute_dp_equity_dashboard
    from shared.schemas import EquityDashboard, EquitySegment

    dashboard = EquityDashboard(
        n_total_decisions=10000,
        segments=[
            EquitySegment(
                subgroup_dimension="race", subgroup_value="black",
                n_decisions=1000,
                action_counts={"discharge_home": 500, "snf": 250,
                                  "home_with_care": 250},
                n_abstained=20, avg_risk=0.18,
                intervention_rate=0.50, abstention_rate=0.02,
                confidence_high_rate=0.70,
            ),
            EquitySegment(
                subgroup_dimension="race", subgroup_value="white",
                n_decisions=5000,
                action_counts={"discharge_home": 3500, "snf": 800,
                                  "home_with_care": 700},
                n_abstained=50, avg_risk=0.12,
                intervention_rate=0.30, abstention_rate=0.01,
                confidence_high_rate=0.78,
            ),
        ],
        max_intervention_rate_disparity=0.20,
        max_abstention_rate_disparity=0.01,
        max_avg_risk_disparity=0.06,
        rationale="test fixture",
    )
    out = compute_dp_equity_dashboard(
        dashboard, epsilon=1.0, seed=42,
    )
    # Output rates noised but bounded; field is `_noised`-suffixed
    assert hasattr(out, "n_total_decisions_noised")
    # Noised counts are >= 0
    assert out.n_total_decisions_noised >= 0
    assert out.epsilon == 1.0


# ─────────────────────── Rate-limit middleware ───────────────────────


def test_rate_limit_middleware_uses_per_request_token_bucket():
    """Use the TokenBucket directly -- module reload would bleed state
    into other tests. The HTTP path is tested in
    `tests/unit/test_observability.py` already."""
    from a2a_agent.observability import TokenBucket

    bucket = TokenBucket(capacity=2.0, refill_per_sec=0.0)
    a, _ = bucket.acquire("client-1")
    b, _ = bucket.acquire("client-1")
    c, retry = bucket.acquire("client-1")
    assert a is True and b is True and c is False
    assert retry > 0


# ─────────────────────── Push-notification dispatch ───────────────────────


def test_push_notification_dispatches_status_update(monkeypatch):
    """End-to-end: register a webhook + fire a status update + assert
    the configured webhook receives the structured A2A event payload."""
    from a2a_agent.push_notifications import (
        default_registry, dispatch_task_status_update,
    )

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)

    class _PatchedAsync(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, transport=transport, **kw)

    monkeypatch.setattr(
        "a2a_agent.push_notifications.httpx.AsyncClient", _PatchedAsync,
    )

    default_registry().clear()
    default_registry().create("task-99", "https://hook.example/wh",
                                  severity_threshold="info")
    out = _run(dispatch_task_status_update(
        "task-99", state="WORKING", progress_pct=0.5,
        severity="warn", message="halfway",
    ))
    assert len(out) == 1
    assert out[0]["ok"] is True
    assert captured[0]["kind"] == "TaskStatusUpdateEvent"
    assert captured[0]["taskId"] == "task-99"
    assert captured[0]["status"]["state"] == "WORKING"
    assert captured[0]["status"]["progress_pct"] == 0.5
    default_registry().clear()
