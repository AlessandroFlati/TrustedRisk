"""Phase 13.10 F1 -- EHR FHIR Composition write-back tests.

Mocks httpx.AsyncClient.put so we don't need a live FHIR server."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.fhir_writeback import compute_write_decision_to_fhir
from mcp_server.sharp.headers import FHIRContext, _fhir_ctx


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
    def json(self):
        return self._body


class _RecordingClient:
    """Stub httpx.AsyncClient that records every PUT and returns a
    canned response."""
    def __init__(self, responses: list[tuple[int, dict]]):
        self._responses = list(responses)
        self.calls: list[dict] = []
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def put(self, url, *, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._responses:
            sc, body = self._responses.pop(0)
        else:
            sc, body = 201, {"id": "stub-id"}
        return _FakeResponse(sc, body)


def _bind_ctx():
    return _fhir_ctx.set(FHIRContext(
        server_url="https://fhir.test/baseR4",
        access_token="bearer-test", patient_id="pt-test-1",
    ))


def _unbind_ctx(token):
    _fhir_ctx.reset(token)


def _patch_httpx(monkeypatch, fake_client: _RecordingClient):
    import httpx
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: fake_client,
    )
    return fake_client


# ─────────────────────────────────────────────────────────────────────
# Happy path -- both PUTs return 201
# ─────────────────────────────────────────────────────────────────────

def test_write_decision_to_fhir_succeeds_when_both_resources_201(monkeypatch):
    fake = _RecordingClient([
        (201, {"id": "decision-from-server"}),
        (201, {"id": "prov-from-server"}),
    ])
    _patch_httpx(monkeypatch, fake)
    tok = _bind_ctx()
    try:
        rep = _run(compute_write_decision_to_fhir(
            decision_summary="Discharge home; close PCP follow-up.",
            decision_card_logical_id="dc-1234",
            recommended_action="discharge_home",
            rationale_text="LACE 6, no abstain triggers.",
            cited_resource_ids=["Condition/c-1", "Observation/o-7"],
        ))
    finally:
        _unbind_ctx(tok)
    assert rep.write_succeeded is True
    assert rep.composition_status_code == 201
    assert rep.provenance_status_code == 201
    assert rep.composition_id == "decision-from-server"
    # Two PUT calls
    assert len(fake.calls) == 2
    # Idempotent identifier query in URL
    for c in fake.calls:
        assert "identifier=https://trustedrisk.local/decision-card-id|" in c["url"]


# ─────────────────────────────────────────────────────────────────────
# Idempotent: same logical id -> same composition id
# ─────────────────────────────────────────────────────────────────────

def test_same_logical_id_yields_same_composition_id(monkeypatch):
    fake1 = _RecordingClient([(201, {}), (201, {})])
    fake2 = _RecordingClient([(200, {}), (200, {})])

    import httpx
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **k: fake1,
    )
    tok = _bind_ctx()
    try:
        rep1 = _run(compute_write_decision_to_fhir(
            decision_summary="x", decision_card_logical_id="dc-stable",
        ))
    finally:
        _unbind_ctx(tok)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake2)
    tok = _bind_ctx()
    try:
        rep2 = _run(compute_write_decision_to_fhir(
            decision_summary="x", decision_card_logical_id="dc-stable",
        ))
    finally:
        _unbind_ctx(tok)

    # Server didn't echo back an id; the tool falls back to the stable
    # local id derived from the identifier
    assert rep1.composition_id == rep2.composition_id
    assert rep1.provenance_id == rep2.provenance_id


# ─────────────────────────────────────────────────────────────────────
# Failure path -- Composition 4xx
# ─────────────────────────────────────────────────────────────────────

def test_write_decision_failure_when_composition_returns_400(monkeypatch):
    fake = _RecordingClient([
        (400, {"resourceType": "OperationOutcome",
               "issue": [{"diagnostics": "Invalid identifier"}]}),
        (201, {}),    # Provenance also attempted (defensive)
    ])
    _patch_httpx(monkeypatch, fake)
    tok = _bind_ctx()
    try:
        rep = _run(compute_write_decision_to_fhir(
            decision_summary="x", decision_card_logical_id="dc-bad",
        ))
    finally:
        _unbind_ctx(tok)
    assert rep.write_succeeded is False
    assert rep.composition_status_code == 400


# ─────────────────────────────────────────────────────────────────────
# Body shape -- Composition contents
# ─────────────────────────────────────────────────────────────────────

def test_composition_payload_lists_subject_and_section(monkeypatch):
    fake = _RecordingClient([(201, {}), (201, {})])
    _patch_httpx(monkeypatch, fake)
    tok = _bind_ctx()
    try:
        _run(compute_write_decision_to_fhir(
            decision_summary="Discharge",
            decision_card_logical_id="dc-x",
            recommended_action="discharge_home",
            rationale_text="LACE 6.",
            cited_resource_ids=["Condition/c-1"],
        ))
    finally:
        _unbind_ctx(tok)
    comp_call = fake.calls[0]
    body = comp_call["json"]
    assert body["resourceType"] == "Composition"
    assert body["subject"]["reference"] == "Patient/pt-test-1"
    section_titles = [s["title"] for s in body["section"]]
    assert "Recommendation" in section_titles
    assert "Rationale" in section_titles


def test_provenance_payload_targets_composition(monkeypatch):
    fake = _RecordingClient([(201, {}), (201, {})])
    _patch_httpx(monkeypatch, fake)
    tok = _bind_ctx()
    try:
        _run(compute_write_decision_to_fhir(
            decision_summary="x",
            decision_card_logical_id="dc-prov-1",
        ))
    finally:
        _unbind_ctx(tok)
    prov_call = fake.calls[1]
    body = prov_call["json"]
    assert body["resourceType"] == "Provenance"
    assert body["target"][0]["reference"].startswith("Composition/")


# ─────────────────────────────────────────────────────────────────────
# Bearer token forwarded
# ─────────────────────────────────────────────────────────────────────

def test_bearer_token_forwarded_to_fhir_server(monkeypatch):
    fake = _RecordingClient([(201, {}), (201, {})])
    _patch_httpx(monkeypatch, fake)
    tok = _bind_ctx()
    try:
        _run(compute_write_decision_to_fhir(
            decision_summary="x", decision_card_logical_id="dc-auth",
        ))
    finally:
        _unbind_ctx(tok)
    headers = fake.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer bearer-test"
    assert headers["Content-Type"] == "application/fhir+json"


# ─────────────────────────────────────────────────────────────────────
# Bundle + scope wiring
# ─────────────────────────────────────────────────────────────────────

def test_write_back_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "fhir_writeback" in BUNDLES
    assert BUNDLES["fhir_writeback"] == ["compute_write_decision_to_fhir"]


def test_write_back_scopes_request_create_update_rights():
    from mcp_server.scopes import BUNDLE_SCOPES
    scopes = BUNDLE_SCOPES["fhir_writeback"]
    assert any(s == "patient/Composition.cu" for s in scopes)
    assert any(s == "patient/Provenance.cu" for s in scopes)
