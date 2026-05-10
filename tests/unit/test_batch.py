"""Unit tests for the deterministic batch composer."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from a2a_agent.batch import (
    compose_decision_for_patient,
    parse_batch_request,
    process_batch,
    recommend_action_from_risk,
    serialize_batch_response,
)
from shared.schemas import (
    Action,
    BatchPatientRequest,
    BatchRequest,
    Factor,
    FairnessReport,
    MedReconReport,
    RiskEstimate,
    SubgroupCalibration,
)


def _run(coro):
    return asyncio.run(coro)


def _make_risk(*, prob_mean: float = 0.15,
               ci_width: float = 0.10,
               lace: int = 8) -> RiskEstimate:
    half = ci_width / 2.0
    lo = max(0.0, prob_mean - half)
    hi = min(1.0, prob_mean + half)
    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="test-v1",
        horizon_days=30,
        lace_raw_score=lace,
        probability_mean=prob_mean,
        probability_ci95=(lo, hi),
        probability_ci_width=hi - lo,
        contributing_factors=[
            Factor(name="LACE_length_of_stay", raw_value=4.0,
                   lace_points=4, weight=0.5),
        ],
        fhir_observations_used=[],
        computed_at=datetime.now(timezone.utc),
    )


def _make_medrecon() -> MedReconReport:
    return MedReconReport(
        added=[], removed=[], dose_changed=[], concerns=[],
        severity_counts={},
        discharge_contract_satisfied=True,
        n_admission_meds=2, n_discharge_meds=2,
        monitoring_window_hours=48,
    )


def _make_fairness(action: str = "no_action") -> FairnessReport:
    return FairnessReport(
        n_subgroups_assessed=1,
        subgroup_drifts=[
            SubgroupCalibration(
                subgroup_name="age_band",
                subgroup_value="65-74",
                expected_rate_baseline=0.18,
                predicted_rate_for_patient=0.15,
                relative_drift=-0.16,
                severity="low",
                citation="test",
            ),
        ],
        max_relative_drift=0.16,
        confidence_action=action,  # type: ignore[arg-type]
        rationale="test",
    )


# ─────────────────────── recommend_action_from_risk ───────────────────────

def test_recommend_discharge_home_low_prob():
    risk = _make_risk(prob_mean=0.05)
    rec = recommend_action_from_risk(risk)
    assert rec.action == Action.DISCHARGE_HOME
    assert rec.confidence == "high"


def test_recommend_home_with_care_moderate_prob():
    risk = _make_risk(prob_mean=0.15)
    rec = recommend_action_from_risk(risk)
    assert rec.action == Action.HOME_WITH_CARE
    assert rec.confidence == "medium"


def test_recommend_snf_elevated_prob():
    risk = _make_risk(prob_mean=0.25)
    rec = recommend_action_from_risk(risk)
    assert rec.action == Action.SNF


def test_recommend_continued_admission_high_prob():
    risk = _make_risk(prob_mean=0.45)
    rec = recommend_action_from_risk(risk)
    assert rec.action == Action.CONTINUED_ADMISSION
    assert rec.confidence == "low"


# ─────────────────────── compose_decision_for_patient ───────────────────────

def _patch_tools(monkeypatch, *, risk=None, medrecon=None, fairness=None,
                 risk_raises=None, medrecon_raises=None, fairness_raises=None):
    """Stub the 3 tool functions imported into batch.py."""
    from a2a_agent import batch as bm

    async def _risk(horizon_days=30, patient_id=None):
        if risk_raises is not None:
            raise risk_raises
        return risk if risk is not None else _make_risk()

    async def _medrecon(patient_id=None, **kwargs):
        if medrecon_raises is not None:
            raise medrecon_raises
        return medrecon if medrecon is not None else _make_medrecon()

    async def _fairness(risk, patient_demographics):
        if fairness_raises is not None:
            raise fairness_raises
        return fairness if fairness is not None else _make_fairness()

    monkeypatch.setattr(bm, "compute_readmission_risk", _risk)
    monkeypatch.setattr(bm, "compute_medication_reconciliation", _medrecon)
    monkeypatch.setattr(bm, "compute_fairness_audit", _fairness)


def test_compose_happy_path(monkeypatch):
    _patch_tools(monkeypatch, risk=_make_risk(prob_mean=0.08))
    req = BatchPatientRequest(
        patient_id="pt-1",
        demographics={"age": 70, "race": "white"},
    )
    res = _run(compose_decision_for_patient(req))
    assert res.status == "ok"
    assert res.recommendation is not None
    assert res.recommendation.action == Action.DISCHARGE_HOME
    assert res.risk_estimate is not None
    assert res.medication_reconciliation is not None
    assert res.fairness is not None
    assert res.abstain == []


def test_compose_isolates_risk_failure(monkeypatch):
    _patch_tools(monkeypatch, risk_raises=RuntimeError("fhir down"))
    req = BatchPatientRequest(patient_id="pt-bad")
    res = _run(compose_decision_for_patient(req))
    assert res.status == "error"
    assert "fhir down" in (res.error or "")
    assert res.recommendation is None


def test_compose_medrecon_failure_is_non_fatal(monkeypatch):
    _patch_tools(monkeypatch,
                 risk=_make_risk(prob_mean=0.08),
                 medrecon_raises=RuntimeError("med list missing"))
    req = BatchPatientRequest(patient_id="pt-1")
    res = _run(compose_decision_for_patient(req))
    assert res.status == "ok"
    assert res.medication_reconciliation is None
    assert res.recommendation is not None  # risk still produced


def test_compose_fairness_failure_is_non_fatal(monkeypatch):
    _patch_tools(monkeypatch,
                 risk=_make_risk(prob_mean=0.08),
                 fairness_raises=RuntimeError("baseline corrupt"))
    req = BatchPatientRequest(
        patient_id="pt-1",
        demographics={"age": 70},
    )
    res = _run(compose_decision_for_patient(req))
    assert res.status == "ok"
    assert res.fairness is None
    assert res.recommendation is not None


def test_compose_abstains_on_wide_ci(monkeypatch):
    # CI width > 0.30 (default threshold) triggers confidence_interval_too_wide
    wide = _make_risk(prob_mean=0.50, ci_width=0.50)
    _patch_tools(monkeypatch, risk=wide)
    req = BatchPatientRequest(patient_id="pt-wide")
    res = _run(compose_decision_for_patient(req))
    assert res.status == "ok"
    assert res.recommendation is None
    assert any(t.type == "confidence_interval_too_wide" for t in res.abstain)


def test_compose_fairness_abstain_recommended_forces_abstain(monkeypatch):
    _patch_tools(monkeypatch,
                 risk=_make_risk(prob_mean=0.08),
                 fairness=_make_fairness(action="abstain_recommended"))
    req = BatchPatientRequest(
        patient_id="pt-1",
        demographics={"age": 88, "race": "black"},
    )
    res = _run(compose_decision_for_patient(req))
    assert res.recommendation is None
    assert any(t.type == "out_of_distribution" for t in res.abstain)


def test_compose_fairness_downgrade_lowers_confidence(monkeypatch):
    _patch_tools(monkeypatch,
                 risk=_make_risk(prob_mean=0.08),
                 fairness=_make_fairness(action="downgrade_confidence"))
    req = BatchPatientRequest(
        patient_id="pt-1",
        demographics={"age": 75, "race": "black"},
    )
    res = _run(compose_decision_for_patient(req))
    assert res.recommendation is not None
    assert res.recommendation.confidence == "low"


def test_compose_skips_medrecon_when_disabled(monkeypatch):
    _patch_tools(monkeypatch, risk=_make_risk(prob_mean=0.08))
    req = BatchPatientRequest(
        patient_id="pt-1",
        include_med_recon=False,
        include_fairness=False,
    )
    res = _run(compose_decision_for_patient(req))
    assert res.medication_reconciliation is None
    assert res.fairness is None


def test_compose_skips_fairness_when_no_demographics(monkeypatch):
    _patch_tools(monkeypatch, risk=_make_risk(prob_mean=0.08))
    req = BatchPatientRequest(
        patient_id="pt-1",
        demographics=None,
        include_fairness=True,
    )
    res = _run(compose_decision_for_patient(req))
    assert res.fairness is None  # demographics required to run audit


# ─────────────────────── process_batch ───────────────────────

def test_process_batch_concurrent_results_in_order(monkeypatch):
    _patch_tools(monkeypatch, risk=_make_risk(prob_mean=0.08))
    req = BatchRequest(
        patients=[
            BatchPatientRequest(patient_id=f"pt-{i}") for i in range(5)
        ],
        max_concurrency=3,
    )
    resp = _run(process_batch(req))
    assert resp.n_requested == 5
    assert resp.n_succeeded == 5
    assert resp.n_failed == 0
    assert resp.n_abstained == 0
    assert [r.patient_id for r in resp.results] == [f"pt-{i}" for i in range(5)]
    assert resp.duration_ms >= 0


def test_process_batch_partial_failure(monkeypatch):
    """One patient fails, others succeed."""
    from a2a_agent import batch as bm

    async def _risk(horizon_days=30, patient_id=None):
        if patient_id == "pt-bad":
            raise RuntimeError("fhir error for pt-bad")
        return _make_risk(prob_mean=0.08)

    async def _medrecon(patient_id=None, **kwargs):
        return _make_medrecon()

    async def _fairness(risk, patient_demographics):
        return _make_fairness()

    monkeypatch.setattr(bm, "compute_readmission_risk", _risk)
    monkeypatch.setattr(bm, "compute_medication_reconciliation", _medrecon)
    monkeypatch.setattr(bm, "compute_fairness_audit", _fairness)

    req = BatchRequest(
        patients=[
            BatchPatientRequest(patient_id="pt-1"),
            BatchPatientRequest(patient_id="pt-bad"),
            BatchPatientRequest(patient_id="pt-3"),
        ],
    )
    resp = _run(process_batch(req))
    assert resp.n_requested == 3
    assert resp.n_succeeded == 2
    assert resp.n_failed == 1
    bad = next(r for r in resp.results if r.patient_id == "pt-bad")
    assert bad.status == "error"
    assert "pt-bad" in (bad.error or "") or "fhir error" in (bad.error or "")


def test_process_batch_counts_abstained(monkeypatch):
    """Abstained patients count separately from failed."""
    wide = _make_risk(prob_mean=0.50, ci_width=0.50)
    _patch_tools(monkeypatch, risk=wide)
    req = BatchRequest(patients=[
        BatchPatientRequest(patient_id="pt-1"),
        BatchPatientRequest(patient_id="pt-2"),
    ])
    resp = _run(process_batch(req))
    assert resp.n_succeeded == 2  # status ok, just no recommendation
    assert resp.n_failed == 0
    assert resp.n_abstained == 2


# ─────────────────────── parse_batch_request / serialize ───────────────────────

def test_parse_batch_request_valid():
    payload = {
        "patients": [{"patient_id": "pt-1"}, {"patient_id": "pt-2"}],
        "max_concurrency": 4,
    }
    req = parse_batch_request(payload)
    assert len(req.patients) == 2
    assert req.max_concurrency == 4


def test_parse_batch_request_rejects_non_dict():
    with pytest.raises(ValueError):
        parse_batch_request([{"patient_id": "pt-1"}])


def test_parse_batch_request_validation_error():
    with pytest.raises(Exception):
        parse_batch_request({"patients": "not a list"})


def test_serialize_batch_response_roundtrip(monkeypatch):
    _patch_tools(monkeypatch, risk=_make_risk(prob_mean=0.08))
    req = BatchRequest(patients=[BatchPatientRequest(patient_id="pt-1")])
    resp = _run(process_batch(req))
    payload = serialize_batch_response(resp)
    assert isinstance(payload, dict)
    assert payload["n_requested"] == 1
    assert payload["n_succeeded"] == 1
    assert isinstance(payload["results"], list)
    assert payload["results"][0]["patient_id"] == "pt-1"
    # datetime fields serialised as ISO strings
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["completed_at"], str)


# ─────────────────────── HTTP endpoint smoke ───────────────────────

def _build_test_app(monkeypatch):
    """Construct the real Starlette app with tools stubbed for the batch path.

    OAuth is left disabled (default) so requests don't need bearer tokens;
    SHARP headers are still required."""
    monkeypatch.delenv("TRUSTEDRISK_OAUTH_ENABLED", raising=False)
    _patch_tools(monkeypatch, risk=_make_risk(prob_mean=0.08))
    from mcp_server.server import build_http_app
    return build_http_app()


def test_endpoint_rejects_request_without_sharp_headers(monkeypatch):
    """Without X-FHIR-Server-URL/Token, SHARP middleware returns 403."""
    from starlette.testclient import TestClient
    app = _build_test_app(monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/batch/decision-cards",
                        json={"patients": [{"patient_id": "pt-1"}]})
    assert r.status_code == 403
    assert "missing_fhir_context" in r.text


def test_endpoint_rejects_invalid_json(monkeypatch):
    """Bad JSON body -> 400 invalid_json (after passing SHARP)."""
    from starlette.testclient import TestClient
    app = _build_test_app(monkeypatch)
    headers = {
        "X-FHIR-Server-URL": "https://hapi.example.com/fhir",
        "X-FHIR-Access-Token": "tok-test",
    }
    with TestClient(app) as client:
        r = client.post("/api/batch/decision-cards",
                        headers=headers,
                        content=b"{not-json")
    assert r.status_code == 400


def test_endpoint_happy_path_returns_batch_response(monkeypatch):
    """Valid request -> 200 with BatchResponse JSON."""
    from starlette.testclient import TestClient
    app = _build_test_app(monkeypatch)
    headers = {
        "X-FHIR-Server-URL": "https://hapi.example.com/fhir",
        "X-FHIR-Access-Token": "tok-test",
    }
    body = {
        "patients": [
            {"patient_id": "pt-1", "demographics": {"age": 70}},
            {"patient_id": "pt-2", "demographics": {"age": 55}},
        ],
        "max_concurrency": 2,
    }
    with TestClient(app) as client:
        r = client.post("/api/batch/decision-cards", headers=headers, json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["n_requested"] == 2
    assert payload["n_succeeded"] == 2
    assert len(payload["results"]) == 2
    assert {res["patient_id"] for res in payload["results"]} == {"pt-1", "pt-2"}
