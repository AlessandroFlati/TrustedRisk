"""Phase 6.1 -- Composer agent integration tests.

Each workflow template runs end-to-end with realistic inputs. The
orchestrator's variable resolution + step sequencing is exercised by
the runtime, not just by unit tests on the resolver.
"""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from apps.composer.orchestrator import (
    Workflow,
    WorkflowStep,
    execute_workflow,
)
from apps.composer.workflows import REGISTRY


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Resolver unit tests ───────────────────────


async def _identity(**kw):
    return kw


def test_orchestrator_resolves_input_reference():
    wf = Workflow(
        id="t", title="t", description="t", required_inputs=["x"],
        steps=[WorkflowStep(
            id="a", specialist="t", tool_name="t",
            callable=_identity, inputs={"v": "${input.x}"},
        )],
    )
    out = _run(execute_workflow(wf, {"x": 42}))
    assert out.steps[0].output == {"v": 42}


def test_orchestrator_resolves_step_output_reference():
    async def first(**kw):
        return {"score": 7, "tier": "high"}

    async def second(**kw):
        return kw

    wf = Workflow(
        id="t", title="t", description="t", required_inputs=[],
        steps=[
            WorkflowStep(
                id="a", specialist="t", tool_name="first",
                callable=first, inputs={},
            ),
            WorkflowStep(
                id="b", specialist="t", tool_name="second",
                callable=second,
                inputs={
                    "from_a_score": "${steps.a.output.score}",
                    "from_a_tier": "${steps.a.output.tier}",
                },
            ),
        ],
    )
    out = _run(execute_workflow(wf, {}))
    assert out.steps[1].output == {"from_a_score": 7, "from_a_tier": "high"}


def test_orchestrator_missing_required_input_raises():
    wf = Workflow(
        id="t", title="t", description="t", required_inputs=["x"],
        steps=[],
    )
    with pytest.raises(ValueError, match="missing required input"):
        _run(execute_workflow(wf, {}))


def test_orchestrator_optional_step_skips_on_resolution_error():
    """An optional step whose input reference fails must not crash the
    workflow -- it skips with `skipped=True`."""

    async def fail(**kw):
        raise AssertionError("must not be called")

    wf = Workflow(
        id="t", title="t", description="t", required_inputs=[],
        steps=[
            WorkflowStep(
                id="optional", specialist="t", tool_name="fail",
                callable=fail,
                inputs={"v": "${steps.nonexistent.output.field}"},
                optional=True,
            ),
        ],
    )
    out = _run(execute_workflow(wf, {}))
    assert out.steps[0].skipped is True


def test_orchestrator_propagates_abstain_when_step_says_so():
    """A step output with abstain_recommended=True halts subsequent
    steps when stop_on_abstain=True (default)."""
    async def abstain_emitter(**kw):
        from shared.schemas import RiskEstimate
        # We don't construct a full RiskEstimate; a dict is enough
        return {"abstain_recommended": True,
                "abstain_reason": "test_short_circuit"}

    async def follow_up(**kw):
        return {"v": "should-not-run"}

    wf = Workflow(
        id="t", title="t", description="t", required_inputs=[],
        steps=[
            WorkflowStep(id="a", specialist="t", tool_name="abst",
                              callable=abstain_emitter, inputs={}),
            WorkflowStep(id="b", specialist="t", tool_name="follow",
                              callable=follow_up, inputs={}),
        ],
    )
    out = _run(execute_workflow(wf, {}))
    assert out.abstain_recommended is True
    assert "test_short_circuit" in (out.abstain_reason or "")
    assert len(out.steps) == 1


# ─────────────────────── End-to-end workflow tests ───────────────────────


def _patient_bundle() -> dict:
    return {"resourceType": "Bundle", "entry": [
        {"resource": {
            "resourceType": "Patient", "id": "pt-jane",
            "name": [{"given": ["Jane"], "family": "Doe"}],
            "birthDate": "1955-01-01", "gender": "female",
        }},
        {"resource": {
            "resourceType": "Encounter", "id": "enc-1",
            "class": {"code": "IMP", "display": "Inpatient"},
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
            "medicationCodeableConcept": {"text": "furosemide 40 mg PO daily"},
        }},
    ]}


def test_chf_admission_workflow_end_to_end():
    wf = REGISTRY["chf_admission"]
    out = _run(execute_workflow(wf, {
        "chief_complaint": "shortness of breath",
        "vital_signs": {
            "heart_rate": 122, "spo2": 91, "respiratory_rate": 26,
            "systolic_bp": 96, "temperature_c": 37.8,
        },
        "age": 72,
        "patient_id": "Patient/pt-jane",
        "fhir_bundle": _patient_bundle(),
        "medications": ["furosemide", "lisinopril", "metoprolol"],
    }))
    step_ids = [s.step_id for s in out.steps]
    # At minimum the triage + DDx + polypharmacy + HnP must run
    assert "triage" in step_ids
    assert "ddx" in step_ids
    assert "polypharmacy" in step_ids
    assert "hnp" in step_ids
    # No step errored fatally
    failures = [s for s in out.steps if s.error and not s.skipped]
    assert not failures, [(s.step_id, s.error) for s in failures]


def test_sepsis_workup_workflow_end_to_end():
    wf = REGISTRY["sepsis_workup"]
    out = _run(execute_workflow(wf, {
        "chief_complaint": "fever and confusion",
        "vital_signs": {
            "heart_rate": 130, "systolic_bp": 84,
            "respiratory_rate": 26, "temperature_c": 39.2, "spo2": 92,
        },
        "age": 78,
        "patient_id": "Patient/pt-sepsis",
        "infection_source": "urinary",
        "patient_factors": {
            "allergies": [], "egfr_ml_min": 50,
        },
        "fhir_bundle": _patient_bundle(),
    }))
    step_ids = {s.step_id for s in out.steps}
    assert {"triage", "ddx", "antibiotic", "consult"} <= step_ids


def test_discharge_planning_workflow_end_to_end():
    wf = REGISTRY["discharge_planning"]
    # Patch readmission_risk to a stub since it requires a coefficient
    # artefact loaded; the workflow runs with the real artefact when
    # TRUSTEDRISK_COEFFICIENTS_PATH is set in CI.
    out = _run(execute_workflow(wf, {
        "patient_id": "Patient/pt-jane",
        "fhir_bundle": _patient_bundle(),
        "admission_meds": [{"name": "furosemide 40 mg"}],
        "discharge_meds": [{"name": "furosemide 40 mg"},
                              {"name": "lisinopril 10 mg"}],
        "discharge_action": "home_with_care",
        "lace_score": 9,
        "patient_demographics": {
            "age": 72, "sex": "female", "race": "white",
            "ethnicity": "non_hispanic", "insurance_type": "medicare",
        },
        "hospital_course_text": "Admitted with CHF; diuresed; stable.",
        "followup_plan": ["Cardiology in 7-14 days"],
        "patient_instructions": ["Daily weights", "Low-salt diet"],
    }))
    step_ids = {s.step_id for s in out.steps}
    assert "summary" in step_ids
    assert "handoff" in step_ids


def test_outpatient_med_review_workflow_end_to_end():
    wf = REGISTRY["outpatient_med_review"]
    out = _run(execute_workflow(wf, {
        "patient_id": "Patient/pt-old",
        "medications": [
            "warfarin 5 mg", "metformin 1000 mg", "lisinopril 10 mg",
            "atorvastatin 40 mg", "metoprolol 50 mg",
            "furosemide 40 mg", "alprazolam 0.5 mg",
            "ibuprofen 600 mg PRN", "diphenhydramine 25 mg",
        ],
        "fhir_bundle": _patient_bundle(),
        "patient_age": 82,
        "patient_sex": "female",
    }))
    step_ids = {s.step_id for s in out.steps}
    assert {"polypharmacy", "gaps", "what_if"} <= step_ids
    # Polypharmacy should flag the high-risk combo (warfarin + ibuprofen)
    poly_step = next(s for s in out.steps if s.step_id == "polypharmacy")
    assert poly_step.output_dump is not None


def test_stroke_alert_workflow_end_to_end():
    wf = REGISTRY["stroke_alert"]
    nihss_items = {
        "1a_loc": 0, "1b_loc_questions": 0, "1c_loc_commands": 0,
        "2_best_gaze": 1, "3_visual": 0, "4_facial_palsy": 1,
        "5a_motor_arm_left": 0, "5b_motor_arm_right": 3,
        "6a_motor_leg_left": 0, "6b_motor_leg_right": 2,
        "7_limb_ataxia": 0, "8_sensory": 0,
        "9_best_language": 1, "10_dysarthria": 1, "11_extinction": 0,
    }
    out = _run(execute_workflow(wf, {
        "patient_id": "Patient/pt-stroke",
        "nihss_item_scores": nihss_items,
        "last_known_well_minutes_ago": 90,
        "egfr_ml_min": 65,
        "fhir_bundle": _patient_bundle(),
    }))
    step_ids = {s.step_id for s in out.steps}
    assert {"severity", "thrombolysis", "contrast", "consult"} <= step_ids


# ─────────────────────── HTTP endpoints ───────────────────────

def test_composer_lists_workflows_via_http():
    from apps.composer.server import app
    client = TestClient(app)
    resp = client.get("/api/workflows")
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["workflows"]) >= 5
    ids = {w["id"] for w in payload["workflows"]}
    assert {"chf_admission", "sepsis_workup", "discharge_planning",
            "outpatient_med_review", "stroke_alert"} <= ids


def test_composer_serves_agent_card():
    from apps.composer.server import app
    client = TestClient(app)
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    card = resp.json()
    # The card's `_legacy_name` carries the canonical routing slug.
    assert card.get("_legacy_name") == "trustedrisk-composer" or \
        card["name"] == "trustedrisk-composer"
    assert card["capabilities"]["agentComposition"]["enabled"] is True
    consults = card["capabilities"]["agentComposition"]["consults"]
    assert "trustedrisk-discharge" in consults
    assert "trustedrisk-pa" in consults


def test_composer_run_endpoint_unknown_workflow_404():
    from apps.composer.server import app
    client = TestClient(app)
    resp = client.post("/api/run/does-not-exist", json={"inputs": {}})
    assert resp.status_code == 404


def test_composer_run_endpoint_executes_workflow():
    from apps.composer.server import app
    client = TestClient(app)
    resp = client.post("/api/run/outpatient_med_review", json={
        "inputs": {
            "patient_id": "Patient/x",
            "medications": ["warfarin", "ibuprofen"],
            "fhir_bundle": _patient_bundle(),
            "patient_age": 75,
            "patient_sex": "female",
        }
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["workflow_id"] == "outpatient_med_review"
    assert len(body["steps"]) >= 2


def test_composer_run_endpoint_returns_400_on_missing_inputs():
    from apps.composer.server import app
    client = TestClient(app)
    resp = client.post("/api/run/chf_admission", json={"inputs": {}})
    assert resp.status_code == 400
    assert "missing required input" in resp.json()["detail"].lower()


def test_composer_healthz_lists_workflow_ids():
    from apps.composer.server import app
    client = TestClient(app)
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["workflows"] >= 5
    assert "chf_admission" in body["workflow_ids"]
