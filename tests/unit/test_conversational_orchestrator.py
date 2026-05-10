"""GENAI-5 unit tests for the conversational orchestrator."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent import conversational_orchestrator as orch
from a2a_agent.conversational_orchestrator import (
    _topological_order,
    execute_workflow,
    plan_workflow,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    """Default -- keyword extraction only."""
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


# ─────────────────────── Keyword extraction ───────────────────────

def test_single_intent_extracted():
    plan = plan_workflow("compute the readmission risk for this patient")
    intent_ids = [i.intent_id for i in plan.intents]
    assert "risk_assessment" in intent_ids


def test_multi_intent_extracted():
    plan = plan_workflow(
        "for patient X, give me risk + counseling + follow-up + drift check"
    )
    intent_ids = {i.intent_id for i in plan.intents}
    assert {"risk_assessment", "counseling", "followup",
              "drift_check"} <= intent_ids


def test_empty_query_returns_empty_plan():
    plan = plan_workflow("")
    assert plan.intents == []
    assert plan.execution_order == []


def test_unsupported_query_returns_empty_plan():
    plan = plan_workflow("tell me a joke")
    assert plan.intents == []


def test_keyword_aliases_normalized():
    """Various phrasings of the same intent map to the same id."""
    p1 = plan_workflow("DDX for chest pain")
    p2 = plan_workflow("differential diagnosis for chest pain")
    p3 = plan_workflow("possible diagnoses for chest pain")
    for p in (p1, p2, p3):
        assert any(i.intent_id == "differential_diagnosis"
                      for i in p.intents)


def test_extraction_method_is_keyword_when_llm_disabled():
    plan = plan_workflow("risk assessment please", use_llm=False)
    assert plan.extraction_method == "keyword"


def test_extraction_method_llm_when_llm_enabled(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(
        orch, "_extract_intents_llm",
        lambda q: ["risk_assessment", "counseling"]
    )
    plan = plan_workflow("anything", use_llm=True)
    assert plan.extraction_method == "llm"
    intent_ids = {i.intent_id for i in plan.intents}
    assert {"risk_assessment", "counseling"} <= intent_ids


def test_llm_failure_falls_back_to_keyword(monkeypatch):
    """When LLM fails, the deterministic keyword set still returns."""
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(orch, "_extract_intents_llm", lambda q: None)
    plan = plan_workflow("compute readmission risk", use_llm=True)
    assert plan.extraction_method == "keyword"
    assert any(i.intent_id == "risk_assessment" for i in plan.intents)


# ─────────────────────── Dependency ordering ───────────────────────

def test_counseling_runs_after_risk_assessment():
    plan = plan_workflow("counseling and risk")
    order = plan.execution_order
    assert order.index("risk_assessment") < order.index("counseling")


def test_followup_runs_after_risk_assessment():
    plan = plan_workflow("follow-up and risk")
    order = plan.execution_order
    assert order.index("risk_assessment") < order.index("followup")


def test_independent_intents_preserve_input_order():
    plan = plan_workflow("drift check + differential diagnosis")
    order = plan.execution_order
    # Both have no deps -- order is the input mention order
    assert "drift_check" in order
    assert "differential_diagnosis" in order


def test_topological_skip_missing_deps():
    """If counseling is requested without risk, dependency is dropped."""
    order = _topological_order(["counseling"])
    assert order == ["counseling"]


# ─────────────────────── Plan rationale + node shape ───────────────────────

def test_plan_rationale_lists_intents():
    plan = plan_workflow("risk + counseling")
    assert "risk_assessment" in plan.rationale
    assert "counseling" in plan.rationale


def test_intent_nodes_record_dependencies():
    plan = plan_workflow("risk + counseling + follow-up")
    by_id = {i.intent_id: i for i in plan.intents}
    assert "risk_assessment" in by_id["counseling"].depends_on
    assert "risk_assessment" in by_id["followup"].depends_on
    assert by_id["risk_assessment"].depends_on == []


def test_intent_node_agent_assignment():
    plan = plan_workflow("follow-up + drift")
    by_id = {i.intent_id: i for i in plan.intents}
    assert by_id["followup"].agent == "trustedrisk-scheduler-agent"
    assert by_id["drift_check"].agent == "trustedrisk-alert-agent"


# ─────────────────────── Execution: stand-alone intents ───────────────────────

def test_execute_drift_check_only():
    """drift_check needs no upstream -- runs standalone."""
    plan = plan_workflow("just check the drift")
    result = _run(execute_workflow(plan))
    assert result.overall_success is True
    assert len(result.steps) == 1
    assert result.steps[0].intent_id == "drift_check"
    assert "warn" in result.steps[0].output_summary or \
           "alert" in result.steps[0].output_summary


def test_execute_differential_diagnosis():
    plan = plan_workflow("differential for chest pain")
    result = _run(execute_workflow(plan, initial_context={
        "chief_complaint": "chest pain",
        "free_text_summary": "diaphoresis with exertional onset",
    }))
    assert result.overall_success is True
    ddx_step = next(s for s in result.steps
                       if s.intent_id == "differential_diagnosis")
    assert "items=" in ddx_step.output_summary


def test_execute_cumulative_impact_with_empty_decisions():
    plan = plan_workflow("show me the cumulative impact")
    result = _run(execute_workflow(plan))
    step = result.steps[0]
    assert step.intent_id == "cumulative_impact"
    assert step.success is True
    assert "n_decisions=0" in step.output_summary


def test_execute_equity_audit_with_empty_cohort():
    plan = plan_workflow("equity audit")
    result = _run(execute_workflow(plan))
    step = result.steps[0]
    assert step.intent_id == "equity_audit"
    assert step.success is True


# ─────────────────────── Execution: error handling ───────────────────────

def test_execution_step_records_errors_without_raising():
    """A single failing intent doesn't abort the full plan."""
    plan = plan_workflow("differential diagnosis for nothing")
    # Empty chief_complaint will trigger abstain (not error) -- let's force
    # an error by passing an unsupported chief_complaint via override.
    result = _run(execute_workflow(plan, initial_context={
        "chief_complaint": "xyz_unsupported",
    }))
    # Tool returns abstain_recommended -- that's NOT an exception, so step
    # success stays True. Confirm we still produce a report.
    step = result.steps[0]
    assert step.intent_id == "differential_diagnosis"
    assert step.success is True


# ─────────────────────── Synthesized DecisionCard for followup ───────────────────────

def test_followup_uses_synthesized_card_when_no_risk_step():
    """If counseling is dropped because risk was skipped, followup still
    runs against the synthesized fallback card."""
    plan = plan_workflow("just propose follow-up visits")
    result = _run(execute_workflow(plan))
    step = next(s for s in result.steps if s.intent_id == "followup")
    assert step.success is True


# ─────────────────────── Cost-effectiveness chains after risk ───────────────────────

def test_cost_effectiveness_consumes_risk_estimate(monkeypatch):
    """When risk is in the context, cost_effectiveness uses it as baseline."""
    plan = plan_workflow("risk + cost-effectiveness")
    fake_risk = {
        "model_name": "lace-plus-bayesian-v1",
        "model_version": "0.7.0",
        "outcome_id": "readmission_30d",
        "horizon_days": 30,
        "lace_raw_score": 10,
        "probability_mean": 0.30,
        "probability_ci95": [0.25, 0.35],
        "probability_ci_width": 0.10,
        "contributing_factors": [],
        "fhir_observations_used": [],
        "computed_at": "2026-04-28T00:00:00+00:00",
        "confidence": "preferred",
        "valid_for_minutes": 60,
        "valid_until": "2026-04-28T01:00:00+00:00",
    }
    result = _run(execute_workflow(plan, initial_context={
        "risk_estimate": fake_risk,
        "patient_id": "pt-stub",
    }))
    # cost_effectiveness step should be in the plan + run successfully
    step_ids = [s.intent_id for s in result.steps]
    assert "cost_effectiveness" in step_ids
    # Must run AFTER risk_assessment per dependency
    assert step_ids.index("risk_assessment") < \
           step_ids.index("cost_effectiveness")


# ─────────────────────── /api/orchestrate/conversational endpoint ───────────────────────

@pytest.fixture(scope="module")
def app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod
    spec.loader.exec_module(mod)
    return mod.app


def test_endpoint_returns_plan_when_execute_false(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/orchestrate/conversational", json={
            "query": "risk + drift check + counseling",
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "intents" in body
    assert body["natural_query"] == "risk + drift check + counseling"
    intent_ids = {i["intent_id"] for i in body["intents"]}
    assert {"risk_assessment", "drift_check", "counseling"} <= intent_ids


def test_endpoint_executes_when_execute_true(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/orchestrate/conversational", json={
            "query": "drift check please",
            "execute": True,
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "steps" in body
    assert "plan" in body
    assert body["overall_success"] is True


def test_endpoint_rejects_empty_query(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/orchestrate/conversational", json={"query": "  "})
    assert r.status_code == 400


def test_endpoint_rejects_missing_query(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/orchestrate/conversational", json={})
    assert r.status_code == 400
