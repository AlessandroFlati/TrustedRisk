"""COMPOSE-2 unit tests for the scheduler agent."""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest


@pytest.fixture()
def sched_mod():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "scheduler_agent_server",
        str(ROOT / "apps" / "scheduler_agent" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scheduler_agent_server"] = mod
    spec.loader.exec_module(mod)
    mod._reset_state()
    yield mod
    mod._reset_state()


def _card(*, action: str = "home_with_care",
            risk: float = 0.20, confidence: str = "high",
            tools: list | None = None,
            request_id: str = "req-001"):
    return {
        "recommendation": {"action": action, "confidence": confidence},
        "reasoning": {"risk_estimate": {"probability_mean": risk}},
        "audit": {"request_id": request_id, "tools": tools or []},
        "validation": {},
        "abstain": [],
    }


# ─────────────────────── agent-card / healthz ───────────────────────

def test_agent_card_served(sched_mod):
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        r = c.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "trustedrisk-scheduler-agent"
    skill_ids = {s["id"] for s in body["skills"]}
    assert "propose_followup_visits" in skill_ids


# ─────────────────────── Risk tier policy ───────────────────────

def test_high_risk_pcp_within_3_to_7_days(sched_mod):
    from fastapi.testclient import TestClient
    payload = {"decision_card": _card(risk=0.45),
                 "discharge_date": "2026-04-27"}
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits", json=payload)
    body = r.json()
    assert body["risk_tier"] == "high"
    pcp_visit = next(v for v in body["visits"]
                       if v["visit_type"] == "primary_care")
    assert pcp_visit["proposed_date"] == "2026-04-30"  # +3
    assert pcp_visit["deadline_date"] == "2026-05-04"  # +7


def test_moderate_risk_pcp_within_7_to_14_days(sched_mod):
    from fastapi.testclient import TestClient
    payload = {"decision_card": _card(risk=0.20),
                 "discharge_date": "2026-04-27"}
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits", json=payload)
    body = r.json()
    assert body["risk_tier"] == "moderate"
    pcp = next(v for v in body["visits"] if v["visit_type"] == "primary_care")
    assert pcp["proposed_date"] == "2026-05-04"
    assert pcp["deadline_date"] == "2026-05-11"


def test_routine_risk_pcp_within_14_to_30_days(sched_mod):
    from fastapi.testclient import TestClient
    payload = {"decision_card": _card(risk=0.05, confidence="high"),
                 "discharge_date": "2026-04-27"}
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits", json=payload)
    body = r.json()
    assert body["risk_tier"] == "routine"
    pcp = next(v for v in body["visits"] if v["visit_type"] == "primary_care")
    assert pcp["proposed_date"] == "2026-05-11"
    assert pcp["deadline_date"] == "2026-05-27"


def test_low_confidence_escalates_routine_to_moderate(sched_mod):
    from fastapi.testclient import TestClient
    payload = {"decision_card": _card(risk=0.05, confidence="low")}
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits", json=payload)
    assert r.json()["risk_tier"] == "moderate"


# ─────────────────────── Action handling ───────────────────────

def test_abstain_yields_no_followup(sched_mod):
    """A DecisionCard with no recommendation produces zero visits."""
    from fastapi.testclient import TestClient
    card = _card()
    card["recommendation"] = None
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": card})
    body = r.json()
    assert body["visits"] == []
    assert "abstain" in body["summary"].lower()


def test_continued_admission_no_outpatient_followup(sched_mod):
    from fastapi.testclient import TestClient
    payload = {"decision_card": _card(action="continued_admission")}
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits", json=payload)
    body = r.json()
    assert all(v["visit_type"] != "primary_care" for v in body["visits"])


def test_snf_yields_transition_visit(sched_mod):
    from fastapi.testclient import TestClient
    payload = {"decision_card": _card(action="snf"),
                 "discharge_date": "2026-04-27"}
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits", json=payload)
    body = r.json()
    visit_types = [v["visit_type"] for v in body["visits"]]
    assert "transition_visit" in visit_types
    transition = next(v for v in body["visits"]
                          if v["visit_type"] == "transition_visit")
    assert transition["proposed_date"] == "2026-05-04"  # +7


# ─────────────────────── Specialist inference ───────────────────────

def test_aki_tool_triggers_nephrology_visit(sched_mod):
    from fastapi.testclient import TestClient
    card = _card(tools=[{"tool": "compute_aki_kdigo_stage",
                            "outputs": {"stage": 2}}])
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": card})
    body = r.json()
    specialties = {v.get("specialty") for v in body["visits"]
                       if v["visit_type"] == "specialist"}
    assert "nephrology" in specialties


def test_dka_tool_triggers_endocrinology_visit(sched_mod):
    from fastapi.testclient import TestClient
    card = _card(tools=[{"tool": "compute_dka_severity"}])
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": card})
    body = r.json()
    assert any(v.get("specialty") == "endocrinology"
                  for v in body["visits"])


def test_suicide_risk_yields_telehealth_mental_health_visit(sched_mod):
    from fastapi.testclient import TestClient
    card = _card(tools=[{"tool": "compute_suicide_risk_assessment"}])
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": card})
    body = r.json()
    mh = next((v for v in body["visits"]
                  if v.get("specialty") == "mental_health"), None)
    assert mh is not None
    assert mh["modality"] == "telehealth"


def test_specialist_dedup(sched_mod):
    """Multiple tools mapping to the same specialty produce ONE specialist visit."""
    from fastapi.testclient import TestClient
    card = _card(tools=[
        {"tool": "compute_aki_kdigo_stage"},
        {"tool": "compute_dialysis_initiation_decision"},
    ])
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": card})
    body = r.json()
    nephro_visits = [v for v in body["visits"]
                       if v.get("specialty") == "nephrology"]
    assert len(nephro_visits) == 1


def test_specialist_accepts_string_tools(sched_mod):
    """audit.tool_trace is sometimes a list[str]; the agent must handle both."""
    from fastapi.testclient import TestClient
    card = _card()
    card["audit"] = {"request_id": "rid",
                       "tool_trace": ["compute_heart_score"]}
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": card})
    body = r.json()
    assert any(v.get("specialty") == "cardiology" for v in body["visits"])


# ─────────────────────── Validation + errors ───────────────────────

def test_rejects_missing_decision_card(sched_mod):
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits", json={})
    assert r.status_code == 400


def test_rejects_bad_discharge_date(sched_mod):
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": _card(),
                            "discharge_date": "not-a-date"})
    assert r.status_code == 400


def test_default_discharge_date_is_today(sched_mod):
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": _card()})
    body = r.json()
    # Server uses UTC date for the default; assert against UTC to avoid
    # local-vs-UTC rollover flakes.
    assert body["discharge_date"] == \
        datetime.now(timezone.utc).date().isoformat()


# ─────────────────────── History buffer ───────────────────────

def test_recent_proposals_buffer(sched_mod):
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        for i in range(3):
            c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": _card(request_id=f"r-{i}")})
        r = c.get("/a2a/skill/list_recent_proposals?limit=5")
    body = r.json()
    assert body["n"] == 3
    # Most recent first
    assert body["proposals"][0]["request_id"] == "r-2"


def test_recent_proposals_rejects_bad_limit(sched_mod):
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        r = c.get("/a2a/skill/list_recent_proposals?limit=999")
    assert r.status_code == 400


# ─────────────────────── References + structure ───────────────────────

def test_plan_includes_references(sched_mod):
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": _card()})
    body = r.json()
    assert any("Jack BW" in ref for ref in body["references"])
    assert any("Hansen" in ref for ref in body["references"])


def test_request_id_propagated(sched_mod):
    from fastapi.testclient import TestClient
    with TestClient(sched_mod.app) as c:
        r = c.post("/a2a/skill/propose_followup_visits",
                     json={"decision_card": _card(request_id="abc-123")})
    assert r.json()["request_id"] == "abc-123"
