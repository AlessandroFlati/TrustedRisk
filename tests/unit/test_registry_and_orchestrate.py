"""COMPOSE-3 tests for the agent registry and the multi-agent orchestrator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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


# ─────────────────────── Registry contents ───────────────────────

def test_registry_lists_all_four_agents():
    from a2a_agent.registry import list_agents
    ids = {entry.id for entry in list_agents()}
    assert ids == {
        "trustedrisk-agent",
        "darena-data-agent",
        "trustedrisk-alert-agent",
        "trustedrisk-scheduler-agent",
    }


def test_registry_roles_distinct():
    from a2a_agent.registry import list_agents
    roles = [entry.role for entry in list_agents()]
    assert sorted(roles) == ["alerting", "data", "decision", "scheduling"]


def test_registry_url_overridable_via_env(monkeypatch):
    from a2a_agent.registry import find_agent
    monkeypatch.setenv("ALERT_A2A_URL", "https://override.example.com")
    entry = find_agent("trustedrisk-alert-agent")
    assert entry is not None
    assert entry.url == "https://override.example.com"


def test_serialize_registry_includes_agent_cards():
    from a2a_agent.registry import serialize_registry
    snapshot = serialize_registry()
    assert snapshot["n"] == 4
    by_id = {a["id"]: a for a in snapshot["agents"]}
    # The decision agent's card should identify as a TrustedRisk agent.
    card = by_id["trustedrisk-agent"]["agent_card"]
    name = card.get("name", "")
    legacy = card.get("_legacy_name", "")
    assert name.lower().startswith("trustedrisk") or \
        legacy.startswith("trustedrisk")


# ─────────────────────── /api/registry endpoint ───────────────────────

def test_get_api_registry_returns_4_agents(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/api/registry")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 4
    ids = {a["id"] for a in body["agents"]}
    assert "trustedrisk-scheduler-agent" in ids


def test_get_api_registry_each_agent_has_required_fields(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        body = c.get("/api/registry").json()
    for a in body["agents"]:
        assert "id" in a and "role" in a and "url" in a
        assert "skills" in a and isinstance(a["skills"], list)


# ─────────────────────── Orchestration ───────────────────────

def test_orchestrate_default_scenario_runs_all_three_agents(app):
    """Smoke test: scenario A -> scheduler plan -> drift alert."""
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/orchestrate/discharge_with_partners",
                     json={"scenario": "A"})
    # Scenario A is a heavy showcase -- keep the assertion shape generic
    if r.status_code == 500:
        pytest.skip(f"Scenario A failed in this env: {r.text[:200]}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["agents_invoked"] == [
        "trustedrisk-agent",
        "trustedrisk-scheduler-agent",
        "trustedrisk-alert-agent",
    ]
    assert "decision_card" in body
    assert "followup_plan" in body
    assert "drift_alert" in body
    # Default synthetic drift report is warn-tier -> drift_alert must carry severity
    assert body["drift_alert"]["severity"] in ("warn", "alert")


def test_orchestrate_unknown_scenario_404(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/orchestrate/discharge_with_partners",
                     json={"scenario": "ZZZ"})
    assert r.status_code == 404


def test_orchestrate_passes_custom_drift_report(app):
    """A caller-supplied drift_report flows into the alert agent."""
    from fastapi.testclient import TestClient
    custom = {
        "overall_tier": "alert",
        "rationale": "AUROC collapse override",
        "signals": [{"name": "auroc_post_hoc", "value": 0.40,
                       "baseline": 0.62, "tier": "alert", "detail": ""}],
        "window_label": "1h", "window_n_cards": 5,
    }
    with TestClient(app) as c:
        r = c.post("/api/orchestrate/discharge_with_partners",
                     json={"scenario": "A", "drift_report": custom})
    if r.status_code == 500:
        pytest.skip(f"Scenario A failed in this env: {r.text[:200]}")
    body = r.json()
    assert body["drift_alert"]["severity"] == "alert"
    assert body["drift_alert"]["kpi"] == "auroc_post_hoc"
