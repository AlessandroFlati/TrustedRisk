"""IMPACT-2 endpoint tests for the playground /api/impact/cumulative route."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))

    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod
    spec.loader.exec_module(mod)
    return mod.app


def _card(*, action: str | None = "home_with_care", risk: float = 0.30):
    return {
        "recommendation": ({"action": action, "confidence": "high"}
                              if action else None),
        "reasoning": {"risk_estimate": {"probability_mean": risk}},
        "validation": {},
        "audit": {"tools": []},
        "abstain": [] if action else [{"trigger": "abstain"}],
        "self_critique": None,
    }


def test_get_cumulative_returns_kpis_shape(app):
    """GET with empty archive returns a well-formed (zeroed) KPIs payload."""
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/api/impact/cumulative")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "n_decisions" in body
    assert "estimated_cost_avoided_usd" in body
    assert "intervention_relative_risk_reduction" in body
    assert body["intervention_relative_risk_reduction"] == 0.25  # default


def test_post_cumulative_with_explicit_decisions(app):
    """POST with a decisions list aggregates without reading the archive."""
    from fastapi.testclient import TestClient
    payload = {
        "decisions": [
            _card(action="home_with_care", risk=0.40),
            _card(action="discharge_home", risk=0.10),
            _card(action="snf", risk=0.50),
        ],
        "intervention_rrr": 0.30,
        "avoided_event_cost_usd": 12_000.0,
    }
    with TestClient(app) as c:
        r = c.post("/api/impact/cumulative", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_decisions"] == 3
    assert body["n_with_recommendation"] == 3
    assert body["action_counts"]["home_with_care"] == 1
    assert body["action_counts"]["snf"] == 1
    assert body["action_counts"]["discharge_home"] == 1
    # Two interventions: events avoided = (0.40 + 0.50) * 0.30 = 0.27
    assert abs(body["estimated_events_avoided"] - 0.27) < 1e-6
    assert abs(body["estimated_cost_avoided_usd"] - 0.27 * 12_000.0) < 1e-3


def test_post_cumulative_rejects_non_list(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/impact/cumulative",
                     json={"decisions": "not-a-list"})
    assert r.status_code == 400


def test_get_with_custom_rrr_query(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/api/impact/cumulative?intervention_rrr=0.40")
    assert r.status_code == 200
    assert r.json()["intervention_relative_risk_reduction"] == 0.40
