"""Unit tests for COIN-1 (NL dialog) + COIN-3 (handshake skill discovery)."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent import coin as mod
from a2a_agent.coin import (
    _keyword_skill_score,
    dialog_with_partner,
    match_skill_to_prompt,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


def _scheduler_card() -> dict:
    return {
        "schemaVersion": "1.0.0",
        "name": "trustedrisk-scheduler-agent",
        "skills": [
            {
                "id": "propose_followup_visits",
                "name": "Propose follow-up visit plan from a DecisionCard",
                "description": "Plan PCP + specialist follow-up visits with "
                                  "AHRQ RED bundle timing.",
                "tags": ["scheduling", "follow-up", "transitions-of-care"],
                "examples": ["Plan follow-up visits for this discharge"],
            },
            {
                "id": "list_recent_proposals",
                "name": "List recent scheduling proposals",
                "description": "Recent follow-up plans dispatched.",
                "tags": ["audit", "scheduling"],
                "examples": ["Show me the last 10 follow-up plans"],
            },
        ],
    }


# ─────────────────────── COIN-3: Skill matching ───────────────────────

def test_keyword_score_zero_when_no_overlap():
    skill = {"id": "x", "name": "y", "description": "z",
                "tags": ["alpha"], "examples": []}
    assert _keyword_skill_score("xyzzy nothing matches", skill) == 0.0


def test_keyword_score_positive_with_overlap():
    skill = {"id": "propose_followup_visits",
                "name": "Propose follow-up visits",
                "description": "Plan follow-up appointments",
                "tags": [], "examples": []}
    score = _keyword_skill_score(
        "when should this patient come back for follow-up?",
        skill,
    )
    assert score > 0.0


def test_match_returns_top_skill():
    matches = match_skill_to_prompt(
        "when should this patient come back for follow-up?",
        _scheduler_card(),
    )
    assert matches
    assert matches[0].skill_id == "propose_followup_visits"


def test_match_orders_by_descending_score():
    matches = match_skill_to_prompt(
        "list the recent follow-up scheduling plans",
        _scheduler_card(),
    )
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_match_top_k_caps_results():
    matches = match_skill_to_prompt(
        "follow-up scheduling proposals",
        _scheduler_card(), top_k=1,
    )
    assert len(matches) == 1


def test_empty_prompt_yields_no_matches():
    assert match_skill_to_prompt("", _scheduler_card()) == []


def test_card_with_no_skills_yields_empty():
    assert match_skill_to_prompt("anything", {"skills": []}) == []


def test_match_uses_examples_field():
    """Skills with examples that contain the prompt tokens should rank high."""
    card = {"skills": [
        {"id": "skill_a", "name": "alpha", "description": "x",
         "examples": ["When should this patient come back?"]},
        {"id": "skill_b", "name": "beta", "description": "y", "examples": []},
    ]}
    matches = match_skill_to_prompt(
        "when should this patient come back?", card,
    )
    assert matches and matches[0].skill_id == "skill_a"


def test_real_scheduler_card_is_loadable():
    """Sanity: registry lookup + card load succeeds for the scheduler agent."""
    from a2a_agent.registry import find_agent
    entry = find_agent("trustedrisk-scheduler-agent")
    assert entry is not None
    card = entry.load_agent_card()
    assert card.get("name") == "trustedrisk-scheduler-agent"
    assert card.get("skills")


# ─────────────────────── COIN-1: end-to-end dialog ───────────────────────

def test_empty_prompt_returns_friendly_error():
    r = _run(dialog_with_partner(
        target_agent_id="trustedrisk-scheduler-agent", prompt=""))
    assert r.matched_skill_id is None
    assert "empty_prompt" in r.safety_warnings


def test_unknown_target_agent_handled():
    r = _run(dialog_with_partner(
        target_agent_id="does-not-exist", prompt="anything"))
    assert "unknown_target_agent" in r.safety_warnings


def test_no_skill_match_handled():
    """A prompt that doesn't overlap any skill returns no_skill_match."""
    r = _run(dialog_with_partner(
        target_agent_id="trustedrisk-alert-agent",
        prompt="xyzzy nonsense unrelated tokens only",
    ))
    assert "no_skill_match" in r.safety_warnings or \
           r.matched_skill_id is None


def test_scheduler_dialog_invokes_followup_skill():
    """End-to-end: prompt -> match propose_followup_visits -> invoke -> NL response."""
    r = _run(dialog_with_partner(
        target_agent_id="trustedrisk-scheduler-agent",
        prompt="when should this patient come back for a follow-up?",
        structured_inputs={
            "decision_card": {
                "recommendation": {"action": "home_with_care",
                                       "confidence": "medium"},
                "reasoning": {"risk_estimate": {"probability_mean": 0.30}},
                "audit": {"request_id": "req-001"},
                "validation": {}, "abstain": [],
            },
            "discharge_date": "2026-04-29",
        },
    ))
    assert r.matched_skill_id == "propose_followup_visits"
    assert r.structured_result is not None
    assert "visits" in r.structured_result
    # Deterministic NL template kicks in (LLM disabled by fixture)
    assert r.method == "deterministic_template"
    assert "follow-up visit" in r.nl_response.lower()


def test_alert_dialog_invokes_drift_skill():
    r = _run(dialog_with_partner(
        target_agent_id="trustedrisk-alert-agent",
        prompt="check the drift status now please",
        structured_inputs={
            "drift_report": {
                "overall_tier": "warn",
                "rationale": "ECE drift",
                "signals": [{"name": "ece_post_hoc", "value": 0.13,
                              "baseline": 0.08, "tier": "warn",
                              "detail": "drift"}],
                "window_label": "test", "window_n_cards": 0,
            }
        },
    ))
    assert r.matched_skill_id == "check_drift_now"
    assert r.structured_result is not None
    assert r.structured_result.get("severity") == "warn"
    assert "warn" in r.nl_response.lower() or \
           "calibration" in r.nl_response.lower()


# ─────────────────────── LLM path with mocked Ollama ───────────────────────

def test_llm_response_used_when_available(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama",
                          lambda _: "I propose 2 visits at moderate priority.")
    r = _run(dialog_with_partner(
        target_agent_id="trustedrisk-scheduler-agent",
        prompt="propose follow-up visits for this discharge",
        structured_inputs={
            "decision_card": {
                "recommendation": {"action": "home_with_care",
                                       "confidence": "medium"},
                "reasoning": {"risk_estimate": {"probability_mean": 0.20}},
                "audit": {}, "validation": {}, "abstain": [],
            },
        },
    ))
    assert r.matched_skill_id == "propose_followup_visits"
    assert r.method == "llm"
    assert "moderate priority" in r.nl_response


def test_llm_failure_falls_back_to_deterministic(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", lambda _: None)
    r = _run(dialog_with_partner(
        target_agent_id="trustedrisk-alert-agent",
        prompt="check drift",
        structured_inputs={"drift_report": {
            "overall_tier": "ok", "rationale": "ok",
            "signals": [], "window_label": "t", "window_n_cards": 0,
        }},
    ))
    assert r.method == "deterministic_template"
    assert "nominal" in r.nl_response.lower() or \
           "no alerts" in r.nl_response.lower()


# ─────────────────────── Invoker error path ───────────────────────

def test_invoker_error_surfaced_safely():
    """If the invoked skill raises, the error appears in safety_warnings
    and the NL response explains the failure -- never crashes."""
    r = _run(dialog_with_partner(
        target_agent_id="trustedrisk-scheduler-agent",
        prompt="follow-up please",
        # Missing decision_card -> propose_followup_visits raises 400
        structured_inputs={},
    ))
    assert any(w.startswith("invoker_error") for w in r.safety_warnings)


# ─────────────────────── /api/coin/dialog endpoint ───────────────────────

@pytest.fixture(scope="module")
def app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod_app = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod_app
    spec.loader.exec_module(mod_app)
    return mod_app.app


def test_endpoint_routes_to_coin_dialog(app):
    from fastapi.testclient import TestClient
    body = {
        "target_agent_id": "trustedrisk-alert-agent",
        "prompt": "check drift status",
        "structured_inputs": {"drift_report": {
            "overall_tier": "ok", "rationale": "ok",
            "signals": [], "window_label": "t", "window_n_cards": 0,
        }},
    }
    with TestClient(app) as c:
        r = c.post("/api/coin/dialog", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["target_agent_id"] == "trustedrisk-alert-agent"
    assert payload["matched_skill_id"] == "check_drift_now"


def test_endpoint_rejects_missing_target(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/coin/dialog",
                     json={"prompt": "anything"})
    assert r.status_code == 400


def test_endpoint_rejects_empty_prompt(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/coin/dialog", json={
            "target_agent_id": "trustedrisk-alert-agent", "prompt": "  "})
    assert r.status_code == 400
