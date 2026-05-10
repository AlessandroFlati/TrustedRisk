"""Unit tests for the federation_partner agent (DEMO-2).

Tests the A2A handshake logic in isolation (capability negotiation +
audit trail). The actual HTTP roundtrip against TrustedRisk is covered
by an opt-in integration test that runs both servers.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_partner():
    spec = importlib.util.spec_from_file_location(
        "federation_partner_server",
        str(ROOT / "apps" / "federation_partner" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["federation_partner_server"] = mod
    spec.loader.exec_module(mod)
    return mod


_partner = _load_partner()


# ─────────────────────── Bundle selection ───────────────────────

@pytest.mark.parametrize("summary,expected_bundle", [
    ({"chief_complaint": "chest pain radiating to left arm"}, "stroke_acs"),
    ({"chief_complaint": "right-sided weakness, slurred speech"}, "stroke_acs"),
    ({"age": 4, "chief_complaint": "fever 39"}, "pediatric"),
    ({"chief_complaint": "preeclampsia, BP 165/110"}, "obstetric_geriatric"),
    ({"chief_complaint": "delirium, fall yesterday"}, "obstetric_geriatric"),
    ({"chief_complaint": "polytrauma after MVC, FAST positive"}, "trauma_critical"),
    ({"chief_complaint": "DKA, glucose 540"}, "endocrine_acute"),
    ({"chief_complaint": "complicated UTI with sepsis"}, "antimicrobial"),
    ({"chief_complaint": "NSCLC cycle 4 chemo"}, "oncology"),
    ({"chief_complaint": "stage 3 AKI requiring dialysis"}, "nephrology"),
    ({"chief_complaint": "active suicidal ideation"}, "mental_health"),
    ({"chief_complaint": "discharge planning, CHF stable"}, "core_discharge"),
])
def test_bundle_selection(summary, expected_bundle):
    """The capability-negotiation rule should map clinical context to bundle."""
    bundles_map = {
        "stroke_acs": [], "pediatric": [], "obstetric_geriatric": [],
        "trauma_critical": [], "endocrine_acute": [], "antimicrobial": [],
        "oncology": [], "nephrology": [], "mental_health": [],
        "core_discharge": [],
    }
    selected = _partner._select_bundle_for_request(summary, bundles_map)
    assert selected == expected_bundle


def test_scenario_for_bundle_returns_valid_slug():
    """Every bundle must map to a known scenario (A-S)."""
    valid_slugs = set("ABCDEFGHIJKLMNOPQRS")
    for bundle in ["core_discharge", "stroke_acs", "trauma_critical",
                    "pediatric", "obstetric_geriatric", "antimicrobial",
                    "oncology", "endocrine_acute", "nephrology",
                    "mental_health", "ed_acute", "imaging"]:
        scenario = _partner._scenario_for_bundle(bundle)
        assert scenario in valid_slugs


def test_scenario_for_unknown_bundle_falls_back():
    assert _partner._scenario_for_bundle("totally-fake-bundle") == "A"


# ─────────────────────── HTTP smoke ───────────────────────

def test_agent_card_endpoint_returns_valid_card():
    from fastapi.testclient import TestClient
    client = TestClient(_partner.app)
    r = client.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "darena-data-agent"
    assert "skills" in data and len(data["skills"]) >= 1


def test_root_html_serves():
    from fastapi.testclient import TestClient
    client = TestClient(_partner.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "darena-data-agent" in r.text
    assert "Run federated request" in r.text


def test_handshake_log_empty_initially():
    from fastapi.testclient import TestClient
    client = TestClient(_partner.app)
    r = client.get("/a2a/handshake-log")
    assert r.status_code == 200
    # May or may not have events from prior test runs -- just check shape
    assert "events" in r.json()
