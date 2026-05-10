"""COMPOSE-1 unit tests for the alert agent (subscriptions + drift dispatch)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def alert_mod():
    """Import the alert_agent.server module fresh per test (state is module-global)."""
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))

    spec = importlib.util.spec_from_file_location(
        "alert_agent_server",
        str(ROOT / "apps" / "alert_agent" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alert_agent_server"] = mod
    spec.loader.exec_module(mod)
    mod._reset_state()
    yield mod
    mod._reset_state()


# ─────────────────────── agent-card + healthz ───────────────────────

def test_agent_card_served(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "trustedrisk-alert-agent"
    skills = {s["id"] for s in data["skills"]}
    assert {"subscribe_to_alerts", "check_drift_now",
              "list_recent_alerts"} <= skills


def test_healthz_reflects_state(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["subscriptions"] == 0
    assert body["recent_alerts"] == 0


# ─────────────────────── subscribe ───────────────────────

def test_subscribe_then_list(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.post("/a2a/skill/subscribe_to_alerts",
                     json={"webhook_url": "https://ops.example.com/hook"})
        assert r.status_code == 200
        sub_id = r.json()["subscription_id"]
        listing = c.get("/a2a/skill/subscribe_to_alerts").json()
    assert listing["n"] == 1
    assert listing["subscriptions"][0]["subscription_id"] == sub_id
    assert listing["subscriptions"][0]["severity_threshold"] == "warn"


def test_subscribe_rejects_non_http_url(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.post("/a2a/skill/subscribe_to_alerts",
                     json={"webhook_url": "ftp://nope"})
    assert r.status_code == 400


def test_subscribe_rejects_bad_severity(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.post("/a2a/skill/subscribe_to_alerts",
                     json={"webhook_url": "https://ops.example.com/hook",
                            "severity_threshold": "FATAL"})
    assert r.status_code == 400


def test_subscribe_rejects_severity_ok(alert_mod):
    """severity=ok would deliver every poll -- explicit ban."""
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.post("/a2a/skill/subscribe_to_alerts",
                     json={"webhook_url": "https://ops.example.com/hook",
                            "severity_threshold": "ok"})
    assert r.status_code == 400


def test_unsubscribe_round_trip(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        sub_id = c.post("/a2a/skill/subscribe_to_alerts",
                          json={"webhook_url": "https://x/hook"}
                          ).json()["subscription_id"]
        r = c.delete(f"/a2a/skill/subscribe_to_alerts/{sub_id}")
        assert r.status_code == 200
        listing = c.get("/a2a/skill/subscribe_to_alerts").json()
    assert listing["n"] == 0


def test_unsubscribe_unknown_id_404(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.delete("/a2a/skill/subscribe_to_alerts/nonexistent")
    assert r.status_code == 404


# ─────────────────────── check_drift_now ───────────────────────

def test_check_drift_now_ok_tier_emits_no_alert(alert_mod):
    from fastapi.testclient import TestClient
    report = {"overall_tier": "ok",
                "rationale": "calibration nominal",
                "signals": [], "window_label": "7d", "window_n_cards": 50}
    with TestClient(alert_mod.app) as c:
        c.post("/a2a/skill/subscribe_to_alerts",
                 json={"webhook_url": "https://ops.example.com/hook"})
        r = c.post("/a2a/skill/check_drift_now",
                     json={"drift_report": report})
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "ok"
    assert body["alert"] is None
    assert body["deliveries"] == []


def test_check_drift_now_warn_tier_dispatches(alert_mod, monkeypatch):
    """A warn-tier report must trigger webhook delivery to all warn-threshold subscribers."""
    from fastapi.testclient import TestClient

    delivered: list[dict] = []

    async def _fake_delivery(url, payload):
        delivered.append({"url": url, "payload": payload})
        return {"webhook_url": url, "status_code": 200,
                "delivered": True, "error": None,
                "started_at": "2026-04-27T00:00:00+00:00"}

    monkeypatch.setattr(alert_mod, "_deliver_webhook", _fake_delivery)

    report = {
        "overall_tier": "warn",
        "rationale": "ECE drifted from 0.08 to 0.14 over 7d",
        "signals": [
            {"name": "ece_post_hoc", "value": 0.14, "baseline": 0.08,
             "tier": "warn", "detail": "ECE breach"}
        ],
        "window_label": "7d", "window_n_cards": 80,
    }
    with TestClient(alert_mod.app) as c:
        c.post("/a2a/skill/subscribe_to_alerts",
                 json={"webhook_url": "https://ops.example.com/hook",
                        "severity_threshold": "warn"})
        # An alert-only subscriber should NOT receive the warn-tier dispatch
        c.post("/a2a/skill/subscribe_to_alerts",
                 json={"webhook_url": "https://pager.example.com/critical",
                        "severity_threshold": "alert"})
        r = c.post("/a2a/skill/check_drift_now",
                     json={"drift_report": report})

    assert r.status_code == 200
    body = r.json()
    assert body["severity"] == "warn"
    assert body["kpi"] == "ece_post_hoc"
    assert len(body["deliveries"]) == 1   # only the warn subscriber
    assert body["deliveries"][0]["delivered"] is True

    assert len(delivered) == 1
    assert delivered[0]["url"] == "https://ops.example.com/hook"
    assert delivered[0]["payload"]["severity"] == "warn"


def test_check_drift_now_alert_tier_reaches_both_subscribers(alert_mod,
                                                                  monkeypatch):
    from fastapi.testclient import TestClient

    delivered: list[str] = []

    async def _fake_delivery(url, payload):
        delivered.append(url)
        return {"webhook_url": url, "status_code": 200, "delivered": True,
                "error": None, "started_at": "2026-04-27T00:00:00+00:00"}

    monkeypatch.setattr(alert_mod, "_deliver_webhook", _fake_delivery)

    report = {"overall_tier": "alert",
                "rationale": "AUROC collapsed",
                "signals": [
                    {"name": "auroc_post_hoc", "value": 0.45, "baseline": 0.62,
                     "tier": "alert", "detail": "AUROC < 0.5"}
                ],
                "window_label": "24h", "window_n_cards": 25}

    with TestClient(alert_mod.app) as c:
        c.post("/a2a/skill/subscribe_to_alerts",
                 json={"webhook_url": "https://warn.example.com",
                        "severity_threshold": "warn"})
        c.post("/a2a/skill/subscribe_to_alerts",
                 json={"webhook_url": "https://alert.example.com",
                        "severity_threshold": "alert"})
        r = c.post("/a2a/skill/check_drift_now",
                     json={"drift_report": report})

    body = r.json()
    assert body["severity"] == "alert"
    assert len(body["deliveries"]) == 2
    assert set(delivered) == {"https://warn.example.com",
                                  "https://alert.example.com"}


def test_check_drift_now_pulls_from_trustedrisk_when_no_body(alert_mod,
                                                                  monkeypatch):
    """When the request body has no drift_report, the agent fetches from TrustedRisk."""
    from fastapi.testclient import TestClient

    async def _fake_fetch():
        return {"overall_tier": "warn", "rationale": "fetched",
                "signals": [{"name": "fetched_kpi", "value": 1.0,
                              "baseline": 0.0, "tier": "warn", "detail": "ok"}],
                "window_label": "auto", "window_n_cards": 0}

    async def _fake_delivery(url, payload):
        return {"webhook_url": url, "status_code": 200, "delivered": True,
                "error": None, "started_at": "2026-04-27T00:00:00+00:00"}

    monkeypatch.setattr(alert_mod, "_fetch_drift_report", _fake_fetch)
    monkeypatch.setattr(alert_mod, "_deliver_webhook", _fake_delivery)

    with TestClient(alert_mod.app) as c:
        c.post("/a2a/skill/subscribe_to_alerts",
                 json={"webhook_url": "https://x.example.com"})
        r = c.post("/a2a/skill/check_drift_now")

    body = r.json()
    assert body["severity"] == "warn"
    assert body["kpi"] == "fetched_kpi"


# ─────────────────────── recent alerts buffer ───────────────────────

def test_alerts_buffer_keeps_history(alert_mod, monkeypatch):
    from fastapi.testclient import TestClient

    async def _fake_delivery(url, payload):
        return {"webhook_url": url, "status_code": 200, "delivered": True,
                "error": None, "started_at": "2026-04-27T00:00:00+00:00"}

    monkeypatch.setattr(alert_mod, "_deliver_webhook", _fake_delivery)

    with TestClient(alert_mod.app) as c:
        c.post("/a2a/skill/subscribe_to_alerts",
                 json={"webhook_url": "https://x.example.com"})
        for i in range(3):
            c.post("/a2a/skill/check_drift_now", json={
                "drift_report": {
                    "overall_tier": "warn",
                    "rationale": f"drift #{i}",
                    "signals": [{"name": f"k{i}", "value": 1.0, "baseline": 0,
                                  "tier": "warn", "detail": ""}],
                    "window_label": "7d", "window_n_cards": 10,
                }
            })
        r = c.get("/a2a/alerts/recent?limit=5")

    body = r.json()
    assert body["n"] == 3
    # Most recent first
    assert body["alerts"][0]["kpi"] == "k2"
    assert body["alerts"][2]["kpi"] == "k0"


def test_alerts_recent_rejects_bad_limit(alert_mod):
    from fastapi.testclient import TestClient
    with TestClient(alert_mod.app) as c:
        r = c.get("/a2a/alerts/recent?limit=0")
    assert r.status_code == 400
